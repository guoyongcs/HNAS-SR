import torch
import torch.nn as nn
import torch.nn.functional as F
from model.operationsbn import *
from model.genotypes import COMPACT_PRIMITIVES, COMPACT_PRIMITIVES_UPSAMPLING
from model.genotypes import Genotype
import model.utils as utils
import numpy as np
from model.utils import arch_to_genotype, draw_genotype, infinite_get, arch_to_string
import os
from model import common
from option import args

def make_model(args, parent=False):
    return NASNetwork(args)


class NASOp(nn.Module):
    def __init__(self, C, stride):
        super(NASOp, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in COMPACT_PRIMITIVES:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, index):
        return self._ops[index](x)


class NASOp_upsampling(nn.Module):
    def __init__(self, C, stride):
        super(NASOp_upsampling, self).__init__()
        self._ops = nn.ModuleList()
        for primitive in COMPACT_PRIMITIVES_UPSAMPLING:
            op = OPS[primitive](C, stride, False)
            if 'pool' in primitive:
                op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
            self._ops.append(op)

    def forward(self, x, index):
        return self._ops[index](x)


class NASCell(nn.Module):
    def __init__(self, steps, device, multiplier, C_prev_prev, C_prev, C, upsample, upsample_prev, loose_end=False):
        super(NASCell, self).__init__()
        print(C_prev_prev, C_prev, C)
        self.steps = steps
        self.device = device
        self.multiplier = multiplier
        self.C = C
        self.upsample = upsample
        self.loose_end = loose_end
        if upsample_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)

        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
        self._steps = steps

        self._ops = nn.ModuleList()

        if upsample:
            for i in range(self._steps):
                for j in range(i + 2):
                    scale = args.scale
                    stride = scale[0] if upsample and j < 2 else 1
                    op = NASOp_upsampling(C, stride)
                    self._ops.append(op)
        else:
            for i in range(self._steps):
                for j in range(i + 2):
                    stride = 1
                    op = NASOp(C, stride)
                    self._ops.append(op)

        self.final_conv = FinalConv(C * multiplier, C * multiplier)

    def forward(self, s0, s1, arch):
        """

        :param s0:
        :param s1:
        :param arch: a list, the element is (op_id, from_node, to_node), sorted by to_node (!!not check
                     the ordering for efficiency, but must be assured when generating!!)
                     from_node/to_node starts from 0, 0 is the prev_prev_node, 1 is prev_node
                     The mapping from (F, T) pair to edge_ID is (T-2)(T+1)/2+S,

        :return:
        """
        s0 = self.preprocess0.forward(s0)
        s1 = self.preprocess1.forward(s1)
        states = {0: s0, 1: s1}
        used_nodes = set()
        for op, f, t in arch:
            edge_id = int((t - 2) * (t + 1) / 2 + f)
            if t in states:
                states[t] = states[t] + self._ops[edge_id](states[f], op)
            else:
                states[t] = self._ops[edge_id](states[f], op)
            used_nodes.add(f)
        if self.loose_end:
            index = torch.tensor([(i - 2) * self.C + j for j in range(self.C) for i in range(2, self._steps + 2) if
                                  i not in used_nodes]).to(self.device)
            out = torch.cat([states[i] for i in range(2, self._steps + 2) if i not in used_nodes], dim=1)
            return self.final_conv(out, index)
        else:
            return torch.cat([states[i] for i in range(2, self._steps + 2)], dim=1)


class ArchMaster(nn.Module):
    def __init__(self, n_layers, n_nodes, device, controller_type='LSTM', controller_hid=None,
                 controller_temperature=None, controller_tanh_constant=None, controller_op_tanh_upsampling=None,
                 lstm_num_layers=2):
        super(ArchMaster, self).__init__()
        self.K = sum([x + 2 for x in range(n_nodes)])
        self.n_layers = n_layers

        self.num_ops_normal = len(COMPACT_PRIMITIVES)
        self.num_ops_upsampling = len(COMPACT_PRIMITIVES_UPSAMPLING)

        self.n_nodes = n_nodes
        self.device = device
        self.controller_type = controller_type

        self.controller_hid = controller_hid
        self.attention_hid = self.controller_hid
        self.temperature = controller_temperature
        self.tanh_constant = controller_tanh_constant
        self.op_tanh_upsampling = controller_op_tanh_upsampling
        self.lstm_num_layers = lstm_num_layers

        # normal cell:
        self.node_op_hidden = nn.Embedding(n_nodes + 1 + self.num_ops_normal, self.controller_hid)
        self.emb_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
        self.hid_attn = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
        self.v_attn = nn.Linear(self.controller_hid, 1, bias=False)
        self.w_soft = nn.Linear(self.controller_hid, self.num_ops_normal)
        self.lstm = nn.LSTMCell(self.controller_hid, self.controller_hid)

        # upsampling cell:
        self.node_op_hidden_upsampling = nn.Embedding(n_nodes + 1 + self.num_ops_upsampling, self.controller_hid)
        self.emb_attn_upsampling = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
        self.hid_attn_upsampling = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
        self.v_attn_upsampling = nn.Linear(self.controller_hid, 1, bias=False)
        self.w_soft_upsampling = nn.Linear(self.controller_hid, self.num_ops_upsampling)
        self.lstm_upsampling = nn.LSTMCell(self.controller_hid, self.controller_hid)

        # upsampling position
        self.node_op_hidden_position = nn.Embedding(n_layers, self.controller_hid)
        self.emb_attn_position = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
        self.hid_attn_position = nn.Linear(self.controller_hid, self.attention_hid, bias=False)
        self.v_attn_position = nn.Linear(self.controller_hid, 1, bias=False)
        self.w_soft_position = nn.Linear(self.controller_hid, self.n_layers)
        self.lstm_position = nn.LSTMCell(self.controller_hid*2, self.controller_hid)

        self.reset_parameters()
        self.static_init_hidden = utils.keydefaultdict(self.init_hidden)
        self.static_inputs = utils.keydefaultdict(self._get_default_hidden)
        self.tanh = nn.Tanh()
        self.prev_nodes, self.prev_ops = [], []
        self.query_index = torch.LongTensor(range(0, n_nodes + 1)).to(device)
        self.query_index_position = torch.LongTensor(range(0, n_layers)).to(device)

    def _get_default_hidden(self, key):
        return utils.get_variable(
            torch.zeros(key, self.controller_hid), self.device, requires_grad=False)

    # device
    def init_hidden(self, batch_size):
        zeros = torch.zeros(batch_size, self.controller_hid)
        return (utils.get_variable(zeros, self.device, requires_grad=False),
                utils.get_variable(zeros.clone(), self.device, requires_grad=False))

    def reset_parameters(self):
        init_range = 0.1
        for param in self.parameters():
            param.data.uniform_(-init_range, init_range)
        self.w_soft.bias.data.fill_(0)

    def forward(self):
        log_p, entropy = 0, 0
        self.prev_nodes, self.prev_ops = [], []
        self.force_uniform = False
        batch_size = 1
        inputs = self.static_inputs[batch_size]  # batch_size x hidden_dim
        hidden = self.static_init_hidden[batch_size]

        # normal cell:
        for node_idx in range(self.n_nodes):
            for i in range(2):  # index_1, index_2
                if node_idx == 0 and i == 0:
                    embed = inputs
                else:
                    embed = self.node_op_hidden(inputs)
                if self.force_uniform:
                    probs = F.softmax(torch.zeros(node_idx + 2).type_as(embed), dim=-1)
                else:
                    hx, cx = self.lstm(embed, hidden)
                    query = self.node_op_hidden.weight.index_select(
                        0, self.query_index[0:node_idx + 2]
                    )
                    query = self.tanh(self.emb_attn(query) + self.hid_attn(hx))
                    logits = self.v_attn(query).view(-1)
                    if self.temperature is not None:
                        logits /= self.temperature
                    if self.tanh_constant is not None:
                        logits = self.tanh_constant * self.tanh(logits)
                    probs = F.softmax(logits, dim=-1)
                    hidden = (hx, cx)
                log_probs = torch.log(probs)
                action = probs.multinomial(num_samples=1)
                selected_log_p = log_probs.gather(0, action)[0]
                self.prev_nodes.append(action)
                log_p += selected_log_p
                entropy += -(log_probs * probs).sum()
                inputs = utils.get_variable(action, self.device, requires_grad=False)
            for i in range(2):  # op_1, op_2
                embed = self.node_op_hidden(inputs)
                if self.force_uniform:
                    probs = F.softmax(torch.zeros(self.n_ops).type_as(embed), dim=-1)
                else:
                    hx, cx = self.lstm(embed, hidden)
                    logits = self.w_soft(hx).view(-1)
                    if self.temperature is not None:
                        logits /= self.temperature
                    if self.tanh_constant is not None:
                        op_tanh = self.tanh_constant / self.op_tanh_upsampling
                        logits = op_tanh * self.tanh(logits)
                    probs = F.softmax(logits, dim=-1)  # absent in ENAS code
                    hidden = (hx, cx)
                log_probs = torch.log(probs)
                action = probs.multinomial(num_samples=1)
                self.prev_ops.append(action)
                selected_log_p = log_probs.gather(0, action)[0]
                log_p += selected_log_p
                entropy += -(log_probs * probs).sum()
                inputs = utils.get_variable(action + self.n_nodes + 1, self.device, requires_grad=False)
        arch_normal = utils.convert_lstm_output(self.n_nodes, torch.cat(self.prev_nodes), torch.cat(self.prev_ops))
        hx_normal = hx

        # upasampling cell :
        log_p_upsampling, entropy_upsampling = 0, 0
        self.prev_nodes, self.prev_ops = [], []
        self.force_uniform = False
        batch_size = 1
        inputs = self.static_inputs[batch_size]  # batch_size x hidden_dim
        hidden = self.static_init_hidden[batch_size]
        for node_idx in range(self.n_nodes):
            for i in range(2):  # index_1, index_2
                if node_idx == 0 and i == 0:
                    embed = inputs
                else:
                    embed = self.node_op_hidden_upsampling(inputs)
                if self.force_uniform:
                    probs = F.softmax(torch.zeros(node_idx + 2).type_as(embed), dim=-1)
                else:
                    hx, cx = self.lstm_upsampling(embed, hidden)
                    query = self.node_op_hidden_upsampling.weight.index_select(
                        0, self.query_index[0:node_idx + 2]
                    )
                    query = self.tanh(self.emb_attn_upsampling(query) + self.hid_attn_upsampling(hx))
                    logits = self.v_attn_upsampling(query).view(-1)
                    if self.temperature is not None:
                        logits /= self.temperature
                    if self.tanh_constant is not None:
                        logits = self.tanh_constant * self.tanh(logits)
                    probs = F.softmax(logits, dim=-1)
                    hidden = (hx, cx)
                log_probs = torch.log(probs)
                action = probs.multinomial(num_samples=1)
                selected_log_p = log_probs.gather(0, action)[0]
                self.prev_nodes.append(action)
                log_p_upsampling += selected_log_p
                entropy_upsampling += -(log_probs * probs).sum()
                inputs = utils.get_variable(action, self.device, requires_grad=False)
            for i in range(2):  # op_1, op_2
                embed = self.node_op_hidden_upsampling(inputs)
                if self.force_uniform:
                    probs = F.softmax(torch.zeros(self.n_ops).type_as(embed), dim=-1)
                else:
                    hx, cx = self.lstm_upsampling(embed, hidden)
                    logits = self.w_soft_upsampling(hx).view(-1)
                    if self.temperature is not None:
                        logits /= self.temperature
                    if self.tanh_constant is not None:
                        op_tanh = self.tanh_constant / self.op_tanh_upsampling
                        logits = op_tanh * self.tanh(logits)
                    probs = F.softmax(logits, dim=-1)  # absent in ENAS code
                    hidden = (hx, cx)
                log_probs = torch.log(probs)
                action = probs.multinomial(num_samples=1)
                self.prev_ops.append(action)
                selected_log_p = log_probs.gather(0, action)[0]
                log_p_upsampling += selected_log_p
                entropy_upsampling += -(log_probs * probs).sum()
                inputs = utils.get_variable(action + self.n_nodes + 1, self.device, requires_grad=False)
        arch_upsampling = utils.convert_lstm_output(self.n_nodes, torch.cat(self.prev_nodes), torch.cat(self.prev_ops))
        hx_upsampling = hx

        # upsampling postion lstm
        log_p_position, entropy_position = 0, 0
        self.prev_nodes, self.prev_ops = [], []
        self.force_uniform = False
        zeros = torch.zeros(batch_size, self.controller_hid)
        hidden = (utils.get_variable(zeros, self.device, requires_grad=False).cuda(),
                  utils.get_variable(zeros.clone(), self.device, requires_grad=False).cuda())
        inputs = torch.cat([hx_normal,hx_upsampling],1).cuda()
        hx, cx = self.lstm_position(inputs, hidden)
        query = self.node_op_hidden_position.weight.index_select(
            0, self.query_index_position[0:self.n_layers]
        )
        query = self.tanh(self.emb_attn_position(query) + self.hid_attn_position(hx))
        logits = self.v_attn_position(query).view(-1)
        if self.temperature is not None:
            logits /= self.temperature
        if self.tanh_constant is not None:
            logits = self.tanh_constant * self.tanh(logits)
        probs = F.softmax(logits, dim=-1)
        hidden = (hx, cx)
        log_probs = torch.log(probs)
        action = probs.multinomial(num_samples=1)
        selected_log_p = log_probs.gather(0, action)[0]
        self.prev_nodes.append(action)
        log_p_position += selected_log_p
        entropy_position += -(log_probs * probs).sum()

        return  arch_normal,log_p, entropy,  arch_upsampling,log_p_upsampling,entropy_upsampling,   action.cpu().numpy()[0],log_p_position, entropy_position


class NASNetwork(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(NASNetwork, self).__init__()
        steps = 4
        multiplier = 4
        stem_multiplier = 3
        C = args.init_channels
        self._C = args.init_channels
        self._layers = args.layers
        self.n_layers = args.layers
        layers = args.layers
        self._steps = steps
        self._multiplier = multiplier
        self._device = 'cuda'
        self.args = args

        self.controller_type = args.controller_type
        self.controller_hid = args.controller_hid
        self.controller_temperature = args.controller_temperature
        self.controller_tanh_constant = args.controller_tanh_constant
        self.controller_op_tanh_upsampling = args.controller_op_tanh_upsampling
        self.entropy_coeff = args.entropy_coeff
        self.edge_hid = args.edge_hid
        self.pruner_nfeat = args.pruner_nfeat
        self.pruner_nhid = args.pruner_nhid
        self.pruner_dropout = args.pruner_dropout
        self.pruner_normalize = args.pruner_normalize
        self.split_fc = args.split_fc
        self.loose_end = args.loose_end
        self.cur_normal_arch = None
        self.cur_upsampling_arch = None
        self.upsampling_position = None

        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        C_curr = multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        upsample_prev = False


        for i in range(layers):
            upsample = False
            cell = NASCell(steps, self._device, multiplier, C_prev_prev, C_prev, C_curr, upsample, upsample_prev, loose_end=self.loose_end)
            upsample_prev = upsample
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, multiplier * C_curr

        upsample = True
        self.upsampling_cell = NASCell(steps, self._device, multiplier, C_prev_prev, C_prev, C_curr, upsample,
                                   upsample_prev, loose_end=self.loose_end)
        scale = args.scale[0]
        self.final_conv = conv(C_prev, args.n_colors, 3)
        self._initialize_archmaster()

    def _initialize_archmaster(self):
        self.arch_normal_master = ArchMaster(self._layers, self._steps, self._device, self.controller_type,
                                             self.controller_hid, self.controller_temperature,
                                             self.controller_tanh_constant, self.controller_op_tanh_upsampling)
        self._arch_parameters = list(self.arch_normal_master.parameters())

    def _initialize_archpruner(self):
        self.arch_normal_pruner = ArchPruner(self._steps, self._device, self.edge_hid, self.pruner_nfeat,
                                             self.pruner_nhid, self.pruner_dropout, self.pruner_normalize,
                                             self.split_fc)
        self._pruner_parameters = list(self.arch_normal_pruner.parameters())


    def _inner_forward(self, input, arch_normal, arch_upsampling, upsampling_position):
        try:
            input = self.sub_mean(input)
        except:
            input = input.cuda()
            input = self.sub_mean(input)
        s0 = s1 = self.stem(input)
        for i in range(self.n_layers):
            if i == upsampling_position:
                archs = arch_upsampling
                s0, s1 = self.upsampling_cell(s0, s1, archs), self.upsampling_cell(s0, s1, archs)
            else:
                archs = arch_normal
                s0, s1 = s1, self.cells[i](s0, s1, archs)
        logits = self.final_conv(s1)
        logits = self.add_mean(logits)
        return logits
    
    def _test_acc(self, test_queue, arch_normal, arch_upsampling):
        top1 = utils.AvgrageMeter()
        for step, (test_input, test_target) in enumerate(test_queue):
            test_input = test_input.to(self._device)
            test_target = test_target.to(self._device)
            n = test_input.size(0)
            logits = self._inner_forward(test_input, arch_normal, arch_upsampling)
            accuracy = utils.accuracy(logits, test_target)[0]
            top1.update(accuracy.item(), n)
        return top1.avg

    def arch_forward(self, valid_input):
        arch_normal, arch_normal_logP, arch_normal_entropy, arch_upsampling, arch_upsampling_logP, arch_upsampling_entropy, position, log_p_position, entropy_position\
            = self.arch_normal_master.forward()
        logits = self._inner_forward(valid_input, arch_normal, arch_upsampling, position)
        return logits, arch_normal, arch_normal_logP, arch_normal_entropy, arch_upsampling, arch_upsampling_logP, arch_upsampling_entropy, position, log_p_position, entropy_position

    def pruner_forward(self, valid_input):
        arch_normal, arch_normal_logP, arch_normal_entropy, arch_upsampling, arch_upsampling_logP, arch_upsampling_entropy, position, log_p_position, entropy_position \
            = self.arch_normal_master.forward()
        logits = self._inner_forward(valid_input, arch_normal, arch_upsampling, position)
        return logits, arch_normal, arch_normal_logP, arch_normal_entropy, arch_upsampling, arch_upsampling_logP, arch_upsampling_entropy, position, log_p_position, entropy_position

    def forward(self, valid_input):
        arch_normal, arch_normal_logP, arch_normal_entropy, arch_upsampling, arch_upsampling_logP, arch_upsampling_entropy, position, log_p_position, entropy_position \
            = self.arch_normal_master.forward()
        self.cur_normal_arch = arch_normal
        self.cur_upsampling_arch = arch_upsampling
        self.upsampling_position = position
        logits = self._inner_forward(valid_input, arch_normal, arch_upsampling, position)
        return logits

    def _loss_arch(self, input, target, baseline=None):
        logits, arch_normal, arch_normal_logP, arch_normal_entropy, arch_upsampling, arch_upsampling_logP, arch_upsampling_entropy, position, log_p_position, entropy_position = self.arch_forward(
            input)
        accuracy = utils.accuracy(logits, target)[0] / 100.0
        reward = accuracy - baseline if baseline else accuracy
        policy_loss = -(arch_normal_logP + arch_upsampling_logP) * reward - (
            self.entropy_coeff[0] * arch_normal_entropy + self.entropy_coeff[1] * arch_upsampling_entropy)
        return policy_loss, accuracy, arch_normal_entropy, arch_upsampling_entropy

    def _loss_pruner(self, input):
        logits, arch_normal, arch_normal_logP, arch_normal_entropy, arch_upsampling, arch_upsampling_logP, arch_upsampling_entropy, position, log_p_position, entropy_position = self.pruner_forward(
            input)

        return logits, arch_normal_logP, arch_normal_entropy, arch_upsampling_logP, arch_upsampling_entropy, position, log_p_position, entropy_position

    def arch_parameters(self):
        return self._arch_parameters

    def pruner_parameters(self):
        return self._pruner_parameters

    def model_parameters(self):
        for k, v in self.named_parameters():
            if 'arch' not in k:
                yield v

    def save_arch_to_pdf(self, suffix):
        genotype = arch_to_genotype(self.cur_normal_arch, self.cur_upsampling_arch, self._steps, "COMPACT")
        return genotype,self.upsampling_position
