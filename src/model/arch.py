import torch
import torch.nn as nn
import torch.nn.functional as F
from model.operations import *
from model.genotypes import COMPACT_PRIMITIVES, PRUNER_PRIMITIVES,COMPACT_PRIMITIVES_UPSAMPLING
from model.genotypes import Genotype
import model.utils as utils
import numpy as np
from model.utils import arch_to_genotype, draw_genotype, infinite_get, arch_to_string
import os
from model import common
import model.genotypes as genotypes
from model.utils import drop_path
import sys
sys.path.append("..")
from option import args

def make_model(args, parent=False):
    return ArchNetwork(args)

class Cell(nn.Module):
    def __init__(self, genotype, C_prev_prev, C_prev, C, upsample, upsample_prev):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)
        self.upsample = upsample

        self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if upsample:
            op_names, f_nodes, t_nodes = zip(*genotype.upsampling)
            concat = genotype.upsampling_concat
        else:
            op_names, f_nodes, t_nodes = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, f_nodes, t_nodes, concat, upsample)

    def _compile(self, C, op_names, f_nodes, t_nodes, concat, upsample):
        assert len(op_names) == len(f_nodes) == len(t_nodes)
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, f in zip(op_names, f_nodes):
            scale = args.scale
            stride = scale[0] if upsample and f < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._f_nodes = f_nodes
        self._t_nodes = t_nodes

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = {0: s0, 1: s1}
        for op, f, t in zip(self._ops, self._f_nodes, self._t_nodes):
            s = op(states[f])
            if self.training and drop_prob > 0.:
                if not isinstance(op, Identity):
                    s = drop_path(s, drop_prob)
            if t in states:
                states[t] = states[t] + s
            else:
                states[t] = s
        return torch.cat([states[i] for i in self._concat], dim=1)


class ArchNetwork(nn.Module):
    def __init__(self,args,conv=common.default_conv):
        super(ArchNetwork, self).__init__()
        self._layers = args.layers
        layers=args.layers
        genotype=eval("genotypes.%s" % args.genotype)
        stem_multiplier = 4
        C=args.init_channels
        self._upsampling_Pos = args.upsampling_Pos
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        upsample_prev = False
        for i in range(layers):
            if i in [self._upsampling_Pos]:
                upsample = True
            else:
                upsample = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, upsample, upsample_prev)
            upsample_prev = upsample
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr

        self.final_conv= conv(C_prev, args.n_colors, 3)

    def model_parameters(self):
        for k, v in self.named_parameters():
            if 'arch' not in k:
                yield v

    def forward(self, input, drop_path_prob=0):
        self.drop_path_prob = drop_path_prob
        try:
            input = self.sub_mean(input)
        except:
            input = input.cuda()
            input = self.sub_mean(input)
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            if cell.upsample == True:
                s0, s1 = cell(s0, s1, self.drop_path_prob), cell(s0, s1, self.drop_path_prob)
            else:
                s0, s1 = s1, cell(s0, s1, self.drop_path_prob)

        logits = self.final_conv(s1)
        logits = self.add_mean(logits)
        return logits

    def save_arch_to_pdf(self,suffix):
        genotype = arch_to_genotype(self.cur_normal_arch, self.cur_normal_arch, self._steps, "COMPACT")
        return genotype

