import os
import math
from decimal import Decimal

import utility
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils as utils
from torch.autograd import Variable
from tqdm import tqdm

class Controller_Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_controller_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_controller_optimizer(args, self.model)
        self.flops_scale=args.flops_scale

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8
        self.entropy_coeff=args.entropy_coeff
        self.ema_baseline_decay=args.ema_baseline_decay

    def get_variable(inputs, **kwargs):
        cuda = True
        if type(inputs) in [list, np.ndarray]:
            inputs = torch.Tensor(inputs)
        if cuda:
            out = Variable(inputs.cuda(), **kwargs)
        else:
            out = Variable(inputs, **kwargs)
        return out
    
    def compute_flops(self, module: nn.Module, size, skip_pattern):
        def size_hook(module: nn.Module, input: torch.Tensor, output: torch.Tensor):
            *_, h, w = output.shape
            module.output_size = (h, w)

        hooks = []
        for name, m in module.named_modules():
            if isinstance(m, nn.Conv2d):
                hooks.append(m.register_forward_hook(size_hook))

        with torch.no_grad():
            training = module.training
            module.eval()
            module(torch.rand(size),2)
            module.train(mode=training)
        for hook in hooks:
            hook.remove()

        flops = 0
        for name, m in module.named_modules():
            if skip_pattern in name:
                continue
            if isinstance(m, nn.Conv2d):
                if hasattr(m, 'output_size'):
                    h, w = m.output_size
                    kh, kw = m.kernel_size
                    flops += h * w * m.in_channels * m.out_channels * kh * kw / m.groups
            if isinstance(module, nn.Linear):
                flops += m.in_features * m.out_features
        return flops

    def controller_train(self):
        self.optimizer.schedule()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Training controller...Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        # self.loss.start_log()
        self.model.model.train()
        baseline = None
        controller_loss=0.0
        avg_psnr=0.0
        avg_adv =0.0
        d=self.loader_train
        scale=self.scale[0]

        for batch, (lr, hr, _) in enumerate(self.loader_train):
            idx_scale = None
            lr, hr = self.prepare(lr, hr)
            self.optimizer.zero_grad()
            sr, arch_normal_logP, arch_normal_entropy, arch_upsampling_logP, arch_upsampling_entropy, position, log_p_position, entropy_position\
                = self.model.forward(lr, idx_scale)
            sr = utility.quantize(sr, self.args.rgb_range)

            psnr=utility.calc_psnr(
                sr, hr, scale, self.args.rgb_range, dataset=d
            )

            flops = self.compute_flops(self.model.model, size=(1, 3, 48, 48), skip_pattern='skip')
            used_flops = flops * 1e-8 * 0.5

            used_flops = float(format(used_flops, '.3f'))
            
            avg_psnr += psnr

            rewards = self.flops_scale *  psnr - (1 - self.flops_scale) * used_flops

            # moving average baseline
            if baseline is None:
                baseline = rewards
            else:
                decay = self.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * rewards
            adv=rewards - baseline
            adv=torch.Tensor(np.array(adv))
            reward = Variable(adv, requires_grad=False)
            policy_loss = -(arch_normal_logP +  arch_upsampling_logP + 10 * log_p_position) * reward - (
                              self.entropy_coeff[0] * (arch_normal_entropy + arch_upsampling_entropy + 10 * entropy_position))

            controller_loss += policy_loss
            avg_adv+=reward
            policy_loss.backward()
            if self.args.gclip > 0:
                nn.utils.clip_grad_norm(self.model.arch_parameters(), self.args.grad_clip)
            self.optimizer.step()

            if (batch + 1) % self.args.controller_print_every == 0:
                self.ckp.write_log('[{}/{}]\t PSNR: {:.3f}\t policy_loss: {:.3f}'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    psnr,
                    policy_loss))

        controller_loss /= len(d)
        avg_psnr/= len(d)

        self.ckp.write_log(
            '[avg policy_loss: {:.3f},  {} x{}]\t  AVG_PSNR: {:.3f}'.format(
                controller_loss,
                'DIV2K',
                scale,
                avg_psnr
            )
        )
        controller_loss=0



    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch() + 1
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename, _ in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs

