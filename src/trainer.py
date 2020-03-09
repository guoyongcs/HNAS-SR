import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn as nn
import torch.nn.utils as utils
from tqdm import tqdm
from model.utils import  draw_genotype

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)
        self.torch_version = float(torch.__version__[0:3])

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

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
            module(torch.rand(size), 2)
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

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def train(self):
        self.optimizer.schedule()
        if self.torch_version < 1.1:
            self.loss.step()
            epoch = self.optimizer.get_last_epoch() + 1
        else:
            epoch = self.optimizer.get_last_epoch()
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        # note when size 48 x 48 and flops * 1e-8  is equal to 480 x 480 and flops * 1e-9
        flops = self.compute_flops(self.model, size=(1, 3, 48, 48), skip_pattern='skip')
        used_flops = flops * 1e-8
        self.ckp.write_log(
            '[Model flops(* 1e-8): {}]'.format(used_flops)
        )

        # param = self.count_parameters(self.model)
        # self.ckp.write_log(
        #     '[Model param: {}]'.format(param)
        # )

        timer_data, timer_model = utility.timer(), utility.timer()
        for batch, (lr, hr, _) in enumerate(self.loader_train):
            idx_scale=None
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            self.optimizer.zero_grad()
            sr = self.model(lr, idx_scale)
            loss = self.loss(sr, hr)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()
            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        if self.torch_version >= 1.1:
            self.loss.step()

    def test(self):
        torch.set_grad_enabled(False)
        if self.torch_version < 1.1:
            epoch = self.optimizer.get_last_epoch() + 1
        else:
            epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()
        timer_test = utility.timer()
        best_psnr = 0
        best_epoch = False
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                if epoch % self.args.sampling_epoch_margin == 0:
                    sampling=self.args.sampling
                else:
                    sampling =1
                for i in range(sampling):
                    self.ckp.log[-1, idx_data, idx_scale]=0
                    for lr, hr, filename in tqdm(d, ncols=80):
                        lr, hr = self.prepare(lr, hr)
                        sr = self.model(lr, idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)
                        save_list = [sr]
                        psnr=utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range, dataset=d
                        )
                        self.ckp.log[-1, idx_data, idx_scale] += psnr
                        if self.args.save_gt:
                            save_list.extend([lr, hr])

                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list, scale)

                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    if self.ckp.log[-1, idx_data, idx_scale] >= best_psnr:
                        best_psnr = float(self.ckp.log[-1, idx_data, idx_scale])
                        genotype,upsampling_position = self.model.model.save_arch_to_pdf(epoch)

                    best = self.ckp.log.max(0)
                    if best[1][idx_data, idx_scale]+1 == epoch:
                        best_epoch=True
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                            d.dataset.name,
                            scale,
                            self.ckp.log[-1, idx_data, idx_scale],
                            best[0][idx_data, idx_scale],
                            best[1][idx_data, idx_scale]+1
                        )
                    )
                if epoch % self.args.sampling_epoch_margin == 0:
                    folder = os.path.join( '..', 'experiment', self.args.save)
                    self.ckp.write_log('epoch:{}'.format(epoch))
                    self.ckp.write_log('upsampling_position:{}'.format(upsampling_position))
                    self.ckp.write_log('genotype:{}'.format(genotype))
                    draw_genotype(genotype.normal, 4,
                                  os.path.join(folder, "normal_{}_upsamplingPos_{}".format(epoch, upsampling_position)))
                    draw_genotype(genotype.upsampling, 4,
                                  os.path.join(folder, "upsampling_{}_upsamplingPos_{}".format(epoch, upsampling_position)))
                self.ckp.write_log('upsampling_position:{}'.format(upsampling_position))
                self.ckp.write_log('genotype:{}'.format(genotype))
                self.ckp.log[-1, idx_data, idx_scale] = best_psnr

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best_epoch))

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
            self.final_test()
            return True
        else:
            if self.torch_version < 1.1:
                epoch = self.optimizer.get_last_epoch() + 1
            else:
                epoch = self.optimizer.get_last_epoch()
            return epoch >= self.args.epochs

    def final_test(self):
        torch.set_grad_enabled(False)

        if self.torch_version < 1.1:
            epoch = self.optimizer.get_last_epoch() + 1
        else:
            epoch = self.optimizer.get_last_epoch()
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
                # for lr, hr, filename, _ in tqdm(d, ncols=80):
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr,idx_scale)
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


    def derive(self):
        torch.set_grad_enabled(False)

        if self.torch_version < 1.1:
            epoch = self.optimizer.get_last_epoch() + 1
        else:
            epoch = self.optimizer.get_last_epoch()
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
                result={}
                best_derive_psnr=0
                for i in range(10):
                    result_psnr = []
                    for lr, hr, filename in tqdm(d, ncols=80):
                        lr, hr = self.prepare(lr, hr)
                        sr = self.model(lr, idx_scale)
                        sr = utility.quantize(sr, self.args.rgb_range)
                        save_list = [sr]
                        psnr=utility.calc_psnr(
                            sr, hr, scale, self.args.rgb_range, dataset=d
                        )
                        result_psnr.append(psnr)
                        if self.args.save_gt:
                            save_list.extend([lr, hr])
                        if self.args.save_results:
                            self.ckp.save_results(d, filename[0], save_list, scale)
                    self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                    best = self.ckp.log.max(0)
                    self.ckp.write_log(
                        '[{} x{}]\tPSNR: {:.3f} '.format(
                            d.dataset.name,
                            scale,
                            sum(result_psnr) / len(result_psnr)
                        )
                    )
                    if sum(result_psnr) / len(result_psnr)>=best_derive_psnr:
                        best_derive_psnr=sum(result_psnr) / len(result_psnr)
                    genotype,upsampling_position = self.model.model.save_arch_to_pdf(epoch)
                    self.ckp.write_log('genotype:{}'.format(genotype))
                    result[sum(result_psnr) / len(result_psnr)] =(genotype,upsampling_position)
        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        self.ckp.write_log('Best psnr:{:.3f}'.format(best_derive_psnr))
        self.ckp.write_log('Best Position:{}'.format(result[best_derive_psnr][1]))
        self.ckp.write_log('Best genotype:{}'.format(result[best_derive_psnr][0]))

        if self.args.save_results:
            self.ckp.end_background()
        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))
        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )
        torch.set_grad_enabled(True)
