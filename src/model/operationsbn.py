import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
sys.path.append("..")
from option import args

OPS = {
    'none': lambda C, stride, affine: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine: Identity(),
    'sep_conv_3x3': lambda C, stride, affine: SepConv(C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C, stride, affine: SepConv(C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C, stride, affine: SepConv(C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
    'conv_7x1_1x7': lambda C, stride, affine: nn.Sequential(
        nn.ReLU(inplace=False),
        nn.Conv2d(C, C, (1, 7), stride=(1, stride), padding=(0, 3), bias=False),
        nn.Conv2d(C, C, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
        nn.BatchNorm2d(C, affine=affine)
    ),
    'conv_3x3': lambda C, stride, affine: ReLUConvBN(C, C, 3, stride, 1, affine=affine),
    'conv_3x3_no_relu': lambda C, stride, affine: ConvBN(C, C, 3, stride, 1, affine=affine),
    'conv_3x3_no_relu_no_bn': lambda C, stride, affine: Conv(C, C, 3, stride, 1, affine=affine),
    'conv_3x3_no_bn': lambda C, stride, affine: ReLUConv(C, C, 3, stride, 1, affine=affine),
    'rcab': lambda C, stride, affine: RCAB( C, 3, 3,bias=True, bn=True, act=nn.ReLU(True), res_scale=1),
                                #first 3 is kernel-size,the second 3 is the number of feature map
    'up_and_down': lambda C, stride, affine: UPANDDOWN( C, scale_factor=args.scale[0]),

    #upsampling cell:
    'sub_pixel':lambda C, stride, affine: SUBPIXEL(C, scale_factor=stride),
    'manual_sub_pixel':lambda C, stride, affine: SUBPIXEL(C, scale_factor=2),
    'deconvolution':lambda C, stride, affine: Deconvolution(C,stride),
    'bilinear':lambda C, stride, affine: Bilinear(stride),
    'nearest':lambda C, stride, affine: Nearest(stride),
    'linear':lambda C, stride, affine: Linear(stride),
    'area':lambda C, stride, affine: Area(stride),
    'upproject': lambda C, stride, affine: Upproject(C, scale_factor=2),
    'downproject': lambda C, stride, affine: Downproject(C, scale_factor=2),
    'upprojectnone': lambda C, stride, affine: UpprojectNone(C, scale_factor=2),
}
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class ReLUConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            # nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

class Downproject(nn.Module):
    def __init__(
        self,  base_filter, scale_factor):
        super(Downproject, self).__init__()

        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2

        # self.up1 = UpBlock(base_filter, kernel, stride, padding)
        self.down1 = DownBlock(base_filter, kernel, stride, padding)

    def forward(self, x):
        x = self.down1(x)
        return x

class UpprojectNone(nn.Module):
    def __init__(
        self,  base_filter, scale_factor):
        super(UpprojectNone, self).__init__()

        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2

        self.up1 = UpBlock(base_filter, kernel, stride, padding)
        # self.down1 = DownBlock(base_filter, kernel, stride, padding)

    def forward(self, x):
        x = self.up1(x)
        return x.mul(0.)

class Upproject(nn.Module):
    def __init__(
        self,  base_filter, scale_factor):
        super(Upproject, self).__init__()

        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2

        self.up1 = UpBlock(base_filter, kernel, stride, padding)
        # self.down1 = DownBlock(base_filter, kernel, stride, padding)

    def forward(self, x):
        x = self.up1(x)
        return x

class Conv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(Conv, self).__init__()
        self.op = nn.Sequential(
            # nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            # nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

class ConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ConvBN, self).__init__()
        self.op = nn.Sequential(
            # nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)

class Bilinear(nn.Module):
    def __init__(self, stride):
        super(Bilinear, self).__init__()
        self.scale=stride

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode='bilinear')

class Linear(nn.Module):
    def __init__(self, stride):
        super(Linear, self).__init__()
        self.scale=stride

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode='linear')

class Area(nn.Module):
    def __init__(self, stride):
        super(Area, self).__init__()
        self.scale=stride

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode='area')

class Nearest(nn.Module):
    def __init__(self, stride):
        super(Nearest, self).__init__()
        self.scale=stride

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale, mode='nearest')

class Deconvolution(nn.Module):
    def __init__(self, C, stride):
        super(Deconvolution, self).__init__()
        if stride==2:
            kernel_size=3
            output_padding=1
        elif stride==4:
            kernel_size=5
            output_padding = 1
        else:
            kernel_size=3
            output_padding = 0
        self.deconv=nn.ConvTranspose2d(C, C,kernel_size=kernel_size,stride=stride, padding=1,output_padding=output_padding)

    def forward(self, x):
        return self.deconv(x)

class SUBPIXEL(nn.Module):
    def __init__(self, C, scale_factor,conv=default_conv):
        super(SUBPIXEL, self).__init__()
        self.upsample=Upsampler(conv, scale_factor, C, act=False)

    def forward(self, x):
        return self.upsample(x)

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)



class ReLUConvBN(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):
    def  __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        return x.mul(0.)

class FactorizedReduce(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        if x.size(2)%2!=0:
            x = F.pad(x, (1,0,1,0), "constant", 0)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out


class ThinConv2d(nn.Conv2d):
    """
    custom convolutional layers for thin convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(ThinConv2d, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def _thin_weight(self, input, index=None):
        n, c, h, w = input.size()
        # print(index.size())
        # print(index)
        num_nodes = index.size(0)
        index = index.view(1, num_nodes, 1, 1)
        final_index = index.expand(n, num_nodes, h, w)
        thin_data = torch.gather(input, 1, final_index)
        return thin_data

    def forward(self, input, index=None):
        if index is not None:
            thin_weight = self._thin_weight(self.weight, index)
        else:
            thin_weight = self.weight
        return F.conv2d(input, thin_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class FinalConv(nn.Module):
    def __init__(self, C_in, C_out, affine=True):
        super(FinalConv, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.thin_conv = ThinConv2d(C_in, C_out, 1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x, index):
        x = self.relu(x)
        out = self.thin_conv(x, index)
        out = self.bn(out)
        return out

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self,  n_feat, kernel_size, reduction,conv=default_conv,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res


class UPANDDOWN(nn.Module):
    def __init__(
        self,  base_filter, scale_factor):
        super(UPANDDOWN, self).__init__()

        if scale_factor == 2:
            kernel = 6
            stride = 2
            padding = 2
        elif scale_factor == 4:
            kernel = 8
            stride = 4
            padding = 2
        elif scale_factor == 8:
            kernel = 12
            stride = 8
            padding = 2

        self.up1 = UpBlock(base_filter, kernel, stride, padding)
        self.down1 = DownBlock(base_filter, kernel, stride, padding)

    def forward(self, x):
        x = self.up1(x)
        x = self.down1(x)
        return x

class UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0

class DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0

class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_size)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_size)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out