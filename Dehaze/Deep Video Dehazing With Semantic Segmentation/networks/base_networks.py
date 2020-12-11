import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeaturePool_3D(torch.nn.Module):
    # Burst Image Deblurring Using Permutation Invariant Convolutional Neural Networks
    def forward(self, input):
        n, c, d, h, w = input.size()

        input_c = input.permute(0, 3, 4, 1, 2).view(n, h * w, d * c)

        maxC = F.max_pool1d(input_c, kernel_size=d, stride=d, padding=0).permute(0, 2, 1).view(n, c, h, w)
        return maxC

class GateMoudle(nn.Module):
    def __init__(self):
        super(GateMoudle, self).__init__()

        self.conv1 = nn.Conv2d(128,  64, (3, 3), 1, 1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 64, (1, 1), 1, padding=0)

        for i in self.modules():
            if isinstance(i, nn.Conv2d):
                j = i.kernel_size[0] * i.kernel_size[1] * i.out_channels
                i.weight.data.normal_(0, math.sqrt(2 / j))
                if i.bias is not None:
                    i.bias.data.zero_()

    def forward(self, x):
        con1 = self.relu(self.conv1(x))
        scoremap = self.conv2(con1)
        return scoremap

class RDB_3DConv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=(3,3,3)):
        super(RDB_3DConv, self).__init__()
        Cin = inChannels
        G = growRate
        self.reflection_padding = (kSize[0] // 2, kSize[1] // 2, kSize[2] // 2, kSize[0] // 2, kSize[1] // 2, kSize[2] // 2) #(padLeft, padRight, padTop, padBottom, padFront, padBack)
        self.conv = nn.Sequential(*[
            nn.Conv3d(Cin, G, kSize, padding=0, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = F.pad(x, self.reflection_padding, mode='replicate')
        out = self.conv(out)
        return torch.cat((x, out), 1)


class RDB3D(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB3D, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_3DConv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv3d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        out = self.conv(x)
        return torch.cat((x, out), 1)


class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G = growRate
        C = nConvLayers

        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c * G, G))
        self.convs = nn.Sequential(*convs)

        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x)) + x

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


############################DBPN

class D_UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0

class D_DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(D_DownBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, activation, norm=None)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, activation, norm=None)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0

class UpBlockTo3D(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(UpBlockTo3D, self).__init__()
        self.conv = nn.Conv2d(num_filter*num_stages, num_filter, 1, padding=0, stride=1)

        self.up_conv1 = nn.Sequential(
            nn.ConvTranspose3d(num_filter, num_filter, (5, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.PReLU(),
            nn.ConvTranspose3d(num_filter, num_filter, (5, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        )
        self.up_conv2 = nn.Sequential(
            nn.Conv3d(num_filter, num_filter, (3, 3, 3), padding=(0, 1, 1), stride=1),
            nn.PReLU(),
            nn.Conv3d(num_filter, num_filter, (3, 3, 3), padding=(0, 1, 1), stride=1)
        )
        self.up_conv3 = nn.Sequential(
            nn.ConvTranspose3d(num_filter, num_filter, (5, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.PReLU(),
            nn.ConvTranspose3d(num_filter, num_filter, (5, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        )

    def forward(self, x):
        x = self.conv(x)
        n, c, h, w = x.size()
        x = x.view(n, c, 1, h, w)
        ft3d_0 = self.up_conv1(x)
        ft2d_0 = self.up_conv2(ft3d_0)
        ft3d_1 = self.up_conv3(ft2d_0 - x)
        return ft3d_1 + ft3d_0


class DownBlockTo2D(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(DownBlockTo2D, self).__init__()
        self.act = nn.PReLU()
        self.conv = nn.Conv3d(num_filter*num_stages, num_filter, 1, padding=0, stride=1)
        self.down_conv1 = nn.Sequential(
            nn.Conv3d(num_filter, num_filter, (3, 3, 3), padding=(0, 1, 1), stride=1),
            nn.PReLU(),
            nn.Conv3d(num_filter, num_filter, (3, 3, 3), padding=(0, 1, 1), stride=1)
        )
        self.down_conv2 = nn.Sequential(
            nn.ConvTranspose3d(num_filter, num_filter, (5, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
            nn.PReLU(),
            nn.ConvTranspose3d(num_filter, num_filter, (5, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
        )
        self.down_conv3 = nn.Sequential(
            nn.Conv3d(num_filter, num_filter, (3, 3, 3), padding=(0, 1, 1), stride=1),
            nn.PReLU(),
            nn.Conv3d(num_filter, num_filter, (3, 3, 3), padding=(0, 1, 1), stride=1)
        )

    def forward(self, x):
        x = self.conv(x)
        n, c, d, h, w = x.size()
        ft2d_0 = self.down_conv1(x)
        ft3d_0 = self.down_conv2(ft2d_0)
        ft2d_1 = self.down_conv3(ft3d_0 - x)
        return (ft2d_1 + ft2d_0).view(n, c, h, w)



class RDB_UNet(nn.Module):
    def __init__(self, inChannels, growRate, kSize=(3,3,3)):
        super(RDB_UNet, self).__init__()
        Cin = inChannels
        G = growRate
        self.reflection_padding = (kSize[0] // 2, kSize[1] // 2, kSize[2] // 2, kSize[0] // 2, kSize[1] // 2, kSize[2] // 2)
        self.prelu = nn.PReLU()
        self.conv = nn.Sequential(*[
            nn.Conv3d(Cin, G, kSize, padding=0, stride=1),
            nn.PReLU(),
        ])
        self.down_conv1 = nn.Conv3d(G, G, (3, 3, 3), padding=(0, 1, 1), stride=1)
        self.down_conv2 = nn.Conv3d(G, G, (3, 3, 3), padding=(0, 1, 1), stride=1)
        self.conv_s4 = nn.Conv3d(Cin+G, G, (3, 3, 3), padding=0, stride=1)
        self.up_conv2 = nn.ConvTranspose3d(G, G, (5, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.up_conv1 = nn.ConvTranspose3d(G, G, (5, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))

    def forward(self, x_2d, x_3d):
        out = F.pad(x_3d, self.reflection_padding, mode='replicate')
        out_s1 = self.conv(out)
        n,c,h,w = x_2d.size()

        out_s2 = self.down_conv1(out_s1)
        out_s4 = self.down_conv2(out_s2)
        out_s4 = torch.cat((x_2d.view(n,c,1, h,w), out_s4),1)
        out_2d = out_s4.view(n,-1,h,w)
        out_s4 = F.pad(out_s4, self.reflection_padding, mode='replicate')
        out_s4 = self.conv_s4(out_s4)


        res2 = self.up_conv2(out_s4) + out_s2
        res1 = self.up_conv1(res2) +  out_s1

        return out_2d, torch.cat((x_3d, res1), 1)

######################################################################
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(self, rgb_range, rgb_mean, rgb_std, sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.weight.data.div_(std.view(3, 1, 1, 1))
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean)
        self.bias.data.div_(std)
        self.requires_grad = False

class BasicBlock(nn.Sequential):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, bias=False,
        bn=True, act=nn.ReLU(True)):

        m = [nn.Conv2d(
            in_channels, out_channels, kernel_size,
            padding=(kernel_size//2), stride=stride, bias=bias)
        ]
        if bn: m.append(nn.BatchNorm2d(out_channels))
        if act is not None: m.append(act)
        super(BasicBlock, self).__init__(*m)

class ResBlock(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if i == 0: m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

class SpaceToDepth(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


############PCD_Align Module from EDVR###################
# try:
#     from networks.DCNv2.dcn_v2 import DCN_sep
# except ImportError:
#     raise ImportError('Failed to import DCNv2 module.')
#
# class PCD_Align(nn.Module):
#     ''' Alignment module using Pyramid, Cascading and Deformable convolution
#     with 3 pyramid levels.
#     '''
#
#     def __init__(self, nf=64, groups=8):
#         super(PCD_Align, self).__init__()
#         # L3: level 3, 1/4 spatial size
#         self.L3_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
#         self.L3_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.L3_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
#                                   deformable_groups=groups)
#         # L2: level 2, 1/2 spatial size
#         self.L2_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
#         self.L2_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
#         self.L2_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.L2_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
#                                   deformable_groups=groups)
#         self.L2_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
#         # L1: level 1, original spatial size
#         self.L1_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
#         self.L1_offset_conv2 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for offset
#         self.L1_offset_conv3 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#         self.L1_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
#                                   deformable_groups=groups)
#         self.L1_fea_conv = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for fea
#         # Cascading DCN
#         self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)  # concat for diff
#         self.cas_offset_conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
#
#         self.cas_dcnpack = DCN_sep(nf, nf, 3, stride=1, padding=1, dilation=1,
#                                    deformable_groups=groups)
#
#         self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
#
#     def forward(self, nbr_fea_l, ref_fea_l):
#         '''align other neighboring frames to the reference frame in the feature level
#         nbr_fea_l, ref_fea_l: [L1, L2, L3], each with [B,C,H,W] features
#         '''
#         # L3
#         L3_offset = torch.cat([nbr_fea_l[2], ref_fea_l[2]], dim=1)
#         L3_offset = self.lrelu(self.L3_offset_conv1(L3_offset))
#         L3_offset = self.lrelu(self.L3_offset_conv2(L3_offset))
#         L3_fea = self.lrelu(self.L3_dcnpack(nbr_fea_l[2], L3_offset))
#         # L2
#         L2_offset = torch.cat([nbr_fea_l[1], ref_fea_l[1]], dim=1)
#         L2_offset = self.lrelu(self.L2_offset_conv1(L2_offset))
#         L3_offset = F.interpolate(L3_offset, scale_factor=2, mode='bilinear', align_corners=False)
#         L2_offset = self.lrelu(self.L2_offset_conv2(torch.cat([L2_offset, L3_offset * 2], dim=1)))
#         L2_offset = self.lrelu(self.L2_offset_conv3(L2_offset))
#         L2_fea = self.L2_dcnpack(nbr_fea_l[1], L2_offset)
#         L3_fea = F.interpolate(L3_fea, scale_factor=2, mode='bilinear', align_corners=False)
#         L2_fea = self.lrelu(self.L2_fea_conv(torch.cat([L2_fea, L3_fea], dim=1)))
#         # L1
#         L1_offset = torch.cat([nbr_fea_l[0], ref_fea_l[0]], dim=1)
#         L1_offset = self.lrelu(self.L1_offset_conv1(L1_offset))
#         L2_offset = F.interpolate(L2_offset, scale_factor=2, mode='bilinear', align_corners=False)
#         L1_offset = self.lrelu(self.L1_offset_conv2(torch.cat([L1_offset, L2_offset * 2], dim=1)))
#         L1_offset = self.lrelu(self.L1_offset_conv3(L1_offset))
#         L1_fea = self.L1_dcnpack(nbr_fea_l[0], L1_offset)
#         L2_fea = F.interpolate(L2_fea, scale_factor=2, mode='bilinear', align_corners=False)
#         L1_fea = self.L1_fea_conv(torch.cat([L1_fea, L2_fea], dim=1))
#         # Cascading
#         offset = torch.cat([L1_fea, ref_fea_l[0]], dim=1)
#         offset = self.lrelu(self.cas_offset_conv1(offset))
#         offset = self.lrelu(self.cas_offset_conv2(offset))
#         L1_fea = self.lrelu(self.cas_dcnpack(L1_fea, offset))
#
#         return L1_fea