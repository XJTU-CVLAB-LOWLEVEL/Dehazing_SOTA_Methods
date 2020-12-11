
# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# --- Build dense --- #
class MakeDense(nn.Module):
    def __init__(self, in_channels, growth_rate, kernel_size=3, dilation = 1):
        super(MakeDense, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=kernel_size, padding=dilation, dilation=dilation)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out

class AdaIn(nn.Module):
    def __init__(self):
        super(AdaIn, self).__init__()
        self.eps = 1e-5

    def forward(self, x, mean_style, std_style):
        B, C, H, W = x.shape

        feature = x.view(B, C, -1)

        #print (mean_feat.shape, std_feat.shape, mean_style.shape, std_style.shape)
        std_style = std_style.view(B, C, 1)
        mean_style = mean_style.view(B, C, 1)
        adain = std_style * (feature) + mean_style

        adain = adain.view(B, C, H, W)
        return adain

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 4, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 4, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 4, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 4, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class ApplyNoise(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(channels))

    def forward(self, x, noise):
        if noise is None:
            noise = torch.randn(x.size(0), 1, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return x + self.weight.view(1, -1, 1, 1) * noise.to(x.device)

# --- Build the Residual Dense Block --- #
class RDB(nn.Module):
    def __init__(self, in_channels, num_dense_layer, growth_rate, dilations = [1, 1, 1, 1]):
        """

        :param in_channels: input channel size
        :param num_dense_layer: the number of RDB layers
        :param growth_rate: growth_rate
        """
        super(RDB, self).__init__()
        _in_channels = in_channels
        modules = []
        for i in range(num_dense_layer):
            modules.append(MakeDense(_in_channels, growth_rate, dilation = dilations[i]))
            _in_channels += growth_rate
        self.residual_dense_layers = nn.Sequential(*modules)

        self.conv_1x1_a = nn.Conv2d(_in_channels, in_channels, kernel_size=1, padding=0)


        _in_channels_no_style = in_channels
        no_style_modules = []
        for i in range(num_dense_layer):
            no_style_modules.append(MakeDense(_in_channels_no_style, growth_rate))
            _in_channels_no_style += growth_rate

        self.residual_dense_layers_no_style = nn.Sequential(*no_style_modules)
        self.conv_1x1_b = nn.Conv2d(_in_channels_no_style, in_channels, kernel_size=1, padding=0)

        self.norm = nn.InstanceNorm2d(in_channels)
        self.norm2 = nn.InstanceNorm2d(in_channels)
        self.adaIn = AdaIn()

        self.global_feat = nn.AdaptiveAvgPool2d((1, 1))
        self.style = nn.Linear(in_channels // 2, in_channels * 2)
        self.conv_1x1_style = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)

        self.conv_gamma = nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1)
        self.conv_beta = nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1)

        self.conv_att = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)

        self.in_channels = in_channels

        self.conv_1x1_final = nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, padding=0)

        self.coefficient = nn.Parameter(torch.Tensor(np.ones((1, 2))), requires_grad=True)
        self.ca = CALayer(in_channels)
        self.pool = nn.AvgPool2d((7, 7), stride=(1, 1), padding=(3, 3))

        #self.noise = ApplyNoise(in_channels)

    def forward(self, x):
        # residual
        bottle_feat = self.residual_dense_layers(x)
        out = self.conv_1x1_a(bottle_feat)
        out = out + x

        # base residual， self-guieded learn mean，std，gamma，and beta
        style_feat_1 = F.relu(self.conv_1x1_style(out))
        style_feat = self.global_feat(style_feat_1)
        style_feat = torch.flatten(style_feat, start_dim = 1)
        style_feat = self.style(style_feat)
        # mean, std
        style_mean = style_feat[:, :self.in_channels]
        style_std = style_feat[:, self.in_channels:]

        gamma = self.conv_gamma(style_feat_1)
        beta = self.conv_beta(style_feat_1)

        y = self.norm(x)
        out_no_style = self.residual_dense_layers_no_style(y)
        out_no_style = self.conv_1x1_b(out_no_style)
        out_no_style = y + out_no_style
        #out_no_style = self.noise(out_no_style, None)
        out_no_style = self.norm2(out_no_style)
        out_att = torch.sigmoid(self.conv_att(out_no_style))

        out_new_style = self.adaIn(out_no_style, style_mean , style_std)
        out_new_gamma = out_no_style * (1 + gamma) + beta
        out_new = out_att * out_new_style + (1 - out_att) * out_new_gamma
        out = self.conv_1x1_final(torch.cat([out, out_new], dim = 1))
        out = self.ca(out)
        out = out + x
        return out
