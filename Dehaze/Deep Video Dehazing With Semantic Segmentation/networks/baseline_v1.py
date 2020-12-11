import torch
import torch.nn as nn
import torch.nn.functional as F
import visualization as vl
from networks.PWC_Net import Backward
# from networks.warp_feature import estimate_ft, Network

def make_model(args, parent=False):
    return Net(args)

class make_dense(nn.Module):
  def __init__(self, nChannels, growthRate, kernel_size=3):
    super(make_dense, self).__init__()
    self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size-1)//2, bias=False)
  def forward(self, x):
    out = F.relu(self.conv(x))
    out = torch.cat((x, out), 1)
    return out

# Residual dense block (RDB) architecture
class RDB(nn.Module):
  def __init__(self, nChannels, nDenselayer, growthRate, scale = 1.0):
    super(RDB, self).__init__()
    nChannels_ = nChannels
    self.scale = scale
    modules = []
    for i in range(nDenselayer):
        modules.append(make_dense(nChannels_, growthRate))
        nChannels_ += growthRate
    self.dense_layers = nn.Sequential(*modules)
    self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)
  def forward(self, x):
    out = self.dense_layers(x)
    out = self.conv_1x1(out) * self.scale
    out = out + x
    return out

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      reflection_padding = kernel_size // 2
      self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out

class Flow_Net(nn.Module):
    def __init__(self, res_blocks=2):
        super(Flow_Net, self).__init__()

        self.conv_input = ConvLayer(34, 32, kernel_size=3, stride=1)
        self.dense0 = nn.Sequential(
            ResidualBlock(32)
        )

        self.conv2x = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.dense1 = nn.Sequential(
            ResidualBlock(64)
        )

        self.conv4x = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.dehaze = nn.Sequential()
        for i in range(0, res_blocks):
            self.dehaze.add_module('res%d' % i, ResidualBlock(128))

        self.convd4x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.dense_2 = nn.Sequential(
            ResidualBlock(64)
        )

        self.convd2x = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
        self.dense_1 = nn.Sequential(
            ResidualBlock(32)
        )

        self.conv_output = ConvLayer(32, 2, kernel_size=3, stride=1)


    def forward(self, tensorAnchor, tensorSecond, flowmap):

        x = torch.cat((tensorAnchor, tensorSecond, flowmap), dim=1)
        res1x = self.conv_input(x)
        x = self.dense0(res1x) + res1x

        res2x = self.conv2x(x)
        res2x =self.dense1(res2x) + res2x

        res4x =self.conv4x(res2x)

        res_dehaze = res4x
        res4x = self.dehaze(res4x) + res_dehaze

        res4x = self.convd4x(res4x)
        res4x = F.upsample(res4x, res2x.size()[2:], mode='bilinear')
        res2x = torch.add(res4x, res2x)
        res2x = self.dense_2(res2x) + res2x


        res2x = self.convd2x(res2x)
        res2x = F.upsample(res2x, x.size()[2:], mode='bilinear')
        x = torch.add(res2x, x)
        x = self.dense_1(x) + x

        flowmap_refine = self.conv_output(x)

        return flowmap_refine

class Net(nn.Module):
    def __init__(self, opt, res_blocks=18):
        super(Net, self).__init__()

        self.frames = opt.frames
        self.warped = opt.warped

        if self.warped == 0 or self.warped == 1:
            self.conv_input = ConvLayer(3* self.frames, 16, kernel_size=11, stride=1)
        elif self.warped == 2:
            self.conv_input = ConvLayer(3, 16, kernel_size=11, stride=1)
            self.conv_fuse = ConvLayer(16 * self.frames, 16, kernel_size=1, stride=1)
        elif self.warped == 3:
            self.conv_input = ConvLayer(3, 16, kernel_size=11, stride=1)
            self.conv_fuse = ConvLayer(16 * self.frames, 16, kernel_size=1, stride=1)
            self.moduleNetwork = Flow_Net()

        self.dense0 = nn.Sequential(
            ResidualBlock(16),
            ResidualBlock(16),
            ResidualBlock(16)
        )

        self.conv2x = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.dense1 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )

        self.conv4x = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.dense2 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        self.conv8x = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.dense3 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        self.conv16x = ConvLayer(128, 256, kernel_size=3, stride=2)
        #self.dense4 = Dense_Block(256, 256)

        self.dehaze = nn.Sequential()
        for i in range(0, res_blocks):
            self.dehaze.add_module('res%d' % i, ResidualBlock(256))

        self.convd16x = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.dense_4 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        self.convd8x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.dense_3 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        self.convd4x = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
        self.dense_2 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )

        self.convd2x = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)
        self.dense_1 = nn.Sequential(
            ResidualBlock(16),
            ResidualBlock(16),
            ResidualBlock(16)
        )

        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)


    def forward(self, x, flowmaps):
        if self.warped == 0 or self.warped == 1:
            res1x = self.conv_input(x)
        elif self.warped == 2:
            n, c, h, w = x.size()
            x = x.view(-1, 3, h, w)
            # x1 = x.view(n,-1,h,w)
            x = self.conv_input(x)
            x = x.view(n, -1, h, w)
            C = x.size()[1] // self.frames

            # feature alignment
            base_3d = []
            for i in range(self.frames):
                if i == self.frames // 2:
                    base_3d.append(x[:, i * C:(i + 1) * C])
                    continue
                base_3d.append(
                    Backward(tensorInput=x[:, i * C:(i + 1) * C], tensorFlow=flowmaps[:, i * 2:(i + 1) * 2]))
            res1x = self.conv_fuse(torch.cat(base_3d, 1))
        elif self.warped == 3:
            n, c, h, w = x.size()
            x = x.view(-1, 3, h, w)
            # x1 = x.view(n,-1,h,w)
            x = self.conv_input(x)
            x = x.view(n, -1, h, w)
            C = x.size()[1] // self.frames

            mid_frame = self.frames // 2
            tensorAnchor = x[:, mid_frame * C:(mid_frame + 1) * C]
            base_3d = []
            for i in range(self.frames):
                if i == self.frames // 2:
                    base_3d.append(tensorAnchor)
                    continue
                tensorSecond = x[:, i * C:(i + 1) * C]
                flowmap = flowmaps[:, i * 2:(i + 1) * 2]
                flowmap_refine = self.moduleNetwork(tensorAnchor, tensorSecond, flowmap)
                base_3d.append(
                    Backward(tensorInput=tensorSecond, tensorFlow=flowmap_refine))
            res1x = self.conv_fuse(torch.cat(base_3d, 1))

        x = self.dense0(res1x) + res1x

        res2x = self.conv2x(x)
        res2x =self.dense1(res2x) + res2x

        res4x =self.conv4x(res2x)
        res4x = self.dense2(res4x) + res4x

        res8x = self.conv8x(res4x)
        res8x = self.dense3(res8x) + res8x

        res16x = self.conv16x(res8x)
        #res16x = self.dense4(res16x)

        res_dehaze = res16x
        in_ft = res16x*2
        res16x = self.dehaze(in_ft) + in_ft - res_dehaze

        res16x = self.convd16x(res16x)
        res16x = F.upsample(res16x, res8x.size()[2:], mode='bilinear')
        res8x = torch.add(res16x, res8x)
        res8x = self.dense_4(res8x) + res8x - res16x

        res8x = self.convd8x(res8x)
        res8x = F.upsample(res8x, res4x.size()[2:], mode='bilinear')
        res4x = torch.add(res8x, res4x)
        res4x = self.dense_3(res4x) + res4x - res8x

        res4x = self.convd4x(res4x)
        res4x = F.upsample(res4x, res2x.size()[2:], mode='bilinear')
        res2x = torch.add(res4x, res2x)
        res2x = self.dense_2(res2x) + res2x - res4x

        res2x = self.convd2x(res2x)
        res2x = F.upsample(res2x, x.size()[2:], mode='bilinear')
        x = torch.add(res2x, x)
        x = self.dense_1(x) + x - res2x

        x = self.conv_output(x)

        return x