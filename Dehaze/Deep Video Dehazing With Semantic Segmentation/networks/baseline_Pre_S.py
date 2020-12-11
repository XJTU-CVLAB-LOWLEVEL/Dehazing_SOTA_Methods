import torch
import torch.nn as nn
import torch.nn.functional as F
import visualization as vl
from networks.PWC_Net import Backward

def make_model(args, parent=False):
    print('Now Initializing MSBDN_Pre_S...')
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

class PreNet(nn.Module):
    def __init__(self, res_blocks=1):
        super(PreNet, self).__init__()

        self.conv_input = ConvLayer(3, 16, kernel_size=11, stride=1)
        self.dense0 = nn.Sequential(
            ResidualBlock(16)
        )

        self.conv2x = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.dense1 = nn.Sequential(
            ResidualBlock(32)
        )

        self.conv4x = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.dense2 = nn.Sequential(
            ResidualBlock(64)
        )

        self.conv8x = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.dense3 = nn.Sequential(
            ResidualBlock(128)
        )

        self.conv16x = ConvLayer(128, 256, kernel_size=3, stride=2)
        #self.dense4 = Dense_Block(256, 256)

        self.dehaze = nn.Sequential()
        for i in range(0, res_blocks):
            self.dehaze.add_module('res%d' % i, ResidualBlock(256))


    def forward(self, x):
        res1x = self.conv_input(x)
        x = self.dense0(res1x) + res1x
        out = x

        res2x = self.conv2x(x)
        res2x =self.dense1(res2x) + res2x
        out_2x = res2x

        res4x =self.conv4x(res2x)
        res4x = self.dense2(res4x) + res4x
        out_4x = res4x

        res8x = self.conv8x(res4x)
        res8x = self.dense3(res8x) + res8x
        out_8x = res8x

        res16x = self.conv16x(res8x)
        #res16x = self.dense4(res16x)
        res_dehaze = res16x
        res16x = self.dehaze(res_dehaze) + res_dehaze
        out_16x = res16x


        return [out, out_2x, out_4x, out_8x, out_16x]

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
    def __init__(self, opt, res_blocks=16):
        super(Net, self).__init__()

        self.warped = opt.warped
        self.frames = opt.frames
        self.pre_net = PreNet()

        if self.warped == 3:
            self.moduleNetwork = Flow_Net()

        self.fuse_1 = ConvLayer(16 * self.frames, 16, kernel_size=3, stride=1)
        self.fuse_2 = ConvLayer(16 * 2, 16, kernel_size=3, stride=1)
        self.dense0 = nn.Sequential(
            ResidualBlock(16),
            ResidualBlock(16)
        )

        self.conv2x = ConvLayer(16, 32, kernel_size=3, stride=2)
        self.fuse2x_1 = ConvLayer(32 * self.frames, 32, kernel_size=3, stride=1)
        self.fuse2x_2 = ConvLayer(32 * 2, 32, kernel_size=3, stride=1)
        self.dense1 = nn.Sequential(
            ResidualBlock(32),
            ResidualBlock(32)
        )

        self.conv4x = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.fuse4x_1 = ConvLayer(64 * self.frames, 64, kernel_size=3, stride=1)
        self.fuse4x_2 = ConvLayer(64 * 2, 64, kernel_size=3, stride=1)
        self.dense2 = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64)
        )

        self.conv8x = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.fuse8x_1 = ConvLayer(128 * self.frames, 128, kernel_size=3, stride=1)
        self.fuse8x_2 = ConvLayer(128 * 2, 128, kernel_size=3, stride=1)
        self.dense3 = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128)
        )

        self.conv16x = ConvLayer(128, 256, kernel_size=3, stride=2)
        self.fuse16x_1 = ConvLayer(256 * self.frames, 256, kernel_size=3, stride=1)
        self.fuse16x_2 = ConvLayer(256 * 2, 256, kernel_size=3, stride=1)
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

        #Pre dehazing
        n, c, h, w = x.size()

        x = x.view(-1, 3, h, w)
        # x1 = x.view(n,-1,h,w)
        [out, out_2x, out_4x, out_8x, out_16x] = self.pre_net(x)

        N, C, H, W = out.size()
        out = out.view(n, -1, H, W)
        N, C2, H2, W2 = out_2x.size()
        out_2x = out_2x.view(n, -1, H2, W2)
        N, C4, H4, W4 = out_4x.size()
        out_4x = out_4x.view(n, -1, H4, W4)
        N, C8, H8, W8 = out_8x.size()
        out_8x = out_8x.view(n, -1, H8, W8)
        N, C16, H16, W16 = out_16x.size()
        out_16x = out_16x.view(n, -1, H16, W16)

        if self.warped == 2:
            base = []
            base2x = []
            base4x = []
            base8x = []
            base16x = []
            for i in range(self.frames):
                if i == self.frames // 2:
                    base.append(out[:, i * C:(i + 1) * C])
                    base2x.append(out_2x[:, i * C2:(i + 1) * C2])
                    base4x.append(out_4x[:, i * C4:(i + 1) * C4])
                    base8x.append(out_8x[:, i * C8:(i + 1) * C8])
                    base16x.append(out_16x[:, i * C16:(i + 1) * C16])
                    continue
                base.append(
                    Backward(tensorInput=out[:, i * C:(i + 1) * C], tensorFlow=flowmaps[:, i * 2:(i + 1) * 2]))
                base2x.append(
                    Backward(tensorInput=out_2x[:, i * C2:(i + 1) * C2],
                             tensorFlow= (1/2) * F.interpolate(flowmaps[:, i * 2:(i + 1) * 2], size=(H2, W2), mode='bilinear', align_corners=False)))
                base4x.append(
                    Backward(tensorInput=out_4x[:, i * C4:(i + 1) * C4],
                             tensorFlow= (1/4) * F.interpolate(flowmaps[:, i * 2:(i + 1) * 2], size=(H4, W4), mode='bilinear', align_corners=False)))
                base8x.append(
                    Backward(tensorInput=out_8x[:, i * C8:(i + 1) * C8],
                             tensorFlow= (1/8) * F.interpolate(flowmaps[:, i * 2:(i + 1) * 2], size=(H8, W8), mode='bilinear', align_corners=False)))
                base16x.append(
                    Backward(tensorInput=out_16x[:, i * C16:(i + 1) * C16],
                             tensorFlow= (1/16) * F.interpolate(flowmaps[:, i * 2:(i + 1) * 2], size=(H16, W16), mode='bilinear', align_corners=False)))

            out = torch.cat(base, 1)
            out_2x = torch.cat(base2x, 1)
            out_4x = torch.cat(base4x, 1)
            out_8x = torch.cat(base8x, 1)
            out_16x = torch.cat(base16x, 1)


        # dehazing
        center = self.frames // 2
        input = out[:, C*center:C*(center+1)]
        res1x = torch.cat((self.fuse_1(out), input), dim=1)
        res1x = self.fuse_2(res1x)
        x = self.dense0(res1x) + res1x


        res2x = self.conv2x(x)
        res2x = torch.cat((self.fuse2x_1(out_2x), res2x), dim=1)
        res2x = self.fuse2x_2(res2x)
        res2x =self.dense1(res2x) + res2x


        res4x =self.conv4x(res2x)
        res4x = torch.cat((self.fuse4x_1(out_4x), res4x), dim=1)
        res4x = self.fuse4x_2(res4x)
        res4x = self.dense2(res4x) + res4x

        res8x = self.conv8x(res4x)
        res8x = torch.cat((self.fuse8x_1(out_8x), res8x), dim=1)
        res8x = self.fuse8x_2(res8x)
        res8x = self.dense3(res8x) + res8x

        res16x = self.conv16x(res8x)
        res16x = torch.cat((self.fuse16x_1(out_16x), res16x), dim=1)
        res16x = self.fuse16x_2(res16x)
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