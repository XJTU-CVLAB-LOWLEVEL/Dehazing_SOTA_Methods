import torch
import torch.nn as nn
import torch.nn.functional as F
import visualization as vl

def make_model(args, parent=False):
    print('Now Initializing V-MSBDN...')
    return Net()

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

class ResidualBlock3D(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, (3, 3, 3), padding=(1, 1, 1), stride=1)
        self.conv2 = nn.Conv3d(channels, channels, (3, 3, 3), padding=(1, 1, 1), stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out

class Net(nn.Module):
    def __init__(self, res_blocks=18):
        super(Net, self).__init__()

        self.conv_input = ConvLayer(3, 16, kernel_size=11, stride=1)
        self.dense0 = nn.Sequential(
            ResidualBlock3D(16),
            ResidualBlock3D(16),
            ResidualBlock3D(16)
        )
        self.GFF0 = nn.Sequential(*[
            nn.Conv3d(16, 16, (3, 3, 3), padding=(0, 1, 1), stride=1),
            nn.Conv3d(16, 16, (3, 3, 3), padding=(0, 1, 1), stride=1),
        ])

        self.conv2x = nn.Conv3d(16, 32, (3, 3, 3), padding=(1, 1, 1), stride=(1,2,2))
        self.dense1 = nn.Sequential(
            ResidualBlock3D(32),
            ResidualBlock3D(32),
            ResidualBlock3D(32)
        )
        self.GFF1 = nn.Sequential(*[
            nn.Conv3d(32, 32, (3, 3, 3), padding=(0, 1, 1), stride=1),
            nn.Conv3d(32, 32, (3, 3, 3), padding=(0, 1, 1), stride=1),
        ])

        self.conv4x = nn.Conv3d(32, 64, (3, 3, 3), padding=(1, 1, 1), stride=(1,2,2))
        self.dense2 = nn.Sequential(
            ResidualBlock3D(64),
            ResidualBlock3D(64),
            ResidualBlock3D(64)
        )
        self.GFF2 = nn.Sequential(*[
            nn.Conv3d(64, 64, (3, 3, 3), padding=(0, 1, 1), stride=1),
            nn.Conv3d(64, 64, (3, 3, 3), padding=(0, 1, 1), stride=1),
        ])

        self.conv8x = nn.Conv3d(64, 128, (3, 3, 3), padding=(1, 1, 1), stride=(1,2,2))
        self.dense3 = nn.Sequential(
            ResidualBlock3D(128),
            ResidualBlock3D(128),
            ResidualBlock3D(128)
        )
        self.GFF3 = nn.Sequential(*[
            nn.Conv3d(128, 128, (3, 3, 3), padding=(0, 1, 1), stride=1),
            nn.Conv3d(128, 128, (3, 3, 3), padding=(0, 1, 1), stride=1),
        ])

        self.conv16x = nn.Conv3d(128, 256, (3, 3, 3), padding=(1, 1, 1), stride=(1,2,2))
        #self.dense4 = Dense_Block(256, 256)

        self.dehaze3D = nn.Sequential()
        for i in range(0, 5):
            self.dehaze3D.add_module('res%d' % i, ResidualBlock3D(256))
        self.GFF4 = nn.Sequential(*[
            nn.Conv3d(256, 256, (3, 3, 3), padding=(0, 1, 1), stride=1),
            nn.Conv3d(256, 256, (3, 3, 3), padding=(0, 1, 1), stride=1),
        ])
        self.dehaze2D = nn.Sequential()
        for i in range(0, 5):
            self.dehaze2D.add_module('res%d' % i, ResidualBlock(256))


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


    def forward(self, x):
        n, c, h, w = x.size()
        frames = c // 3
        inputs = []
        for i in range(frames):
            f__1 = self.conv_input(x[:, i * 3:(i + 1) * 3])
            if i == frames // 2:
                base = f__1
            inputs.append(f__1)

        res1x_3D = torch.cat(inputs, 1).view(n, frames, 16, h, w).permute(0, 2, 1, 3, 4).contiguous()
        res1x_3D = self.dense0(res1x_3D) + res1x_3D
        res1x = self.GFF0(res1x_3D).squeeze(2)


        res2x_3D = self.conv2x(res1x_3D)
        res2x_3D = self.dense1(res2x_3D) + res2x_3D
        res2x = self.GFF1(res2x_3D).squeeze(2)

        res4x_3D = self.conv4x(res2x_3D)
        res4x_3D = self.dense2(res4x_3D) + res4x_3D
        res4x = self.GFF2(res4x_3D).squeeze(2)

        res8x_3D = self.conv8x(res4x_3D)
        res8x_3D = self.dense3(res8x_3D) + res8x_3D
        res8x = self.GFF3(res8x_3D).squeeze(2)

        res16x_3D = self.conv16x(res8x_3D)
        #res16x = self.dense4(res16x)

        res_dehaze_3D = res16x_3D
        res16x_3D = self.dehaze3D(res16x_3D) + res_dehaze_3D
        res16x = self.GFF4(res16x_3D).squeeze(2)
        res_dehaze = res16x

        in_ft = res16x*2
        res16x = self.dehaze2D(in_ft) + in_ft - res_dehaze

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
        res2x = F.upsample(res2x, res1x.size()[2:], mode='bilinear')
        res1x = torch.add(res2x, res1x)
        x = self.dense_1(res1x) + res1x - res2x

        x = self.conv_output(x)

        return x