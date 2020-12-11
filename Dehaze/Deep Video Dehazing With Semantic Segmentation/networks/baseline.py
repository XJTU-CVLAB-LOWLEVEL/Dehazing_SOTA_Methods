import torch
import torch.nn as nn
import torch.nn.functional as F
import visualization as vl
from networks.PWC_Net import Backward

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

class Net(nn.Module):
    def __init__(self, opt, res_blocks=18):
        super(Net, self).__init__()

        self.frames = opt.frames
        self.warped = opt.warped

        if self.warped !=2:
            self.conv_input = ConvLayer(3* self.frames, 16, kernel_size=11, stride=1)
        else:
            self.conv_input = ConvLayer(3, 16, kernel_size=11, stride=1)
            self.conv_fuse = ConvLayer(16 * self.frames, 16, kernel_size=11, stride=1)

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


    def forward(self, x):

        res1x = self.conv_input(x)


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
        out_16x = res16x

        res16x = self.convd16x(res16x)
        res16x = F.upsample(res16x, res8x.size()[2:], mode='bilinear')
        res8x = torch.add(res16x, res8x)
        res8x = self.dense_4(res8x) + res8x - res16x
        out_8x = res8x

        res8x = self.convd8x(res8x)
        res8x = F.upsample(res8x, res4x.size()[2:], mode='bilinear')
        res4x = torch.add(res8x, res4x)
        res4x = self.dense_3(res4x) + res4x - res8x
        out_4x = res4x

        res4x = self.convd4x(res4x)
        res4x = F.upsample(res4x, res2x.size()[2:], mode='bilinear')
        res2x = torch.add(res4x, res2x)
        res2x = self.dense_2(res2x) + res2x - res4x
        out_2x = res2x

        res2x = self.convd2x(res2x)
        res2x = F.upsample(res2x, x.size()[2:], mode='bilinear')
        x = torch.add(res2x, x)
        x = self.dense_1(x) + x - res2x
        out = x

        dehaze = self.conv_output(x)

        return dehaze

import torch.optim as optim
import argparse
import os
from os.path import join
import torch
from torch.utils.data import DataLoader
from datasets.dataset import DataSetV, DataValSetV
from importlib import import_module
import random
import re
from networks.warp_image import estimate, Network
from networks.base_networks import SpaceToDepth
from networks.warp_image import Backward
import visualization as vl
import time
from math import log10
import torch.nn.functional as F
from vgg_loss import LossNetwork
from torchvision.models import vgg16
from hard_example_mining import HEM, HEM_CUDA
input=torch.rand((5,15,256,256))
parser = argparse.ArgumentParser(description="PyTorch Train")
parser.add_argument("--batchSize", type=int, default=5, help="Training batch size")
parser.add_argument("--start_training_step", type=int, default=1, help="Training step")
parser.add_argument("--nEpochs", type=int, default=60, help="Number of epochs to train")
parser.add_argument("--start-epoch", type=int, default=1, help="Start epoch from 1")

parser.add_argument('--dataset', default="/home/lry/VideoHazy_v2_re_syn/Train", type=str, help='Path of the training dataset(.h5)')
parser.add_argument('--dataset_test', default="/home/lry/VideoHazy_v2_re_syn/Test", type=str, help='Path of the training dataset(.h5)')
parser.add_argument('--flow_dir', default="hazy", type=str, help='Path of the training dataset(.h5)')
parser.add_argument("--scale", default=1, type=int, help="Scale factor, Default: 4")
parser.add_argument("--cropSize", type=int, default=256, help="LR patch size")
parser.add_argument("--frames", type=int, default=5, help="the amount of input frames")
parser.add_argument("--repeat", type=int, default=1, help="the amount of the dataset repeat per epoch")

# parser.add_argument("--pre", type=bool, default=False, help="Activated Pre-Dehazing Module")
parser.add_argument("--pre", default=0, type=int, help="Ways of Pre-Dehazing Module, 0: No Pre-Dehazing / 1: Pre-Dehazing / 2: Pre-Dehazing and Finetune")
parser.add_argument("--warped", default=0, type=int, help="Ways of Alignment, 0: No Alignment / 1: Input Alignment / 2: Feature Alignment / 3: Feature-based flow")
parser.add_argument("--tof", type=bool, default=False, help="Activated PWC-Net finwtuning")
parser.add_argument("--hr_flow", type=bool, default=False, help="Activated hr_flow")
parser.add_argument("--residual", type=bool, default=False, help="Activated hr_flow")
parser.add_argument("--cobi", type=bool, default=False, help="Use CoBi Loss")
parser.add_argument("--isTest", type=bool, default=False, help="Test or not")
parser.add_argument("--vgg", type=bool, default=False, help="Activated vgg loss")
parser.add_argument("--hem", type=bool, default=False, help="Activated hard negative mining")

parser.add_argument('--model', default='baseline', type=str, help='Import which network')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--name', default='baseline_LR_flow', type=str, help='Filename of the training models')
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")

parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate, default=1e-5")
parser.add_argument("--step", type=int, default=7, help="Change the learning rate for every 30 epochs")
parser.add_argument("--lr_decay", type=float, default=0.5, help="Decay scale of learning rate, default=0.5")
parser.add_argument("--lambda_db", type=float, default=0.5, help="Weight of deblurring loss, default=0.5")
parser.add_argument("--lambda_GL", type=float, default=0.5, help="Weight of deblurring loss, default=0.5")

# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='A',
                    help='parameters config of RDN. (Use in RDN)')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
opt = parser.parse_args()
net=Net(opt)
out=net(input)