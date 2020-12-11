import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet101
import matplotlib.pyplot as plt

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
from torchvision import transforms
import visualization as vl

def make_model(args, parent=False):
    return  VDHNet(args)

def show_image(image, is_Variable=True, Interpolation='bilinear'):
# input 4D Varialble or Tensor (N*C*H*W)
    if is_Variable:
        img = image.cpu().data[0].numpy().transpose((1,2,0))
    else:
        img = image[0].numpy().transpose((1,2,0))

    plt.imshow(img)
    plt.axis('off')
    plt.show()


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,dilation):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.relu=nn.ReLU(inplace=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        out =self.relu(out)
        return out

class ConvLayer1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,dilation):
        super(ConvLayer1, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride,dilation)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      # reflection_padding = kernel_size // 2
      # self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
      self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        # out= self.reflection_pad(x)
        out = self.conv2d(x)
        out = self.relu(out)
        return out


class VDHNet(torch.nn.Module):
    def __init__(self, opt):
        super(VDHNet, self).__init__()

        self.frames = opt.frames
        self.warped = opt.warped


        self.conv_input = ConvLayer(15, 64, kernel_size=5, stride=1,dilation=1)

       # self.segment= torchvision.models.segmentation.deeplabv3_resnet101(pretrained=False, progress=True, num_classes=1, aux_loss=None)
        self.pool= nn.MaxPool2d(kernel_size=3,stride=2,padding=1,dilation=1)
        self.con0x=nn.Conv2d(5, 64, kernel_size=1, stride=1,dilation=1)

        self.conv21x = ConvLayer(64, 128, kernel_size=3, stride=2,dilation=2)
        self.conv22x = ConvLayer(128, 128, kernel_size=3, stride=1,dilation=2)
        self.conv23x = ConvLayer(128, 128, kernel_size=3, stride=1,dilation=2)

        self.conv31x = ConvLayer(128, 256, kernel_size=3, stride=2, dilation=2)
        self.conv32x = ConvLayer(256, 256, kernel_size=3, stride=1, dilation=2)
        self.conv33x = ConvLayer(256, 256, kernel_size=3, stride=1, dilation=2)

        self.conv41x = ConvLayer(256, 512, kernel_size=3, stride=2, dilation=2)
        self.conv42x = ConvLayer(512, 512, kernel_size=3, stride=1, dilation=2)
        self.conv43x = ConvLayer(512, 512, kernel_size=3, stride=1, dilation=2)

        self.convd1x = UpsampleConvLayer(576, 256, kernel_size=3, stride=2)

        self.conv51x = ConvLayer(256, 256, kernel_size=3, stride=1, dilation=2)
        self.conv52x = ConvLayer(256, 256, kernel_size=3, stride=1, dilation=2)

        self.convd2x = UpsampleConvLayer(256, 128, kernel_size=3, stride=2)
        self.conv61x = ConvLayer(128, 128, kernel_size=3, stride=1, dilation=2)
        self.conv62x = ConvLayer(128, 128, kernel_size=3, stride=1, dilation=2)

        self.convd3x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.conv71x = ConvLayer(64, 64, kernel_size=3, stride=1, dilation=2)
        # self.conv72x = ConvLayer(64, 64, kernel_size=3, stride=1, dilation=2)
        #
        # #self.dense4 = Dense_Block(256, 256)
        #
        # self.conv_output = ConvLayer1(64, 3, kernel_size=3, stride=1,dilation=0)
        self.conv73x = ConvLayer(64, 3, kernel_size=3, stride=1, dilation=2)

    def forward(self, x,y):

        # n, c, h, w = x.size()
        # frames = c // 3
        # inputs = []
        # for i in range(frames):
        #     f__1 = x[:, i * 3:(i + 1) * 3]
        #     inputs.append(f__1)
        # inputs = torch.cat(inputs, 1).view(n,frames, 3, h, w).permute(0, 1, 2, 3, 4).contiguous()
        #
        # a1=inputs.view(-1,3,w,h)
        #
        # seg=self.segment(a1)['out']
        # j=seg[0:1, 0:1]
        # print(j)
        # vl.show_image(seg[0:1, 0:1])
        # show_image(seg[0:1, 0:1])
        # a,b,d,d = seg.size()
        # frames = a // 5
        # inputs = []
        # for i in range(frames):
        #     f__1 = seg[i * 5:(i + 1) * 5]
        #     inputs.append(f__1)
        # e = torch.cat(inputs, 0).view(frames, 5, 1, h, w).permute(0, 1, 2, 3, 4).contiguous()
        # e1 = e.view(4, -1, w, h)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        a,b,c,d=x.size()
       # y = torch.zeros(a, 5, 256, 256).cuda()
        seg1=self.con0x(y)
        seg2=self.pool(seg1)
        seg3=self.pool(seg2)
        seg4=self.pool(seg3)
        res11x = self.conv_input(x)


        res21x = self.conv21x(res11x)
        res22x =self.conv22x(res21x)
        res23x = self.conv23x(res22x)

        res31x = self.conv31x(res23x)
        res32x = self.conv32x(res31x)
        res33x = self.conv33x(res32x)

        res41x = self.conv41x(res33x)
        res42x = self.conv42x(res41x)
        res43x = self.conv43x(res42x)
       # print(res43x.shape)
        #print(seg4.shape)
        res4x=torch.cat((res43x,seg4),dim=1)

        res51x=self.convd1x(res4x)
        res51x = F.upsample(res51x, res33x.size()[2:], mode='bilinear')
        res5x=torch.add(res51x,res33x)
        res52x=self.conv51x(res5x)
        res53x=self.conv52x(res52x)
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        res61x = self.convd2x(res53x)
        res61x = F.upsample(res61x, res23x.size()[2:], mode='bilinear')
        res6x = torch.add(res61x, res23x)
        res62x = self.conv61x(res6x)
        res63x = self.conv62x(res62x)

        res71x = self.convd3x(res63x)
        res71x = F.upsample(res71x, res11x.size()[2:], mode='bilinear')
        res7x = torch.add(res71x, res11x)
        res72x = self.conv71x(res7x)
        # res73x = self.conv72x(res72x)
        #
        # transmiaaion = self.conv_output(res73x)
       # vl.show_image(transmiaaion[:, 0:1])
        #print(transmiaaion)
        transmission = self.conv73x(res72x)
        return transmission
