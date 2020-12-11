# python 2.7, pytorch 0.3.1

import os, sys
import time

sys.path.insert(1, '../')
import torch
import torchvision
import numpy as np
import subprocess
import random
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image

from pietorch import data_convertors
from pietorch.DuRN_US import cleaner
from pietorch.pytorch_ssim import ssim as ssim
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ski_ssim
"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: utils.py
about: all utilities
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import time
import torch
import torch.nn.functional as F
import torchvision.utils as utils
from math import log10
from skimage import measure


def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list


def to_ssim_skimage(dehaze, gt):
    dehaze_list = torch.split(dehaze, 1, dim=0)
    gt_list = torch.split(gt, 1, dim=0)

    dehaze_list_np = [dehaze_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    gt_list_np = [gt_list[ind].permute(0, 2, 3, 1).data.cpu().numpy().squeeze() for ind in range(len(dehaze_list))]
    ssim_list = [measure.compare_ssim(dehaze_list_np[ind],  gt_list_np[ind], data_range=1, multichannel=True) for ind in range(len(dehaze_list))]

    return ssim_list


def validation(net, val_data_loader, category, save_tag=False):
    """
    :param net: GateDehazeNet
    :param val_data_loader: validation loader
    :param device: The GPU that loads the network
    :param category: indoor or outdoor test dataset
    :param save_tag: tag of saving image or not
    :return: average PSNR value
    """
    psnr_list = []
    ssim_list = []

    for batch_id, val_data in enumerate(val_data_loader):

        with torch.no_grad():
            haze, gt, image_name = val_data
            haze = haze.to(device)
            gt = gt.to(device)
            dehaze = net(haze)

        # --- Calculate the average PSNR --- #
        psnr_list.extend(to_psnr(dehaze, gt))

        # --- Calculate the average SSIM --- #
        ssim_list.extend(to_ssim_skimage(dehaze, gt))

        # --- Save image --- #
        if save_tag:
            save_image(dehaze, image_name, category)

    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim


def save_image(dehaze, image_name, category):
    dehaze_images = torch.split(dehaze, 1, dim=0)
    batch_num = len(dehaze_images)

    for ind in range(batch_num):
        utils.save_image(dehaze_images[ind], './{}_results/{}'.format(category, image_name[ind][:-4] + '_fog.png'))

#------- Option --------
tag = 'DuRN_US'
# Choose a dataset.
data_name = 'RESIDE' # 'DCPDNData' or 'RESIDE'
#-----------------------
category = 'indoor_'
if data_name == 'RESIDE':
    # testroot = "../data/"+data_name+"/sots_indoor_test/"
    # test_list_pth = "../lists/RESIDE_indoor/sots_test_list2.txt"
    testroot = "../data/RESIDE/"
    test_list_pth = "../listsRESIDE_indoor/sots_test_list.txt"
elif data_name == 'DCPDNData':
    testroot = "../data/"+data_name+"/TestA/"
    test_list_pth = '../lists/'+data_name+'/testA_list.txt'
else:
    print('Unknown dataset name.')

Pretrained = '../train/trainedmodels/'+data_name+'/'+tag+'/DURN.pt'

show_dst = '../cleaned_images/'+data_name+'/'+tag+'/'
subprocess.check_output(['mkdir', '-p', show_dst])

# Set transformer, convertor, and data_loader
transform = transforms.ToTensor()
convertor = data_convertors.ConvertImageSet(testroot, test_list_pth, data_name,
                                            transform=transform)
dataloader = DataLoader(convertor, batch_size=1, shuffle=False, num_workers=0)
device = torch.device("cuda:0")
# Make the network
cleaner = cleaner()

cleaner.load_state_dict(torch.load(Pretrained, map_location='cuda:0'))
cleaner = cleaner.to(device)
cleaner.eval()

ave_psnr = 0.0
ave_ssim = 0.0
ct_num = 0
print('Start testing '+tag+'...')
start = time.time()
val_psnr, val_ssim = validation(cleaner, dataloader, category=category, save_tag=True)

end = time.time()
elp = end-start


print('val_psnr: {0:.2f}, val_ssim: {1:.4f}'.format(val_psnr, val_ssim))
print('validation time is {0:.4f}'.format(elp))


