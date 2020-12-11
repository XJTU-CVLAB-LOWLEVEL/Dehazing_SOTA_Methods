import os, sys
import time
from math import log10

from skimage import measure

sys.path.insert(1, '../')
import torch
import cv2
import shutil
import torchvision
import numpy as np
import itertools
import subprocess
import random
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
from pietorch import data_convertors
from pietorch.DuRN_US import cleaner as cleaner
from pietorch.pytorch_ssim import ssim as ssim
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


def validation(net, val_data_loader):
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
    avr_psnr = sum(psnr_list) / len(psnr_list)
    avr_ssim = sum(ssim_list) / len(ssim_list)
    return avr_psnr, avr_ssim
#------ Options -------
tag        = 'DuRN_US'
data_name  = 'RESIDE'
bch_size   = 8
base_lr    = 0.0001
epoch_size = 3000
gpus       = 1
crop_size  = 256

ssim_weight    = 1.1
l1_loss_weight = 0.75
with_data_aug  = False
#----------------------

# Set pathes
data_root  = '../data/' +data_name+'/indoor_train/'
imlist_pth = '../lists/'+data_name+'_indoor/train_list.txt'
test_root = '../data/' + data_name + '/sots_indoor_test/'
testlist_pth = '../lists/'+data_name+'_indoor/sots_test_list.txt'
# dstroot for saving models. 
# logroot for writting some log(s), if is needed.
dstroot = './trainedmodels/'+data_name+'/'+tag+'/'
logroot = './logs/'+data_name+'/'+tag+'/'
subprocess.check_output(['mkdir', '-p', dstroot])
subprocess.check_output(['mkdir', '-p', logroot])
device = torch.device('cuda:0')
# Transform
transform = transforms.ToTensor()
# Dataloader
convertor  = data_convertors.ConvertImageSet(data_root, imlist_pth, data_name,
                                             transform=transform, is_train=True,
                                             with_aug=with_data_aug, crop_size=crop_size)
dataloader = DataLoader(convertor, batch_size=bch_size, shuffle=False)
test_data = data_convertors.ConvertImageSet(test_root, testlist_pth, data_name, transform=transform, is_train=False, with_aug=with_data_aug, crop_size=crop_size)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
Pretrained = 'trainedmodels/RESIDE/DuRN_US1/epoch_210_model.pt'
# Make network
cleaner = cleaner()
cleaner.load_state_dict(torch.load(Pretrained))
cleaner = cleaner.to(device)
cleaner.train()

# Optimizer and Loss
optimizer = optim.Adam(cleaner.parameters(), lr=base_lr)
L1_loss = nn.L1Loss()

# Start training
print('Start training...')
for epoch in range(epoch_size):
    psnr_list = []
    start = time.time()

    for iteration, data in enumerate(dataloader):
        img, label, _ = data
        img = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        # Cleaning noisy images
        cleaned = cleaner(img)

        # Compute ssim loss (not used)
        ssim_loss = -ssim(cleaned, label)
        ssim_loss = ssim_loss*ssim_weight

        # Compute L1 loss (not used)
        l1_loss   = L1_loss(cleaned, label)
        l1_loss   = l1_loss*l1_loss_weight

        loss = ssim_loss + l1_loss
        # Backward and update params        

        loss.backward()
        optimizer.step()
        psnr_list.extend(to_psnr(cleaned, label))

    epoch_psnr = sum(psnr_list) / len(psnr_list)
    end = time.time()
    elp = end - start
    print('epoch:{}, psnr:{}, time:{}secondes.'.format(epoch+1, epoch_psnr, elp))
    a = open('log/log1.txt', 'a')
    print('epoch:{}, psnr:{}, time:{}secondes.'.format(epoch + 1, epoch_psnr, elp), file=a)
    a.close()
    if epoch%10 == 9:
        torch.save(cleaner.state_dict(),        dstroot+'epoch_'+str(epoch+1)+'_model.pt')
        val_psnr, val_ssim = validation(cleaner, test_loader)
        f = open('log/log1_1.txt', 'a')
        print("epoch:{}, psnr:{}, val_psnr:{}, val_ssim:{}".format(epoch+1, epoch_psnr, val_psnr, val_ssim), file=f)
        f.close()
        print("epoch:{}, psnr:{}, val_psnr:{}, val_ssim:{}".format(epoch+1, epoch_psnr, val_psnr, val_ssim))

    if epoch in [100, 180]:
        for param_group in optimizer.param_groups:
            param_group['lr']*= 0.1    

