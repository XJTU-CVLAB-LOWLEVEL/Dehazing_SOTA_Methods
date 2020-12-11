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
import torch.nn as nn


def to_psnr(dehaze, gt):
    mse = F.mse_loss(dehaze, gt, reduction='none')
    mse_split = torch.split(mse, 1, dim=0)
    mse_list = [torch.mean(torch.squeeze(mse_split[ind])).item() for ind in range(len(mse_split))]

    intensity_max = 1.0
    psnr_list = [10.0 * log10(intensity_max / mse) for mse in mse_list]
    return psnr_list

def to_psnr_test(dehaze, gt):
    #print(dehaze.size())
    #print(gt.size())
    #m = nn.Upsample(scale_factor=4)
    #dehaze = m(dehaze)
    #print(dehaze.size())
    #import ff
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


def validation(net, val_data_loader, device, category, save_tag=False):
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

def test_net(G2, G1, G3, test_data_loader, device, save_tag=False):
    net_time = 0.
    net_count = 0.
    for batch_id, test_data in enumerate(test_data_loader):
        with torch.no_grad():
            haze, image_name = test_data
            haze = haze.to(device)
            start_time = time.time()
            dehaze_1, feat1 = G1(F.interpolate(haze, scale_factor = 0.25))
            dehaze_2, feat, feat2 = G2(dehaze_1)
            dehaze, _, _ = G3(haze, F.interpolate(dehaze_2, scale_factor = 4), feat, feat1, feat2)
            end_time = time.time() - start_time
            net_time += end_time
            net_count += 1

        # --- Save image --- #
        if save_tag:
            save_image(dehaze, image_name, 'NH')

    print('net time is {0:.4f}'.format(net_time / net_count))

def save_image(dehaze, image_name, category):
    dehaze_images = torch.split(dehaze, 1, dim=0)
    batch_num = len(dehaze_images)

    for ind in range(batch_num):
        utils.save_image(dehaze_images[ind], '{}_results/{}'.format(category, image_name[ind][:-3] + 'png'))


def print_log(epoch, train_psnr, category):
    # --- Write the training log --- #
    with open('./training_log/{}_log.txt'.format(category), 'a') as f:
        print('Date: {0}s, Time_Cost: {1:.0f}s, Epoch: [{2}/{3}], Train_PSNR: {4:.2f}, Val_PSNR: {5:.2f}, Val_SSIM: {6:.4f}'
              .format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                      0, epoch, 0, train_psnr, 0, 0), file=f)
def print_val_log(test_psnr):
    # --- Write the training log --- #
    with open('./test_log/test_log.txt', 'a') as fil:
        fil.write(str(test_psnr)+'\n')


def adjust_learning_rate(optimizer, epoch, category, lr_decay=0.5):

    # --- Decay learning rate --- #
    step = 20 if category == 'indoor' else 2000

    if not epoch % step and epoch > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
            print('Learning rate sets to {}.'.format(param_group['lr']))
    else:
        for param_group in optimizer.param_groups:
            print('Learning rate sets to {}.'.format(param_group['lr']))

def positiivate_weights(x):
    return F.relu(x) / (F.relu(x) + 1e-10)
