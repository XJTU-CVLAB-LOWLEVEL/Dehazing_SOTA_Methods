import os
import numpy as np
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from test_nitre_data import TestData as data
from test_nitre_data import ValData as data1
from model import Generate_quarter, Generate_quarter_refine, Generate, Discriminator, LossD, LossFeat
from utils import to_psnr_test, print_val_log, validation, adjust_learning_rate, to_psnr
from torchvision.models import vgg16
from perceptual import LossNetwork
from torch.utils.data import ConcatDataset
import torchvision.utils as utils

def main( val_data,test_phrase, test_epoch):#test_img, test_gt,
    device_ids = [1]
    device = torch.device("cuda:1")
    # test_data_dir = test_img
    # test_data_gt = test_gt
    val_data_dir = val_data
    test_batch_size = 1
    network_height = 3
    network_width = 6
    num_dense_layer = 4
    growth_rate = 16
    test_phrase = test_phrase
    crop_size = [1600, 1200]
    test_data = data1(val_data_dir)
    # test_data = data(test_data_dir, test_data_gt)
    test_data_loader = DataLoader(test_data, batch_size=test_batch_size)

    def save_image(dehaze, image_name, category):
        #dehaze_images = torch.split(dehaze, 1, dim=0)
        batch_num = len(dehaze)
    
        for ind in range(batch_num):
            utils.save_image(dehaze[ind], '{}_results/{}'.format(category, image_name[ind][:-3] + 'JPG'))

    if test_phrase == 1:
        G1 = Generate_quarter(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)
        G1 = G1.to(device)
        G1 = nn.DataParallel(G1, device_ids=device_ids)
        G1.load_state_dict(torch.load('./checkpoint/1_'+str(360)+'.tar'))
        G1.eval()
        psnr=[]
        net_time = 0.
        net_count = 0.
        for batch_id, test_data in enumerate(test_data_loader):
            with torch.no_grad():
                haze, gt, image_name = test_data
                #haze = F.interpolate(haze, scale_factor = 0.25)
                haze = haze.to(device)
                gt = gt.to(device)
                start_time = time.time()
                dehaze, _ = G1(haze)
                end_time = time.time() - start_time
                net_time += end_time
                net_count += 1
                test_info = to_psnr_test(dehaze, gt)
                psnr.append(sum(test_info) / len(test_info))

                print (sum(test_info) / len(test_info))
                # print_val_log(sum(test_info) / len(test_info))
        # --- Save image --- #
            save_image(dehaze, image_name, 'NH')
        test_psnr = sum(psnr) / len(psnr)

        print ('Test PSNR:' + str(test_psnr))
        print('net time is {0:.4f}'.format(net_time / net_count))

    if test_phrase == 2:
        G1 = Generate_quarter(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)
        G1 = G1.to(device)
        G1 = nn.DataParallel(G1, device_ids=device_ids)
        #G1.load_state_dict(torch.load('./checkpoint/1.tar'))
        G2 = Generate_quarter_refine(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)
        G2 = G2.to(device)
        G2 = nn.DataParallel(G2, device_ids=device_ids)
        G1.load_state_dict(torch.load('./checkpoint/2-'+str(test_epoch)+'_G1.tar'))

        G2.load_state_dict(torch.load('./checkpoint/2_' + str(test_epoch) +'_G2.tar'))
        G1.eval()
        G2.eval()
        psnr=[]
        net_time = 0.
        net_count = 0.
        for batch_id, test_data in enumerate(test_data_loader):
            with torch.no_grad():
                haze, gt, image_name = test_data
                #haze = F.interpolate(haze, scale_factor = 0.25,recompute_scale_factor=True)
                haze = haze.to(device)
                gt =gt.to(device)
                start_time = time.time()
                dehaze_1, feat1 = G1(haze)
                dehaze, _, _ = G2(dehaze_1)
                gt = gt
                end_time = time.time() - start_time
                net_time += end_time
                net_count += 1
                test_info = to_psnr_test(dehaze, gt)
                psnr.append(sum(test_info) / len(test_info))
                print (sum(test_info) / len(test_info))
            # --- Save image --- #
            save_image(dehaze, image_name, 'NH')
        test_psnr = sum(psnr) / len(psnr)
        print ('Test PSNR:' + str(test_psnr))
        print('net time is {0:.4f}'.format(net_time / net_count))

    if test_phrase == 3:
        G1 = Generate_quarter(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)
        G1 = G1.to(device)
        G1 = nn.DataParallel(G1, device_ids=device_ids)
        G1.load_state_dict(torch.load('./checkpoint/3-'+str(test_epoch)+'_G1.tar'))
        G2 = Generate_quarter_refine(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)
        G2 = G2.to(device)
        G2 = nn.DataParallel(G2, device_ids=device_ids)
        G2.load_state_dict(torch.load('./checkpoint/3_'+str(test_epoch)+'_G2.tar'))
        G3 = Generate(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)
        G3 = G3.to(device)
        G3 = nn.DataParallel(G3, device_ids=device_ids)
        G3.load_state_dict(torch.load('./checkpoint/33_'+str(test_epoch)+'_G3.tar'))
        G1.eval()
        G2.eval()
        G3.eval()
        psnr=[]
        net_time = 0.
        net_count = 0.
        for batch_id, test_data in enumerate(test_data_loader):
            with torch.no_grad():
                haze, gt, image_name = test_data
                haze = haze.to(device)
                gt = gt.to(device)
                start_time = time.time()
                dehaze_1, feat1 = G1(F.interpolate(haze, scale_factor = 0.25,recompute_scale_factor=True))
                dehaze_2, feat, feat2 = G2(dehaze_1)
                dehaze= G3(haze, F.interpolate(dehaze_2, scale_factor = 4,recompute_scale_factor=True), feat)
                end_time = time.time() - start_time
                net_time += end_time
                net_count += 1
                test_info = to_psnr(dehaze, gt)
                psnr.append(sum(test_info) / len(test_info))
                print (sum(test_info) / len(test_info))
            # --- Save image --- #
            save_image(dehaze, image_name, 'NH')
        test_psnr = sum(psnr) / len(psnr)
        print ('Test PSNR:' + str(test_psnr))
        print('net time is {0:.4f}'.format(net_time / net_count))
    return test_psnr
val_data = './data/VideoHazy_v3/resize_test/'
main(val_data,1,360)





