"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: train_data.py
about: build the training dataset
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from random import randrange
from torchvision.transforms import Compose, ToTensor, Normalize
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import random
import os
from glob import glob
import re
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')



# --- Training dataset --- #
class TrainData(data.Dataset):
    def __init__(self, crop_size, train_data_dir, train_data_gt):
        super().__init__()
        #pattern = re.compile(r'\d+(?<=hazy)')
        fpaths = glob(os.path.join(train_data_dir, '*.JPG'))
        gtpaths = glob(os.path.join(train_data_gt, '*.JPG'))

        haze_names = []
        gt_names = []
        for path in fpaths:
            haze_names.append(path.split('/')[-1])
            # gt = path.split('/')[-1].split('_')[0].split('.')[0]
            # if '2019' in train_data_gt:
            #     gt = gt + '_GT'
            # gt_names.append(str(gt)+'.png')
            # gt_names.append(())
        for path in gtpaths:
            gt_names.append(path.split('/')[-1])

        self.haze_names = haze_names
        self.gt_names = gt_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir
        self.train_data_gt = train_data_gt
        self.haze_cache = {}
        self.gt_cache = {}

        for haze_name in haze_names:
            if haze_name in self.haze_cache:
                continue
            haze_img = Image.open(self.train_data_dir + '/' +haze_name )#.convert('RGB')
            self.haze_cache[haze_name] = haze_img

        for gt_name in gt_names:
            if gt_name in self.gt_cache:
                continue
            gt_img = Image.open(self.train_data_gt + gt_name)#.convert('RGB')
            self.gt_cache[gt_name] = gt_img

        print ('use cache')

    def generate_scale_label(self, haze, gt):
        f_scale = 0.8 + random.randint(0, 7) / 10.0
        width, height = haze.size
        haze = haze.resize((int(width * f_scale), (int(height * f_scale))), resample = (Image.BICUBIC))
        gt = gt.resize((int(width * f_scale), (int(height * f_scale))), resample = (Image.BICUBIC))
        return haze, gt

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]

        haze_img = self.haze_cache[haze_name]
        gt_img = self.gt_cache[gt_name]

        haze_img, gt_img = self.generate_scale_label(haze_img, gt_img)
        
        width, height = haze_img.size

        if width < crop_width or height < crop_height:
            raise Exception('Bad image size: {}'.format(gt_name))

        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        rand_hor=random.randint(0,1)
        rand_rot=random.randint(0,3)
        haze_crop_img=tfs.RandomHorizontalFlip(rand_hor)(haze_crop_img)
        gt_crop_img=tfs.RandomHorizontalFlip(rand_hor)(gt_crop_img)
        if rand_rot:
          haze_crop_img=FF.rotate(haze_crop_img,90*rand_rot)
          gt_crop_img=FF.rotate(gt_crop_img,90*rand_rot)

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_crop_img)
        gt = transform_gt(gt_crop_img)
        haze_gt = transform_gt(gt_crop_img)

        # --- Check the channel is 3 or not --- #
        if list(haze.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        return haze, gt, haze_gt

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        print(len(self.haze_names))
        return len(self.haze_names)

# --- Training dataset --- #
class TrainData_gdn(data.Dataset):
    def __init__(self, crop_size, train_data_dir):
        super().__init__()
        train_list = train_data_dir + 'train_.txt'
        with open(train_list) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = [i for i in haze_names]

        self.haze_names = haze_names
        self.gt_names = gt_names
        self.crop_size = crop_size
        self.train_data_dir = train_data_dir

    def generate_scale_label(self, haze, gt):
        f_scale = 0.8 + random.randint(0, 7) / 10.0
        width, height = haze.size
        haze = haze.resize((int(width * f_scale), (int(height * f_scale))), resample = (Image.BICUBIC))
        gt = gt.resize((int(width * f_scale), (int(height * f_scale))), resample = (Image.BICUBIC))
        return haze, gt

    def get_images(self, index):
        crop_width, crop_height = self.crop_size
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]

        haze_img = Image.open(self.train_data_dir + 'hazy/' + haze_name)

        try:
            gt_img = Image.open(self.train_data_dir + 'gt/' + gt_name)
        except:
            gt_img = Image.open(self.train_data_dir + 'gt/' + gt_name).convert('RGB')

        width, height = haze_img.size

        if width < crop_width or height < crop_height:
            raise Exception('Bad image size: {}'.format(gt_name))

        # --- x,y coordinate of left-top corner --- #
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        haze_crop_img = haze_img.crop((x, y, x + crop_width, y + crop_height))
        gt_crop_img = gt_img.crop((x, y, x + crop_width, y + crop_height))

        rand_hor=random.randint(0,1)
        rand_rot=random.randint(0,3)
        haze_crop_img=tfs.RandomHorizontalFlip(rand_hor)(haze_crop_img)
        gt_crop_img=tfs.RandomHorizontalFlip(rand_hor)(gt_crop_img)
        if rand_rot:
          haze_crop_img=FF.rotate(haze_crop_img,90*rand_rot)
          gt_crop_img=FF.rotate(gt_crop_img,90*rand_rot)

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_crop_img)
        gt = transform_gt(gt_crop_img)
        haze_gt = transform_gt(gt_crop_img)

        # --- Check the channel is 3 or not --- #
        if list(haze.shape)[0] is not 3 or list(gt.shape)[0] is not 3:
            raise Exception('Bad image channel: {}'.format(gt_name))

        return haze, gt, haze_gt

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)




