"""
paper: GridDehazeNet: Attention-Based Multi-Scale Network for Image Dehazing
file: test_data.py
about: build the validation/test dataset
author: Xiaohong Liu
date: 01/08/19
"""

# --- Imports --- #
import torch.utils.data as data
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, CenterCrop
from glob import glob
import os


# --- Validation/test dataset --- #
class TestData(data.Dataset):
    def __init__(self, test_data_dir, test_data_gt):
        super().__init__()
        '''
        test_list = test_data_dir + 'test_list.txt'
        with open(test_list) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
        '''
        #pattern = re.compile(r'') 
        fpaths = glob(os.path.join(test_data_dir, '*.JPG'))
        gtpaths = glob(os.path.join(test_data_gt, '*.JPG'))
        haze_names = []
        gt_names = []
        for path in fpaths:
            haze_names.append(path.split('/')[-1])
        for path in gtpaths:
            gt_names.append(path.split('/')[-1])
        self.haze_names = haze_names
        self.gt_names = gt_names
        self.test_data_dir = test_data_dir
        self.test_data_gt = test_data_gt

    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]
        haze_img = Image.open(os.path.join(self.test_data_dir, haze_name))
        gt_img = Image.open(os.path.join(self.test_data_gt, gt_name))

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        #haze = transform_haze(haze_img)

        transform_gt = Compose([ToTensor()])
        gt = transform_gt(gt_img)
        haze = transform_haze(haze_img)

        return haze, gt, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)
# --- Validation/test dataset --- #
class ValData(data.Dataset):
    def __init__(self, val_data_dir):
        super().__init__()
        val_list = val_data_dir + 'test.txt'
        with open(val_list) as f:
            contents = f.readlines()
            haze_names = [i.strip() for i in contents]
            gt_names = [i for i in haze_names]

        self.haze_names = haze_names
        self.gt_names = gt_names
        self.val_data_dir = val_data_dir

    def get_images(self, index):
        haze_name = self.haze_names[index]
        gt_name = self.gt_names[index]
        haze_img = Image.open(self.val_data_dir + 'hazy/' + haze_name)
        gt_img = Image.open(self.val_data_dir + 'gt/' + gt_name)

        # --- Transform to tensor --- #
        transform_haze = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        transform_gt = Compose([ToTensor()])
        haze = transform_haze(haze_img)
        gt = transform_gt(gt_img)

        return haze, gt, haze_name

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        return len(self.haze_names)
