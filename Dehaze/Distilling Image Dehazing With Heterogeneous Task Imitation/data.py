import torch.utils.data as data
import torchvision.transforms as transforms
import os
import glob
from PIL import Image
import numbers
import random
import torchvision.transforms.functional as F
import torch

def _get_image_size(img):
    if F._is_pil_image(img):
        return img.size
    elif isinstance(img, torch.Tensor) and img.dim() > 2:
        return img.shape[-2:][::-1]
    else:
        raise TypeError("Unexpected type {}".format(type(img)))


class RandomCrop(object):

    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode='constant'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def get_params(self, img, output_size):

        w, h = _get_image_size(img)
        th, tw = output_size
        if w == tw and h == th:
            return 0, 0, h, w

        i = random.randint(0, h - th)
        j = random.randint(0, w - tw)
        return i, j, th, tw

    def __call__(self, img, img2):

        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        # pad the width if needed
        if self.pad_if_needed and img.size[0] < self.size[1]:
            img = F.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and img.size[1] < self.size[0]:
            img = F.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w), F.crop(img2, i, j, h, w)

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}, padding={1})'.format(self.size, self.padding)


class DataGeneratorPaired(data.Dataset):

    def __init__(self, splits, mode="train"):


        if mode == "train":
            self.gt_paths = splits["gt_paths"]
            self.hazy_paths = splits["hazy_paths"]
            self.crop = True
        elif mode == "val":
            self.gt_paths = splits["gt_paths"]
            self.hazy_paths = splits["hazy_paths"]
            self.crop = False
        elif mode == "test":
            self.gt_paths = splits["gt_paths"]
            self.hazy_paths = splits["hazy_paths"]
            self.crop = False
        else:
            raise "Incorrect dataset mode"

        self.RandomCrop = RandomCrop(255)
        self.transform_CenterCrop = transforms.CenterCrop(255)

        self.transform_hazy = self.get_transforms()
        self.transform_gt = self.get_transforms()



    def get_transforms(self):


        transforms_list = []

        # transforms_list.append(transforms.CenterCrop(255))
        transforms_list.append(transforms.ToTensor())
        transforms_list.append(transforms.Normalize( (0.5, 0.5, 0.5), (0.5, 0.5, 0.5) ))
        return transforms.Compose(transforms_list)


    def __getitem__(self, index):

        gt = Image.open(self.gt_paths[index])
        hazy = Image.open(self.hazy_paths[index])

        if self.crop:
            gt, hazy = self.RandomCrop(gt, hazy)
            #gt = self.transform_CenterCrop(gt)
            #hazy = self.transform_CenterCrop(hazy)

        if self.transform_hazy is not None:
            hazy = self.transform_hazy(hazy)

        if self.transform_gt is not None:
            gt = self.transform_gt(gt)

        item = {"hazy":hazy,
                "gt":gt,
                "gt_paths":self.gt_paths[index],
                "hazy_paths":self.hazy_paths[index]}

        return item

    def __len__(self):

        return len(self.gt_paths)