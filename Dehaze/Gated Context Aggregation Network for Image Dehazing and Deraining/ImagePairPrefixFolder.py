import os
import bisect
import threading
import torch
import numpy as np
import numpy.random as random
from PIL import Image
from torch.utils.data import Dataset
from folder_loader import FolderLoader
import torchvision.transforms as transforms
from utils import batch_edge_compute

def pil_loader(img_path):
    return Image.open(img_path).convert("RGB")


class ImagePairPrefixFolder(Dataset):
    def __init__(self, input_folder, gt_folder, max_img_size=0, size_unit=1, force_rgb=False):
        super(ImagePairPrefixFolder, self).__init__()

        self.gt_loader = FolderLoader(gt_folder)
        # build the map from image name to index
        self.gt_map = dict()
        for idx, img_name in enumerate(self.gt_loader.img_names):
            self.gt_map[os.path.splitext(img_name)[0].split('_')[0]] = idx

        self.input_loader = FolderLoader(input_folder)
        assert all([os.path.splitext(x)[0].split('_')[0] in self.gt_map for x in self.input_loader.img_names]), \
                'cannot find corresponding gt names'


        self.input_folder = input_folder
        self.gt_folder = gt_folder
        self.max_img_size = max_img_size
        self.size_unit = size_unit
        self.force_rgb = force_rgb

    def __getitem__(self, index):
        input_name, input_img = self.input_loader[index]
        input_basename = os.path.splitext(input_name)[0].split('_')[0]
        gt_idx = self.gt_map[input_basename]

        gt_name, gt_img = self.gt_loader[gt_idx]
        if self.force_rgb:
            input_img = input_img.convert('RGB')
            gt_img = gt_img.convert('RGB')
        im_w, im_h = input_img.size
        gt_w, gt_h = gt_img.size
        assert im_w==gt_w and im_h==gt_h, 'input image and gt image size not match'

        im_w, im_h = input_img.size
        if 0 < self.max_img_size < max(im_w, im_h):
            if im_w < im_h:
                out_h = int(self.max_img_size) // self.size_unit * self.size_unit
                out_w = int(im_w / im_h * out_h) // self.size_unit * self.size_unit
            else:
                out_w = int(self.max_img_size) // self.size_unit * self.size_unit
                out_h = int(im_h / im_w * out_w) // self.size_unit * self.size_unit
        else:
            out_w = im_w // self.size_unit * self.size_unit
            out_h = im_h // self.size_unit * self.size_unit

        if im_w != out_w or im_h != out_h:
            input_img = input_img.resize((out_w, out_h), Image.BILINEAR)
            gt_img = gt_img.resize((out_w, out_h), Image.BILINEAR)

        im_w, im_h = input_img.size

        input_img = np.array(input_img).astype('float')
        gt_img = np.array(gt_img).astype('float')
        if len(input_img.shape) == 2:
            input_img = input_img[:, :, np.newaxis]
        if len(gt_img.shape) == 2:
            gt_img = gt_img[:, :, np.newaxis]
        return {'input_img': input_img, 'gt_img': gt_img,  'input_h': im_h, "input_w": im_w}

    def get_input_info(self, index):
        image_name = os.path.splitext(self.input_loader.img_names[index])[0]
        return self.input_loader, image_name

    def __len__(self):
        return len(self.input_loader)


def var_custom_collate(batch):
    min_h, min_w = 10000, 10000
    for item in batch:
        min_h = min(min_h, item['input_h'])
        min_w = min(min_w, item['input_w'])
    inc = 1 if len(batch[0]['input_img'].shape)==2 else batch[0]['input_img'].shape[2]
    batch_input_images = torch.Tensor(len(batch), inc, min_h, min_w)
    batch_gt_images = torch.Tensor(len(batch), inc, min_h, min_w)

    for idx, item in enumerate(batch):
        off_y = 0 if item['input_h']==min_h else random.randint(0, item['input_h'] - min_h)
        off_x = 0 if item['input_w']==min_w else random.randint(0, item['input_w'] - min_w)
        crop_input_img = item['input_img'][off_y:off_y + min_h, off_x:off_x + min_w, :]
        crop_gt_img = item['gt_img'][off_y:off_y + min_h, off_x:off_x + min_w, :]
        batch_input_images[idx] = torch.from_numpy(crop_input_img.transpose((2, 0, 1))) - 128
        batch_gt_images[idx] = torch.from_numpy(crop_gt_img.transpose((2, 0, 1)))


    batch_input_edges = batch_edge_compute(batch_input_images) - 128
    return batch_input_images, batch_input_edges,  batch_gt_images
