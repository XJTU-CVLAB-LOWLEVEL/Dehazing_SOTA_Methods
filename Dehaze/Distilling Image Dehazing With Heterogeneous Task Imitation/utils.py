import numpy as np
import os 
import glob
import configparser as cp
import socket
import math
import torch
import shutil
import random
from torchvision.utils import save_image

# https://towardsdatascience.com/adaptive-and-cyclical-learning-rates-using-pytorch-2bf904d18dee
def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3):

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda

def read_config():
    config = cp.ConfigParser()
    cur_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    config.read(os.path.join(cur_path, 'config.ini'))
    host = socket.gethostname()
    return config[host]

def restricted_float(x, inter):
    x = float(x)
    if x < inter[0] or x > inter[1]:
        raise argparse.ArgumentTypeError("%r not in range [1e-5, 1e-4]" % (x,))
    return x

def save_checkpoint(state, directory):

    if not os.path.isdir(directory):
        os.makedirs(directory)
    checkpoint_file = os.path.join(directory, 'checkpoint.pth')
    best_model_file = os.path.join(directory, 'model_best.pth')
    torch.save(state, checkpoint_file)
    shutil.copyfile(checkpoint_file, best_model_file)

def load_files_and_partition(root_path, hazy_dir_name="hazy", gt_dir_name="gt"):

    np.random.seed(0)
    
    gt_paths, hazy_paths = load_pairs(root_path, hazy_dir_name=hazy_dir_name, gt_dir_name=gt_dir_name)

    hazy_paths = pair_selection(gt_paths, hazy_paths)

    if len(gt_paths) != len(hazy_paths):
        raise "Inconsistent dataset length"

    indexes = list(range(len(gt_paths)))

    te_indexes = np.random.choice(indexes, int(len(indexes)), replace=False).astype(int)


    splits = dict()


    splits["gt_paths"] = gt_paths[te_indexes]

    splits["hazy_paths"] = hazy_paths[te_indexes]

    return splits


def get_last_part_of_path(paths):

    parted_paths = []

    for i in range(len(paths)):

        last_part = os.path.basename(os.path.normpath(paths[i])).split(".")[0].split("_")[0]

        parted_paths.append(last_part)

    return parted_paths



def pair_selection(gt_paths, hazy_paths):

    m = len(gt_paths)

    last_part_hazy_paths = get_last_part_of_path(hazy_paths)

    selected_hazy_paths = []

    for i in range(m):


        # get last part of the path
        name = os.path.basename(os.path.normpath(gt_paths[i])).split(".")[0].split("_")[0]
        
        # find matching names
        indexes = []
        x = 0
        for y in last_part_hazy_paths:
            if y == name:
                indexes.append(x)
            x += 1


        # Select 1 pair from N matching examples
        selected_path = np.random.choice(hazy_paths[indexes], 1, replace=False)

        selected_hazy_paths.append(selected_path[0])


    return np.array(selected_hazy_paths)

def load_pairs(root_path, hazy_dir_name="hazy", gt_dir_name="GT"):

    return (load_gt_images(root_path, gt_dir_name), load_hazy_images(root_path, hazy_dir_name))

def load_gt_images(root_path, gt_dir_name):

    return np.array([name for name in glob.glob(os.path.join(root_path, gt_dir_name, "*"))])

    
def load_hazy_images(root_path, hazy_dir_name):
    return np.array([name for name in glob.glob(os.path.join(root_path, hazy_dir_name, "*"))])


def save_an_image(path, path_results, img, postfix="_REC"):

    *base, ext = os.path.basename(path).split(".")
    base = ".".join(base)

    base = base + postfix

    ext = "png"

    img_name = base + "." + ext
    path = os.path.join(path_results, img_name)

    save_image(img, path, normalize=True, range=(-1, 1))

def gaussian_filter(size: int, sigma: float) -> torch.Tensor:
    r"""Returns 2D Gaussian kernel N(0,`sigma`^2)
    Args:
        size: Size of the lernel
        sigma: Std of the distribution
    Returns:
        gaussian_kernel: 2D kernel with shape (1 x kernel_size x kernel_size)
    References:
        https://github.com/photosynthesis-team/piq/blob/master/piq/functional/filters.py
    """
    coords = torch.arange(size).to(dtype=torch.float32)
    coords -= (size - 1) / 2.

    g = coords ** 2
    g = (- (g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma ** 2)).exp()

    g /= g.sum()
    return g.unsqueeze(0)