import os 
import torch
import torch
import numpy as np
from PIL import Image
import os

from scipy import signal
from torchvision.utils import make_grid

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def edge_compute(x):
    x_diffx = torch.abs(x[:,:,1:] - x[:,:,:-1])
    x_diffy = torch.abs(x[:,1:,:] - x[:,:-1,:])

    y = x.new(x.size())
    y.fill_(0)
    y[:,:,1:] += x_diffx
    y[:,:,:-1] += x_diffx
    y[:,1:,:] += x_diffy
    y[:,:-1,:] += x_diffy
    y = torch.sum(y,0,keepdim=True)/3
    y /= 4
    return y

def batch_edge_compute(x):
    x_diffx = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1])
    x_diffy = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:])

    y = x.new(x.size())
    y.fill_(0)
    y[:,:,:,1:] += x_diffx
    y[:,:,:,:-1] += x_diffx
    y[:,:,1:,:] += x_diffy
    y[:,:,:-1,:] += x_diffy
    y = torch.sum(y,1,keepdim=True)/3
    y /= 4
    return y

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
def tensor2im(input_image, imtype=np.uint8):
    if isinstance(input_image, torch.Tensor):
        image_tensor = input_image.data
    else:
        return input_image
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = image_numpy.clip(0, 255)
    return image_numpy.astype(imtype)


def tensor2imgrid(input_image):
    im_grid = make_grid(input_image[:4, ...], nrow=2, normalize=True, range=(-128, 128))
    return im_grid
    # ndarr = im_grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    # im = Image.fromarray(ndarr)
    # return im


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()


def filter2(x, kernel, mode='same'):
    return signal.convolve2d(x, np.rot90(kernel, 2), mode=mode)


def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1
    and img2 (images are assumed to be uint8)

    This function attempts to mimic precisely the functionality of ssim.m a
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255  # bitdepth of image
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    mu1 = filter2(img1, window, mode='valid')
    mu2 = filter2(img2, window, mode='valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(img1 * img1, window, mode='valid') - mu1_sq
    sigma2_sq = filter2(img2 * img2, window, mode='valid') - mu2_sq
    sigma12 = filter2(img1 * img2, window, mode='valid') - mu1_mu2
    if cs_map:
        return np.mean(np.mean((((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                             (sigma1_sq + sigma2_sq + C2)),
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))))
    else:
        return np.mean(np.mean(((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))))



class MovingAvg(object):
    def __init__(self, pool_size=100):
        from queue import Queue
        self.pool = Queue(maxsize=pool_size)
        self.sum = 0
        self.curr_pool_size = 0
        self.pool_size = pool_size

    def set_curr_val(self, val):
        if not self.pool.full():
            self.curr_pool_size += 1
            self.pool.put_nowait(val)
        else:
            last_first_val = self.pool.get_nowait()
            self.pool.put_nowait(val)
            self.sum -= last_first_val

        self.sum += val
        return self.sum / self.curr_pool_size

    def reset(self):
        from queue import Queue
        self.pool = Queue(maxsize=self.pool_size)
        self.sum = 0
        self.curr_pool_size = 0