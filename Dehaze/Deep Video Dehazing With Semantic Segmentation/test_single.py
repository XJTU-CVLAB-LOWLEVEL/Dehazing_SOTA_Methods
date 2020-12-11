from __future__ import print_function
import argparse
import os
import time
from math import log10
from os.path import join
#from torchvision import transforms
from torchvision import utils as utils
import torch
from torch.utils.data import DataLoader
from datasets.dataset_delete import DataValSet
import statistics
import matplotlib.pyplot as plt
import re
from functools import reduce
from networks.warp_image import estimate, Network
from networks.base_networks import SpaceToDepth
from networks.warp_image import Backward
import visualization as vl
from networks.VDHnet import VDHNet
from datasets.dataset_test import Video_dataset_train, Video_dataset_test
from Resolution import Repair1,Repair2
import numpy as np
import cv2
import math
from torchvision import utils
parser = argparse.ArgumentParser(description="PyTorch LapSRN Test")
parser.add_argument("--scale", default=4, type=int, help="scale factor, Default: 4")
parser.add_argument("--aug", type=bool, default=False, help="Self-ensemble")
parser.add_argument("--gated", type=bool, default=True, help="Activated gate module")
parser.add_argument("--isTest", type=bool, default=True, help="Test or not")
parser.add_argument('--dataset', default='/home/lry/video_dehazing/datasets/VideoHazy_v2_re/Test/',type=str, help='Path of the validation dataset')
parser.add_argument('--gpu_ids', type=str, default='-1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument("--checkpoint", default='/home/lry/models/VDH11/VHD1-epoch03.pkl', type=str, help="Test on intermediate pkl (default: none)")
parser.add_argument('--name', type=str, default='VDHnet', help='filename of the training models')
parser.add_argument('--flow_dir', default="LR", type=str, help='Path of the training dataset(.h5)')
parser.add_argument("--hr_flow", type=bool, default=False, help="Activated hr_flow")
parser.add_argument("--start", type=int, default=48, help="Activated gate module")
# os.environ['CUDA_VISIBLE_DEVICES'] = '4'
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

test_set=[
    {'gated':False},
    {'gated':False},
    {'gated':True},
    {'gated':True}
]

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def is_pkl(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])

def which_trainingstep_epoch(resume):
   #trainingstep = "".join(re.findall(r"\d", resume)[-3:-2])
    start_epoch = "".join(re.findall(r"\d", resume)[-2:])
    return  int(start_epoch)

def x8_forward(img, model, gated_Tensor, test_Tensor, precision='single'):
    def _transform(v, op):
        if precision != 'single': v = v.float()

        v2np = v.data.cpu().numpy()
        if op == 'vflip':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'hflip':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 'transpose':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()

        ret = torch.Tensor(tfnp).cuda()

        if precision == 'half':
            ret = ret.half()
        elif precision == 'double':
            ret = ret.double()

        with torch.no_grad():
            ret = ret.to(device)

        return ret

    inputlist = [img]
    for tf in 'vflip', 'hflip', 'transpose':
        inputlist.extend([_transform(t, tf) for t in inputlist])

    flowlist = []
    for LR in inputlist:
        # warp the inputs
        n, c, h, w = LR.size()
        frames = c // 3
        mid_frame = frames // 2
        tensorAnchor = LR[:, mid_frame * 3:(mid_frame + 1) * 3]
        LR_Warped = []
        flowmap = []
        with torch.no_grad():
            for frame in range(frames):
                if frame == mid_frame:
                    LR_Warped.append(tensorAnchor)
                    flowmap.append(torch.cuda.FloatTensor().resize_(n, 2, h, w).zero_())
                    continue
                tensorSecond = LR[:, frame * 3:(frame + 1) * 3]
                flow, warpedTensorSecond = estimate(moduleNetwork, tensorAnchor, tensorSecond)
                LR_Warped.append(warpedTensorSecond)
                flowmap.append(flow)

            flowlist.append(torch.cat(flowmap, 1).detach())

    outputlist = [model(inputlist[i], flowlist[i], gated_Tensor, test_Tensor)[1] for i in range(8)]

    for i in range(len(outputlist)):
        if i > 3:
            outputlist[i] = _transform(outputlist[i], 'transpose')
        if i % 4 > 1:
            outputlist[i] = _transform(outputlist[i], 'hflip')
        if (i % 4) % 2 == 1:
            outputlist[i] = _transform(outputlist[i], 'vflip')

    output = reduce((lambda x, y: x + y), outputlist) / len(outputlist)

    return output








# def save_image_tensor(input_tensor: torch.Tensor, filename):
#     """
#     将tensor保存为图片
#     :param input_tensor: 要保存的tensor
#     :param filename: 保存的文件名
#
#     """
#     assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
#     input_tensor = input_tensor.clone().detach()
#     # 到cpu
#     #input_tensor = input_tensor.to(torch.device('cpu'))
#     # 反归一化
#     # input_tensor = unnormalize(input_tensor)
#     vutils.save_image(input_tensor, filename)

# def saveImg(self, img, save_dir, type):
#      #fname, fext = name.split('.')
#      #imgPath = os.path.join(save_dir, "%s_%s.%s" % (fname, type, fext))
#      imgPath = os.path.join(save_dir, "%s_%s.%s" % (fname, type))
#      utils.save_image(img, imgPath)
#      # # 改写：torchvision.utils.save_image
     # grid = utils.make_grid(img, nrow=8, padding=2, pad_value=0,
     #                                    normalize=False, range=None, scale_each=False)
     # ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
     # im = Image.fromarray(ndarr)
     # # im.show()
     # if Gray:
     #     im.convert('L').save(imgPath)  # Gray = 0.29900 * R + 0.58700 * G + 0.11400 * B
     # else:
     #     im.save(imgPath)
# def save_image(dehaze, image_name, category):
#     dehaze_images = torch.split(dehaze, 1, dim=0)
#     batch_num = len(dehaze_images)
#
#     for ind in range(batch_num):
#         utils.save_image(dehaze_images[ind], './{}_results/{}'.format(category, image_name[ind][:-3] + 'png')
def save_an_image(path, path_results, img, postfix="_REC"):

    *base, ext = os.path.basename(path).split(".")
    base = ".".join(base)
    a=ext
    base = base + postfix
    ext = "JPG"
    img_name = base + "." + ext

    dir = os.path.dirname(path)
    last_part = os.path.basename(dir)

    path = os.path.join(path_results, last_part, img_name)

    if not os.path.exists(os.path.join(path_results, last_part)):
        os.makedirs(os.path.join(path_results, last_part))

    utils.save_image(img, path, normalize=True, range=(0, 1))

def test(test_gen, model, criterion):
    avg_psnr = 0
    num = 0
    med_time = []

    with torch.no_grad():
        for iteration, batch in enumerate(test_gen, 1):
            # if iteration%50 != 0:
            # continue
            Hazy = batch[0].to(device)
            HAZY_3 = Hazy[:, 6:9]
            HAZY_3 = torch.squeeze(HAZY_3)
            HAZY1=HAZY_3
            HAZY_3 = HAZY_3.permute(1, 2, 0)#.cpu()
            HAZY_3 = HAZY_3.numpy()
            # plt.imshow(HAZY_3)
            # plt.show()


            # Flow = batch[1].to(device)
            # show_image(Hazy)
            # show_image(Flow)
            GT = batch[1].to(device)
            GT_3 = GT[:,3:6]

            GT_3 = torch.squeeze(GT_3)

            GT1 = GT_3.permute(1, 2, 0)#.cpu()
            GT1 = GT1.numpy()
            # plt.imshow(GT1)
            # plt.show()
            SEG = batch[2].to(device)
            # name = batch[3]
            gt_dir = batch[3]
            gt_dir = "".join(gt_dir)
            #save_an_image(gt_dir, save_dir, GT_3, postfix="RES")
            start_time = time.perf_counter()  # -------------------------begin to deal with an image's time
            flowmap = []
            Dehazed = model(Hazy, SEG)
            # modify
            Dehazed = torch.clamp(Dehazed, min=0, max=1)
            #torch.cuda.synchronize()  # wait for CPU & GPU time syn
            #evalation_time = time.perf_counter() - start_time  # ---------finish an image
            #med_time.append(evalation_time)
            # Dehazed_3 = Dehazed[:, 0:1]
            # Dehazed_3 = torch.squeeze(Dehazed_3)
            # Dehazed_3 = Dehazed_3#.cpu()
            # Dehazed_31 = Dehazed_3#.cpu()
            # Dehazed_31 = Dehazed_31.numpy()
            # plt.imshow(Dehazed_31)
            # plt.show()
            Dehazed_3 = Dehazed[:, 1:2]
            Dehazed_3 = torch.squeeze(Dehazed_3)
            Dehazed_31 = Dehazed_3#.cpu()
            Dehazed_31 = Dehazed_31.numpy()
            # plt.imshow(Dehazed_31)
            # plt.show()
            # Dehazed_3 = Dehazed[:, 2:3]
            # Dehazed_3 = torch.squeeze(Dehazed_3)
            # Dehazed_31 = Dehazed_3#.cpu()
            # Dehazed_31 = Dehazed_31.numpy()
            # plt.imshow(Dehazed_31)
            # plt.show()

            Dehazed_3 = Dehazed_3.numpy()
            cv2.normalize(Dehazed_3, Dehazed_3, 0, 0.5, cv2.NORM_MINMAX)
            # plt.imshow(Dehazed_3)
            # plt.show()

            Repair = Repair2(HAZY_3, Dehazed_3)
            # plt.imshow(Repair)
            # plt.show()
            Repair = torch.from_numpy(Repair)
            Repair = Repair.permute(2, 0, 1)#.cuda()
            #save_an_image(sub_gt_dir_test, save_dir, Repair, postfix="RES")
            save_an_image(gt_dir, save_dir, Repair, postfix="RES")
            dir = os.path.dirname(gt_dir)
            last_part = os.path.basename(dir)

            criterion1=torch.nn.MSELoss()
            mse1 = criterion1(HAZY1, GT_3)
            psnr1 = 10 * log10(1 / mse1)
            mse = criterion1(Repair, GT_3)
            psnr = 10 * log10(1 / mse)
            avg_psnr += psnr
            num += 1

        print("{},Avg. PSNR:{:4f} ".format(last_part,avg_psnr / num))
        # median_time = statistics.median(med_time)
        # print(median_time)
        return avg_psnr

def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))),
                normalize=False).numpy()
            img_np = np.transpose(img_np[[2, 1, 0], :, :],
                                  (1, 2, 0))  # HWC, BGR
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = np.transpose(img_np[[2, 1, 0], :, :],
                                  (1, 2, 0))  # HWC, BGR
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. '
                            f'But received with dimension: {n_dim}')
def model_test(model,):
    model = model.to(device)
    criterion = torch.nn.MSELoss(size_average=True)
    criterion = criterion.to(device)
    print(opt)
    psnr= test(testloader, model, criterion)
    print(psnr)
    return psnr

opt = parser.parse_args()
if str.lower(opt.checkpoint).find('flow') > -1:
    opt.flow = True
    print("===============> Use Flow")
else:
    opt.flow = False
#device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
str_ids = opt.gpu_ids.split(',')
torch.cuda.set_device(int(str_ids[0]))
root_val_dir = opt.dataset# #----------Validation path
save_dir = join(root_val_dir, 'Results11')  #--------------------------SR results save path
isexists = os.path.exists(save_dir)
if not isexists:
    os.makedirs(save_dir)
print("The results of testing images store in {}.".format(save_dir))
gt_dir_test=join(root_val_dir, 'gt')
seg_dir_test=join(root_val_dir, 'segment')
input_dir_test=join(root_val_dir, 'hazy')
train_sets = [x for x in sorted(os.listdir(gt_dir_test)) ]
for j in range(len(train_sets)):
        sub_input_dir_test = os.path.join(input_dir_test, train_sets[j])
        sub_gt_dir_test = os.path.join(gt_dir_test, train_sets[j])
        sub_seg_dir_test = os.path.join(seg_dir_test, train_sets[j])

#testloader = DataLoader(Video_dataset_test(input_dir_test,gt_dir_test,seg_dir_test,cropSize=256,frames=5), batch_size=1, shuffle=False, pin_memory=False)
        test_set = Video_dataset_test(sub_input_dir_test, sub_gt_dir_test,sub_seg_dir_test, cropSize=256,frames=5)
        testloader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, pin_memory=False)
        print("===> Loading model and criterion")

        if is_pkl(opt.checkpoint):
            test_pkl = opt.checkpoint
            if is_pkl(test_pkl):
                print("Testing model {}----------------------------------".format(opt.checkpoint))
                epoch = which_trainingstep_epoch(opt.checkpoint)
                #opt.gated = test_set[train_step-1]['gated']
                #model = torch.load(test_pkl, map_location=lambda storage, loc: storage)
                model = torch.load(test_pkl, map_location="cpu")
                print(get_n_params(model))
                #model = model.eval()
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
                model_test(model)
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            else:
                print("It's not a pkl file. Please give a correct pkl folder on command line for example --opt.checkpoint /models/1/GFN_epoch_25.pkl)")
# else:
#     test_list = [x for x in sorted(os.listdir(opt.checkpoint)) if is_pkl(x)]
#     print("Testing on the given 3-step trained model which stores in /models, and ends with pkl.")
#     Results = []
#     Max = {'max_psnr': 0, 'max_epoch': 0}
#     for i in range(len(test_list)):
#         print("Testing model is {}----------------------------------".format(test_list[i]))
#         train_step, epoch = which_trainingstep_epoch(join(opt.checkpoint, test_list[i]))
#         if epoch < opt.start:
#             continue
#         opt.gated = test_set[train_step-1]['gated']
#         model = torch.load(join(opt.checkpoint, test_list[i]), map_location=lambda storage, loc: storage)
#         print(get_n_params(model))
#         model = model.eval()
#         psnr = model_test(model)
#         Results.append({'epoch': "".join(re.findall(r"\d", test_list[i])[:]), 'psnr': psnr})
#         if psnr > Max['max_psnr']:
#             Max['max_psnr'] = psnr
#             Max['max_epoch'] = "".join(re.findall(r"\d", test_list[i])[:])
#
#     for Result in Results:
#         print(Result)
#     print('The Best Result of {} is : ============================>'.format(opt.name))
#     print(Max)
