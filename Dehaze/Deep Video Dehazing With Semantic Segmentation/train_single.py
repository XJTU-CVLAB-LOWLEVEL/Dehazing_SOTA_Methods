# Citation:
#     Gated Fusion Network for Joint Image Deblurring and Super-Resolution
#     The British Machine Vision Conference(BMVC2018 oral)
#     Xinyi Zhang, Hang Dong, Zhe Hu, Wei-Sheng Lai, Fei Wang and Ming-Hsuan Yang
# Contact:
#     cvxinyizhang@gmail.com
# Project Website:
#     http://xinyizhang.tech/bmvc2018
#     https://github.com/jacquelinelala/GFN

from __future__ import print_function
import torch.optim as optim
import argparse
import os
from os.path import join
import torch
from torch.utils.data import DataLoader
from datasets.dataset_delete import DataSet
from importlib import import_module
import random
import re
import visualization as vl

# Training settings
parser = argparse.ArgumentParser(description="PyTorch Train")
parser.add_argument("--batchSize", type=int, default=16, help="Training batch size")
parser.add_argument("--cropSize", type=int, default=32, help="LR patch size")
parser.add_argument("--frames", type=int, default=5, help="the amount of input frames")
parser.add_argument("--start_training_step", type=int, default=1, help="Training step")
parser.add_argument("--nEpochs", type=int, default=60, help="Number of epochs to train")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate, default=1e-4")
parser.add_argument("--step", type=int, default=7, help="Change the learning rate for every 30 epochs")
parser.add_argument("--forget_epoch", type=bool, default=False, help="preload model but train from epoch 1")
parser.add_argument("--start-epoch", type=int, default=1, help="Start epoch from 1")
parser.add_argument("--lr_decay", type=float, default=0.5, help="Decay scale of learning rate, default=0.5")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--scale", default=1, type=int, help="Scale factor, Default: 1")
parser.add_argument("--lambda_db", type=float, default=0.5, help="Weight of deblurring loss, default=0.5")
parser.add_argument("--lambda_GL", type=float, default=0.5, help="Weight of deblurring loss, default=0.5")
parser.add_argument("--gated", type=bool, default=False, help="Activated gate module")
parser.add_argument("--isTest", type=bool, default=False, help="Test or not")
parser.add_argument('--dataset', default="/data/Dataset/VSR/NTIRE-VDBSR/", type=str, help='Path of the training dataset(.h5)')
parser.add_argument('--model', default='RDN', type=str, help='Import which network')
parser.add_argument('--name', default='RDN', type=str, help='Filename of the training models')
parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='B',
                    help='parameters config of RDN. (Use in RDN)')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
# Option for RCAN

parser.add_argument('--act', type=str, default='relu',
                    help='activation function')
parser.add_argument('--pre_train', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--extend', type=str, default='.',
                    help='pre-trained model directory')
parser.add_argument('--n_resblocks', type=int, default=20,
                    help='number of residual blocks')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--res_scale', type=float, default=1,
                    help='residual scaling')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')
parser.add_argument('--n_resgroups', type=int, default=10,
                    help='number of residual groups')
parser.add_argument('--reduction', type=int, default=16,
                    help='number of feature maps reduction')

training_settings=[
    #{'nEpochs': 80, 'lr': 1e-4, 'step':  50, 'lr_decay': 0.1, 'lambda_db': 0.5, 'gated': False}
    {'nEpochs': 80, 'lr': 1e-4, 'step':  50, 'lr_decay': 0.1, 'lambda_db': 0.5, 'gated': False} #V-DBPN
]

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def mkdir_steptraing():
    root_folder = os.path.abspath('.')
    models_folder = join(root_folder, 'models')
    models_folder = join(models_folder, opt.name)
    step1_folder, step2_folder, step3_folder, step4_folder = join(models_folder,'1'), join(models_folder,'2'), join(models_folder, '3'), join(models_folder, '4')
    isexists = os.path.exists(step1_folder) and os.path.exists(step2_folder) and os.path.exists(step3_folder) and os.path.exists(step4_folder)
    if not isexists:
        os.makedirs(step1_folder)
        os.makedirs(step2_folder)
        os.makedirs(step3_folder)
        os.makedirs(step4_folder)
        print("===> Step training models store in models/1 & /2 & /3.")

def is_hdf5_file(filename):
    return any(filename.endswith(extension) for extension in [".h5"])

def which_trainingstep_epoch(resume):
    trainingstep = "".join(re.findall(r"\d", resume)[-3:-2])
    start_epoch = "".join(re.findall(r"\d", resume)[-2:])
    return int(trainingstep), int(start_epoch)+1

def adjust_learning_rate(epoch):
        lr = opt.lr * (opt.lr_decay ** (epoch // opt.step))
        print(lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def checkpoint(step, epoch):
    root_folder = os.path.abspath('.')
    models_folder = join(root_folder, 'models')
    models_folder = join(models_folder, opt.name)
    model_out_path = join(models_folder, "{0}/GFN_epoch_{1:02d}.pkl".format(step, epoch))
    torch.save(model, model_out_path)
    print("===>Checkpoint saved to {}".format(model_out_path))

def gradient_loss(pred, GT, kernel_size=15):
    pad_size = (kernel_size - 1) // 2
    pred = pred.sum(1)
    GT = GT.sum(1)
    BN = pred.size()[0]
    M = pred.size()[1]
    N = pred.size()[2]
    pred_pad = torch.nn.ZeroPad2d(pad_size)(pred)
    GT_pad = torch.nn.ZeroPad2d(pad_size)(GT)

    gradient_loss = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            if i == pad_size and j == pad_size:
                continue
            data = pred
            neighbour = pred_pad[:, i:M + i, j:N + j]
            pred_gradient = data - neighbour

            data = GT
            neighbour = GT_pad[:, i:M + i, j:N + j]
            GT_gradient = data - neighbour

            gradient_loss = gradient_loss + (pred_gradient - GT_gradient).abs().sum()

    return gradient_loss / (BN * 3 * M * N * (kernel_size ** 2 - 1))

def train(train_gen, model, criterion, optimizer, epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(train_gen, 1):
        #input, targetdeblur, targetsr
        LR_Blur = batch[0]
        HR = batch[2]
        LR_Blur = LR_Blur.to(device)
        HR = HR.to(device)
        sr = model(LR_Blur)

        loss2 = criterion(sr, HR)
        mse = loss2 
        epoch_loss += loss2
        optimizer.zero_grad()
        mse.backward()
        optimizer.step()
        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): SR Loss{:.4f};".format(epoch, iteration, len(trainloader), loss2.cpu()))
    print("===>Epoch{} Part: Avg loss is :{:4f}".format(epoch, epoch_loss / len(trainloader)))
    return epoch_loss / len(trainloader)

if __name__ == '__main__':
    opt = parser.parse_args()
    Net = import_module('networks.' + opt.model)
    print(opt.resume)
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if torch.cuda.is_available() else torch.device('cpu')
    str_ids = opt.gpu_ids.split(',')
    torch.cuda.set_device(int(str_ids[0]))
    opt.seed = random.randint(1, 10000)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    train_dir = opt.dataset
    input_dir = join(train_dir, 'TrainLR/X4')
    lr_dir = join(train_dir, 'TrainLR/X4')
    gt_dir = join(train_dir, 'TrainGT')
    train_sets = [x for x in sorted(os.listdir(input_dir)) if x not in ['000', '011', '015', '020']]
    print("===> Loading model {} and criterion".format(opt.model))

    if opt.resume:
        if os.path.isfile(opt.resume):
            print("Loading from checkpoint {}".format(opt.resume))
            model = Net.make_model(opt)
            model_dict = model.state_dict()
            print(get_n_params(model))
            pretrained_model = torch.load(opt.resume, map_location=lambda storage, loc: storage)
            pretrained_dict = pretrained_model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(get_n_params(model))
            mkdir_steptraing()
    else:
        model = Net.make_model(opt)
        print(get_n_params(model))
        mkdir_steptraing()

    model = model.to(device)
    criterion = torch.nn.MSELoss(size_average=True)
    criterion = criterion.to(device)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    print()

    for i in range(opt.start_training_step, 5):
        opt.nEpochs   = training_settings[i-1]['nEpochs']
        opt.lr        = training_settings[i-1]['lr']
        opt.step      = training_settings[i-1]['step']
        opt.lr_decay  = training_settings[i-1]['lr_decay']
        opt.lambda_db = training_settings[i-1]['lambda_db']
        opt.gated     = training_settings[i-1]['gated']
        print(opt)
        for epoch in range(opt.start_epoch, opt.nEpochs+1):
            num = 0
            psnr_sr = 0
            adjust_learning_rate(epoch)
            random.shuffle(train_sets)
            for j in range(len(train_sets)):
                # if train_sets[j] in training_filters:
                #     continue
                print("Step {}:Training folder is {}".format(i, join(gt_dir, train_sets[j])))
                train_set = DataSet(join(input_dir, train_sets[j]), join(lr_dir, train_sets[j]), join(gt_dir, train_sets[j]), cropSize=opt.cropSize)
                trainloader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True, num_workers=1)
                avg_psnr_sr = train(trainloader, model, criterion, optimizer, epoch)
                psnr_sr += avg_psnr_sr
                num += 1
            checkpoint(i, epoch)
            psnr = psnr_sr / num
            print("===>Epoch{} Complete: Avg SR loss is :{:4f}".format(epoch, psnr))
            with open("Output_{}.txt".format(opt.name), "a+") as text_file:
                print("===>Epoch{} Complete: Avg SR loss is :{:4f}".format(epoch, psnr), file=text_file)

        opt.start_epoch = 1