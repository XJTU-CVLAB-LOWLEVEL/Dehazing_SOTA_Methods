from __future__ import print_function
import torch.optim as optim
import argparse
import os
from os.path import join
import torch
from torch.utils.data import DataLoader
from datasets.dataset_train_synt import Video_dataset_train, Video_dataset_test
from importlib import import_module
import random
import re
from networks.warp_image import estimate, Network
from networks.base_networks import SpaceToDepth
from networks.warp_image import Backward
import visualization as vl
import time
from math import log10
from vgg_loss import LossNetwork
from torchvision.models import vgg16
from hard_example_mining import HEM, HEM_CUDA
from visualization import show_image
from networks.VDHnet import VDHNet
import matplotlib.pyplot as plt
# Training settings
parser = argparse.ArgumentParser(description="PyTorch Train")
parser.add_argument("--batchSize", type=int, default=6, help="Training batch size")
parser.add_argument("--start_training_step", type=int, default=1, help="Training step")
parser.add_argument("--nEpochs", type=int, default=30, help="Number of epochs to train")
parser.add_argument("--start-epoch", type=int, default=1, help="Start epoch from 1")

parser.add_argument('--dataset', default="/home/lry/video_dehazing/datasets/VideoHazy_v2_synt/Train", type=str, help='Path of the training dataset(.h5)')
parser.add_argument('--dataset_test', default="/home/lry/video_dehazing/datasets/VideoHazy_v2_synt/Test", type=str, help='Path of the training dataset(.h5)')
parser.add_argument('--flow_dir', default="dx_Res_new", type=str, help='Path of the training dataset(.h5)')
parser.add_argument("--scale", default=1, type=int, help="Scale factor, Default: 4")
parser.add_argument("--cropSize", type=int, default=256, help="LR patch size")
parser.add_argument("--frames", type=int, default=5, help="the amount of input frames")
#parser.add_argument("--repeat", type=int, default=1, help="the amount of the dataset repeat per epoch")

# parser.add_argument("--pre", type=bool, default=False, help="Activated Pre-Dehazing Module")
parser.add_argument("--pre", default=0, type=int, help="Ways of Pre-Dehazing Module, 0: No Pre-Dehazing / 1: Pre-Dehazing / 2: Pre-Dehazing and Finetune")
parser.add_argument("--warped", default=0, type=int, help="Ways of Alignment, 0: No Alignment / 1: Input Alignment / 2: Feature Alignment / 3: Feature-based flow")
parser.add_argument("--tof", type=bool, default=False, help="Activated PWC-Net finwtuning")
parser.add_argument("--hr_flow", type=bool, default=False, help="Activated hr_flow")
parser.add_argument("--residual", type=bool, default=False, help="Activated hr_flow")
parser.add_argument("--cobi", type=bool, default=False, help="Use CoBi Loss")
parser.add_argument("--isTest", type=bool, default=False, help="Test or not")
parser.add_argument("--vgg", type=bool, default=False, help="Activated vgg loss")
parser.add_argument("--hem", type=bool, default=False, help="Activated hard negative mining")

parser.add_argument('--model', default='VDHnet', type=str, help='Import which network')
parser.add_argument('--gpu_ids', type=str, default='1', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
parser.add_argument('--name', default='VHDTrain', type=str, help='Filename of the training models')
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")

parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate, default=1e-5")
parser.add_argument("--step", type=int, default=7, help="Change the learning rate for every 30 epochs")
parser.add_argument("--lr_decay", type=float, default=0.5, help="Decay scale of learning rate, default=0.5")
parser.add_argument("--lambda_db", type=float, default=0.5, help="Weight of deblurring loss, default=0.5")
parser.add_argument("--lambda_GL", type=float, default=0.5, help="Weight of deblurring loss, default=0.5")

# Option for Residual dense network (RDN)
parser.add_argument('--G0', type=int, default=64,
                    help='default number of filters. (Use in RDN)')
parser.add_argument('--RDNkSize', type=int, default=3,
                    help='default kernel size. (Use in RDN)')
parser.add_argument('--RDNconfig', type=str, default='A',
                    help='parameters config of RDN. (Use in RDN)')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')


training_settings = [
    # {'nEpochs': 19, 'lr': 1e-4, 'step': 10, 'lr_decay': 0.75, 'lambda_db': 0.5, 'tof': False, 'batchSize': 6},
    # {'nEpochs': 80, 'lr': 1e-4, 'step': 8, 'lr_decay': 0.75, 'lambda_db': 0.5, 'tof': True, 'batchSize': 4}
    # {'nEpochs': 100, 'lr': 1e-4, 'step': 10, 'lr_decay': 0.75, 'lambda_db': 0.5, 'tof': False, 'batchSize': 6}
    # {'nEpochs': 20, 'lr': 1e-4, 'step': 10, 'lr_decay': 0.75, 'lambda_db': 0.5, 'tof': False, 'batchSize': 6},
    # {'nEpochs': 100, 'lr': 1e-4, 'step': 10, 'lr_decay': 0.75, 'lambda_db': 0.1, 'tof': True, 'batchSize': 4}
    # {'nEpochs': 100, 'lr': 1e-4, 'step': 45, 'lr_decay': 0.1, 'lambda_db': 0.1, 'tof': False, 'batchSize': 6}
    {'nEpochs': 30, 'lr': 1e-4, 'step': 45, 'lr_decay': 0.1, 'lambda_db': 0.1, 'tof': True, 'batchSize': 4}
]


os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        pp += nn
    return pp


def mkdir_steptraing():
    root_folder = os.path.abspath('.')
    models_folder = join(root_folder, 'models')
    models_folder = join(models_folder, opt.name)
    step1_folder, step2_folder, step3_folder, step4_folder = join(models_folder, '1'), join(models_folder, '2'), join(
        models_folder, '3'), join(models_folder, '4')
    isexists = os.path.exists(step1_folder) and os.path.exists(step2_folder) and os.path.exists(
        step3_folder) and os.path.exists(step4_folder)
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
    return int(trainingstep), int(start_epoch) + 1


def adjust_learning_rate(epoch):
    lr = opt.lr * (opt.lr_decay ** (epoch // opt.step))
    print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    for param_group in optimizer_pwc.param_groups:
        param_group['lr'] = lr


def checkpoint(step, epoch, model, moduleNetwork):
    root_folder = os.path.abspath('.')
    models_folder = join(root_folder, 'models')
    models_folder = join(models_folder, opt.name)
    model_out_path = join(models_folder, "VHDepoch{:02d}.pkl".format(epoch))
    torch.save(model, model_out_path)
    print("===>Checkpoint saved to {}".format(model_out_path))
    if opt.tof:
        model_out_path = 'models/{}/network-finetuned_{:02d}.pytorch'.format(opt.name,  epoch)
        torch.save(moduleNetwork.state_dict(), model_out_path)
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


def input_warping(LR, HR):
    # warp the inputs
    n, c, h, w = LR.size()
    frames = c // 3
    mid_frame = frames // 2
    tensorAnchor = LR[:, mid_frame * 3:(mid_frame + 1) * 3]
    tensorAnchor_HR = HR[:, mid_frame * 3:(mid_frame + 1) * 3]
    # LR = batch[3][:, mid_frame*3:(mid_frame+1)*3].to(device)
    LR_Warped = []
    flowmap = []
    if not opt.hr_flow:
        for frame in range(frames):
            if frame == frames // 2:
                LR_Warped.append(tensorAnchor)
                flowmap.append(torch.cuda.FloatTensor().resize_(n, 2, h, w).zero_())
                continue
            tensorSecond = LR[:, frame * 3:(frame + 1) * 3]
            flow, warpedTensorSecond = estimate(moduleNetwork, tensorAnchor, tensorSecond)
            LR_Warped.append(warpedTensorSecond)
            flowmap.append(flow)
        LR_Warped = torch.cat(LR_Warped, 1)
        flowmap = torch.cat(flowmap, 1)
        # print(LR_Warped.size())
    else:
        for frame in range(frames):
            if frame == mid_frame:
                LR_Warped.append(SpaceToDepth(block_size=4)(tensorAnchor_HR))
                flowmap.append(torch.cuda.FloatTensor().resize_(n, 2, 4 * h, 4 * w).zero_())
                continue
            tensorSecond_HR = HR[:, frame * 3:(frame + 1) * 3]
            flow_HR, warpedTensorSecond = estimate(moduleNetwork, tensorAnchor_HR, tensorSecond_HR)
            warpedTensorSecond_LR = SpaceToDepth(block_size=4)(warpedTensorSecond)
            LR_Warped.append(warpedTensorSecond_LR)
            flowmap.append(flow_HR)
        LR_Warped = torch.cat(LR_Warped, 1)
    return LR_Warped, flowmap


def train(train_gen, model, criterion, optimizer, epoch):
    epoch_loss = 0
    part_time_gpu = time.perf_counter()

    for iteration, batch in enumerate(train_gen, 1):
        # input, targetdeblur, targetsr
        Hazy = batch[0].to(device)
        HAZY_3 = Hazy[:, 6:9]
        #vl.show_image(HAZY_3)


        # vl.show_image(Hazy[0:1, 0:3])
        #vl.show_image(GT[:, 0:1])
       # Flow = batch[1].to(device)
        GT = batch[1].to(device)
        gt= GT[:,3:6]
        #vl.show_image(GT[:,3:6])
        SEG=batch[2].to(device)
        #vl.show_image(GT[0:1,0:1])

        mid_frame = opt.frames // 2
        # HR = F.interpolate(LR, scale_factor=4, mode='bicubic', align_corners=False)
        if opt.pre == 1:
            with torch.no_grad():
                pre_model.eval()
                n, c, h, w = Hazy.size()
                Hazy = Hazy.view(-1, 3, h, w)
                Hazy = pre_model(Hazy, Hazy)
                Hazy = Hazy.view(n, -1, h, w)
        elif opt.pre == 2:
            pre_model.train()
            n, c, h, w = Hazy.size()
            Hazy = Hazy.view(-1, 3, h, w)
            Hazy = pre_model(Hazy, Hazy)
            Hazy = Hazy.view(n, -1, h, w)

        if opt.warped != 0:
            if not opt.tof:
                with torch.no_grad():
                    moduleNetwork.eval()
                    Hazy_Warped, flowmap = input_warping(Hazy)
            else:
                moduleNetwork.train()
                Hazy_Warped, flowmap = input_warping(Hazy)

        '''
        try:
            [lr, sr] = model(LR_Blur_Warped, gated_Tensor, test_Tensor)
        except:
            [lr, sr] = model(LR_Blur, flowmap, gated_Tensor, test_Tensor)
        '''
        if opt.warped == 0:
            #flowmap = []
            Dehazedtrans = model(Hazy,SEG)
            Dehazed_31 = Dehazedtrans[:, 0:1]
            #vl.show_image(Dehazed_31)
            Dehazed_32 = Dehazedtrans[:, 1:2]
            #vl.show_image(Dehazed_32)
            Dehazed_33 = Dehazedtrans[:, 2:3]
            #vl.show_image(Dehazed_33)
            # a, b, c, d = HAZY_3.size()
            # outputs=[]
            # for i in range(a):
            #         TUy = HAZY_3[i:i + 1]
            #         TUy=torch.squeeze(TUy).cpu()
            #         TUy = (TUy.permute(1, 2, 0)).numpy()
            #         DehazedY=Dehazed_32[i:i+1]
            #         DehazedY = torch.squeeze(DehazedY).cpu()
            #         DehazedY= DehazedY.detach().numpy()
            #         Repair = Repair2(TUy, DehazedY)
            #         #plt.imshow(Repair)
            #         #plt.show()
            #         Repair=torch.from_numpy(Repair)
            #         Repair1=Repair.permute(2, 0, 1)
            #         # vl.show_image(Repair1)
            #         outputs.append(Repair)
            # RepairALL = torch.cat(outputs, 0)
            # H,I,J=RepairALL.size()
            #
            # h=H//256
            # RepairALL=RepairALL.view(h,256,256,3).permute(0, 3, 1, 2).contiguous().cuda()
        elif opt.warped == 1:
            # vl.show_image(Hazy[:, 0:3])
            # vl.show_image(Hazy[:, 3:6])
            # vl.show_image(Hazy[:, 6:9])
            # vl.show_image(Hazy[:, 9:12])
            # vl.show_image(Hazy[:, 12:15])
            # vl.show_image(Hazy_Warped[:, 0:3])
            # vl.show_image(Hazy_Warped[:, 3:6])
            # vl.show_image(Hazy_Warped[:, 6:9])
            # vl.show_image(Hazy_Warped[:, 9:12])
            # vl.show_image(Hazy_Warped[:, 12:15])
            # vl.show_image(GT)
            try:
                Dehazedtrans = model(Hazy_Warped, flowmap)
            except:
                Dehazed = model(Hazy_Warped, 1, 1, 1, 1, 1, flowmap)
        elif opt.warped == 2 or opt.warped == 3:
            try:
                Dehazed = model(Hazy, flowmap)
            except:
                Dehazed = model(Hazy, 1, 1, 1, 1, 1, flowmap)

        if opt.residual:
        #     Dehazed = Dehazed + Hazy[:, mid_frame * 3:(mid_frame + 1) * 3]
        # # loss1 = criterion(lr_deblur, LR_Deblur) * (max(15-epoch, 0) / 15)
        # if opt.hem:
        #     if opt.vgg:
        #         perceptual_loss = loss_network(Dehazed, GT)
        #     # loss_time = time.perf_counter()
        #     mask = hem_loss(Dehazed, GT)*0.5 + 1

            loss = criterion(Dehazedtrans, GT)
            mse = loss  # + opt.lambda_db * loss1
            #mse = mse + 0.1 * perceptual_loss
            # loss = criterion(Dehazed, GT)
            # torch.cuda.synchronize()
            # loss_time_gpu = time.perf_counter() - loss_time
            # print("===> Part Time: {:.6f}".format(loss_time_gpu))
        else:
            loss = criterion(Dehazedtrans, GT)
            mse = loss  # + opt.lambda_db * loss1
            mse.requires_grad_(True)
            if opt.vgg:
                perceptual_loss = loss_network(Dehazed, GT)
                mse = mse + 0.1 * perceptual_loss

        epoch_loss += loss
        # epoch_loss_db += loss1
        optimizer.zero_grad()
        #optimizer_pwc.zero_grad()
        mse.backward()
        optimizer.step()
        print(iteration)
        # if opt.tof:
        #     optimizer_pwc.step()
        # if iteration % 50 == 0:
        #     print("===> Epoch[{}]({}/{}): MSE Loss{:.4f};".format(epoch, iteration, len(trainloader), loss.cpu()))
    print("===>Avg MSE loss is :{:4f}".format(epoch_loss / len(trainloader)))
    part_eval_time_gpu = time.perf_counter() - part_time_gpu
    print("===> Part Time: {:.6f}".format(part_eval_time_gpu))
    #torch.save(model.state_dict(), "Epoch" + str(epoch) + '.pth')
    return epoch_loss / len(trainloader)


def test(test_gen, model, criterion):
    avg_psnr = 0
    num = 0
    med_time = []
    mid_frame = opt.frames // 2

    with torch.no_grad():
        for iteration, batch in enumerate(test_gen, 1):
            # if iteration%50 != 0:
            # continue
            Hazy= batch[0].to(device)
            GT = batch[1].to(device)
            SEG = batch[2].to(device)

            start_time = time.perf_counter()  # -------------------------begin to deal with an image's time
            if opt.warped == 0:
                Dehazed = model(Hazy,SEG)

            # modify
            Dehazed = torch.clamp(Dehazed, min=0, max=1)
            torch.cuda.synchronize()  # wait for CPU & GPU time syn
            evalation_time = time.perf_counter() - start_time  # ---------finish an image
            med_time.append(evalation_time)
            mse = criterion(Dehazed, GT)
            psnr = 10 * log10(1 / mse)
            # print('{}'.format(psnr))
            # print("Processing {},{}".format(iteration, psnr))
            avg_psnr += psnr
            num += 1

        print("Avg. PSNR:{:4f} ".format(avg_psnr / num))
        return avg_psnr / num


def model_test(model):
    model = model.to(device)
    criterion = torch.nn.MSELoss(size_average=True)
    criterion = criterion.to(device)
    print(opt)
    psnr = test(testloader, model, criterion)
    return psnr


if __name__ == '__main__':
    opt = parser.parse_args()
    Net = import_module('networks.' + opt.model)
    print('Alignment Mode: {} '.format(opt.warped))
    device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if torch.cuda.is_available() else torch.device('cpu')
    str_ids = opt.gpu_ids.split(',')
    gpus = [int(i) for i in str_ids]

    torch.cuda.set_device(int(str_ids[0]))
    opt.seed = random.randint(1, 10000)
    opt.seed = 7439
    print('============> The Seed is {}'.format(opt.seed))
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    train_dir = opt.dataset
    input_dir = join(train_dir, 'dx_Res_new')
    seg_dir = join(train_dir, 'segment')
    gt_dir = join(train_dir, 'dx_Transmission')
    train_sets = [x for x in sorted(os.listdir(gt_dir)) ]


    test_dir = opt.dataset_test
    input_dir_test = join(test_dir, 'dx_Res_new')

    gt_dir_test = join(test_dir, 'dx_Transmission')
    seg_dir_test = join(test_dir, 'segment')
    test_sets = [x for x in sorted(os.listdir(gt_dir_test))]

    print("===> Loading model {} and criterion".format(opt.model))
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("Loading VSE model from checkpoint {}".format(opt.resume))
            model = Net.make_model(opt)
            model_dict = model.state_dict()
            print(get_n_params(model))
            pretrained_model = torch.load(opt.resume, map_location=lambda storage, loc: storage)
            pretrained_dict = pretrained_model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print(get_n_params(model))
            opt.start_training_step, opt.start_epoch = which_trainingstep_epoch(opt.resume)
            opt.tof = training_settings[opt.start_training_step - 1]['tof']
            moduleNetwork = VDHNet().to(device)
            if opt.tof == True:
                print("Loading PWC model from models/{}/{}/network-finetuned_{:02d}.pytorch".format(opt.name,
                                                                                                    opt.start_training_step,
                                                                                                    opt.start_epoch - 1))
                moduleNetwork.load_state_dict(torch.load(
                    'models/{}/{}/network-finetuned_{:02d}.pytorch'.format(opt.name, opt.start_training_step,
                                                                           opt.start_epoch - 1)))
            else:
                print("Loading PWC model from models/network-default.pytorch")
                moduleNetwork.load_state_dict(torch.load('models/network-default.pytorch'))
            mkdir_steptraing()
    else:
        model =Net.make_model(opt)
        moduleNetwork = VDHNet(opt).to(device)


    if opt.pre != 0:
        print('================== Use Pre-trained Dehazing Network ===================')
        # pre_model = torch.load('models/baseline_S_single_L1/1/GFN_epoch_87.pkl', map_location=lambda storage, loc: storage)
        pre_model = torch.load('models/baseline_single_L1/1/GFN_epoch_64.pkl',
                               map_location=lambda storage, loc: storage)
        pre_model = pre_model.to(device)

    model = model.to(device)
    # model = torch.nn.DataParallel(model, device_ids=gpus)
    # if opt.cobi:
    #     criterion = cl.ContextualBilateralLoss()
    # else:
    #     criterion = torch.nn.L1Loss(size_average=True)
    criterion=torch.nn.MSELoss()
    criterion = criterion.to(device)

    if opt.vgg:
        print('======>Initializing VGG network')
        vgg_model = vgg16(pretrained=True).features[:16]
        vgg_model = vgg_model.to(device)
        for param in vgg_model.parameters():
            param.requires_grad = False
        loss_network = LossNetwork(vgg_model)
        loss_network.eval()

    if opt.hem:
        print('======>Initializing hard example mining loss')
        hem_loss = HEM(device = device)
        # hem_loss_cuda = HEM_CUDA(random_thre_p=1)

    # criterion = torch.nn.DataParallel(criterion, device_ids=gpus)
    # moduleNetwork = torch.nn.DataParallel(moduleNetwork, device_ids=gpus)
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    optimizer_pwc = optim.Adam(moduleNetwork.parameters(), lr=opt.lr)

    for i in range(opt.start_training_step, 5):
        opt.nEpochs = training_settings[i - 1]['nEpochs']
        opt.lr = training_settings[i - 1]['lr']
        opt.step = training_settings[i - 1]['step']
        opt.lr_decay = training_settings[i - 1]['lr_decay']
        opt.lambda_db = training_settings[i - 1]['lambda_db']
        opt.tof = training_settings[i - 1]['tof']
        opt.batchSize = training_settings[i - 1]['batchSize']
        print(opt)
        for epoch in range(opt.start_epoch, opt.nEpochs + 1):
            avg_val_psnr = 0
            avg_mse_loss = 0
            num = 0

            # ###########Train##############
            adjust_learning_rate(epoch)
            random.shuffle(train_sets)
            for j in range(len(train_sets)):
                for k in range(9):
                    sub_input_dir = os.path.join(input_dir, train_sets[j],str(k+1))
                    sub_gt_dir = os.path.join(gt_dir, train_sets[j],str(k+1))
                    sub_seg_dir = os.path.join(seg_dir,train_sets[j],str(1))
                   # sub_seg_dir = os.path.join(seg_dir, train_sets[j])
                    print("\nStep [{}] Epoch [{}] Part [{}]: Input folder is {}, GT folder is {}".format(i, epoch, j, join(sub_input_dir),
                                                                                  join(sub_gt_dir)))
                    train_set = Video_dataset_train(sub_input_dir,
                                         sub_gt_dir ,sub_seg_dir,cropSize=opt.cropSize,frames=opt.frames)
                    trainloader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True, num_workers=0)
                    if hasattr(torch.cuda, 'empty_cache'):
                        torch.cuda.empty_cache()
                    mse_loss = train(trainloader, model, criterion, optimizer, epoch)
                    avg_mse_loss += mse_loss
                    # psnr_db += avg_psnr_db
                    num += 1
            checkpoint(i, epoch, model, moduleNetwork)
            avg_mse_loss /= num
            print("\n===>Step [{}] Epoch [{}] Complete \n===>Training Avg. MSE loss is :{:4f} \n".format(i, epoch, avg_mse_loss))

            ###########Test##############
            print('Testing model of Step [{}] Epoch[{}]'.format(i, epoch))
            for j in range(len(test_sets)):
                for k in range(1):
                    sub_input_dir_test = os.path.join(input_dir_test, test_sets[j],str(k+1))
                    sub_gt_dir_test = os.path.join(gt_dir_test, test_sets[j],str(k+1))
                    sub_seg_dir_test= os.path.join(seg_dir_test, test_sets[j],str(k+1))
                    print("Folder {}: Input folder is {},segment folder is {}, GT folder is {}".format(j+1, join(sub_input_dir_test, test_sets[j],str(k+1)),
                                                                                                       join(sub_seg_dir_test,test_sets[j],str(k+1)),
                                                                                  join(sub_gt_dir_test, test_sets[j]),str(k+1)))
                    test_set = Video_dataset_test(sub_input_dir_test,
                                         sub_gt_dir_test,sub_seg_dir_test, cropSize=opt.cropSize,frames=opt.frames)
                    testloader = DataLoader(dataset=test_set, batch_size=1, shuffle=False, pin_memory=False)

                    # testloader = DataLoader(DataValSetV('/data/Dataset/NYUv2/test_video', 'hazy', frames=opt.frames), batch_size=1,
                    #                         shuffle=False,
                    #                         pin_memory=False)
                    val_psnr = model_test(model)

                    avg_val_psnr += val_psnr
            avg_val_psnr /= (len(test_sets))
            print("===>Step [{}] Epoch [{}] Test Complete \n ===>Testing Avg. PSNR is :{:4f}".format(i, epoch, avg_val_psnr))
            # print('test now model of Epoch:{} on REDS4_2'.format(epoch-1))
            # testloader = DataLoader(DataValSetV('/4TB1/VSR/NTIRE-VDBSR/REDS4_2', 'GT', frames=5), batch_size=1,
            #                         shuffle=False,
            #                         pin_memory=False)
            # model_test(model)
            with open("/home/lry/video_dehazing/Logs/{}.txt".format(opt.name), "a+") as text_file:
                print("===>Epoch{} Complete: Avg MSE loss is :{:4f}, Avg Testing PSNR is :{:4f}".format(epoch, avg_mse_loss,
                                                                                                   avg_val_psnr),
                      file=text_file)
        opt.start_epoch = 1