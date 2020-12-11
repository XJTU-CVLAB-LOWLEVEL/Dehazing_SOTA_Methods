

# --- Imports --- #
import os
import time
import torch
import argparse
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from train_nitre_data import TrainData as data
from train_nitre_data import TrainData_gdn as data1
from model import Generate_quarter, Generate_quarter_refine, Generate, Discriminator, LossD, LossFeat, Lap
from utils import to_psnr, print_log,print_val_log, validation, adjust_learning_rate
from torchvision.models import vgg16
from perceptual import LossNetwork
from torch.utils.data import ConcatDataset
from torch.nn import init
import os
from test_nitre import main

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        #init.xavier_uniform(m.weight)
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        #init.xavier_uniform(m.weight)
        m.weight.data.normal_(0, 0.02)
        m.bias.data.zero_()

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for GridDehazeNet')
parser.add_argument('--learning_rate', help='Set the learning rate', default=1e-4, type=float)
parser.add_argument('--crop_size', help='Set the crop_size', default=[240, 240], nargs='+', type=int)
parser.add_argument('--train_phrase', help='Three phrase 1,2,3', default=1, type=float)
parser.add_argument('--train_batch_size', help='Set the training batch size', default=4,type=int)
parser.add_argument('--network_height', help='Set the network height (row)', default=3, type=int)
parser.add_argument('--network_width', help='Set the network width (column)', default=6, type=int)
parser.add_argument('--num_dense_layer', help='Set the number of dense layer in RDB', default=4, type=int)
parser.add_argument('--growth_rate', help='Set the growth rate in RDB', default=16, type=int)
parser.add_argument('--lambda_loss', help='Set the lambda in loss function', default=0.04, type=float)
parser.add_argument('--category', help='Set image category (indoor or outdoor?)', default='nitre', type=str)
parser.add_argument('--test_img', help='Test image directory', default='', type=str)
parser.add_argument('--test_gt', help='Test gt directory', default='', type=str)
parser.add_argument('--val_data', help='val data directory', default='./data/VideoHazy_v3/resize_test/', type=str)
args = parser.parse_args()

learning_rate = args.learning_rate
crop_size = args.crop_size
train_phrase = args.train_phrase
train_batch_size = args.train_batch_size
network_height = args.network_height
network_width = args.network_width
num_dense_layer = args.num_dense_layer
growth_rate = args.growth_rate
lambda_loss = args.lambda_loss
category = args.category
test_img = args.test_img
test_gt = args.test_gt
val_data = args.val_data


print('--- Hyper-parameters for training ---')
print('learning_rate: {}\ncrop_size: {}\ntrain_batch_size: {}\nnetwork_height: {}\nnetwork_width: {}\n'
      'num_dense_layer: {}\ngrowth_rate: {}\nlambda_loss: {}\ncategory: {}'.format(learning_rate, crop_size,
      train_batch_size, network_height, network_width, num_dense_layer, growth_rate, lambda_loss, category))

# --- Set category-specific hyper-parameters  --- #
if category == 'nitre':
    num_epochs = 8000
    train_data_dir = './data/VideoHazy_v3/train/'
    train_data_dir1 = './data/VideoHazy_v3/train/hazy/'
    train_data_gt1 = './data/VideoHazy_v3/train/gt/'
    # train_data_dir2 = './data/train/ntire_hom/'
    # train_data_gt2 = './data/train/ntire_gt/'
    # train_data_dir3 = './data/train/medium/'
    # train_data_gt3 = './data/train/ntire_gt/'
    # train_data_dir4 = './data/train/light/'
    # train_data_gt4 = './data/train/ntire_gt/'
else:
    raise Exception('Wrong image category. Set it to indoor or outdoor for RESIDE dateset.')


# --- Gpu device --- #
#device_ids = [3]
device_ids = [1]
print(device_ids)
device = torch.device("cuda:1")
writer = SummaryWriter()

# --- Define the network --- #
if train_phrase == 1:
    net = Generate_quarter(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)
    net.apply(initialize_weights)
    print('First Phrase Init!')
    optimizer = torch.optim.Adam(list(net.parameters()), lr=learning_rate, betas=(0.5, 0.999))
    net = net.to(device)
    net = nn.DataParallel(net, device_ids=device_ids)
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
if train_phrase == 2:
    net = Generate_quarter(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)
    params=[]
    G2 = Generate_quarter_refine(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)
    print('Second Phrase Init!')
    #net.apply(initialize_weights)
    #G2.apply(initialize_weights)
    params = list(net.parameters())+ list(G2.parameters())
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.5, 0.999))
    #optimizer = torch.optim.SGD(params, lr=learning_rate)
    G2 = G2.to(device)
    net = net.to(device)
    G2 = nn.DataParallel(G2, device_ids=device_ids)
    net = nn.DataParallel(net, device_ids=device_ids)
    net.load_state_dict(torch.load('./checkpoint/1_615.tar'))
    G2.load_state_dict(torch.load('./checkpoint/1_615.tar'))	
    pytorch_total_params = sum(p.numel() for p in G2.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
if train_phrase == 3:
    net = Generate_quarter(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)
    G2 = Generate_quarter_refine(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)
    G2 = G2.to(device)
    net = net.to(device)
    G2 = nn.DataParallel(G2, device_ids=device_ids)
    net = nn.DataParallel(net, device_ids=device_ids)
    G3 = Generate(height=network_height, width=network_width, num_dense_layer=num_dense_layer, growth_rate=growth_rate)
    G3.apply(initialize_weights)
    net.load_state_dict(torch.load('./checkpoint/2-1810_G1.tar'))
    G2.load_state_dict(torch.load('./checkpoint/2_1810_G2.tar'))
    params = list(net.parameters())+ list(G2.parameters()) + list(G3.parameters())
    G3 = G3.to(device)
    #G3.apply(initialize_weights)
    G3 = nn.DataParallel(G3, device_ids=device_ids)
    #G3.load_state_dict(torch.load('./checkpoint/33_35_G3.tar'))
    optimizer = torch.optim.Adam(params, lr=learning_rate, betas=(0.5, 0.999))
    pytorch_total_params = sum(p.numel() for p in G3.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))

# --- Define the perceptual loss network --- #
vgg_model = vgg16(pretrained=True).features[:16]
vgg_model = vgg_model.to(device)
for param in vgg_model.parameters():
    param.requires_grad = False
loss_network = LossNetwork(vgg_model)
loss_network.eval()
loss_lap = Lap()

start_epoch = 0

# --- Load training data and validation/test data --- #
# train_data1 = data(crop_size, train_data_dir1)
train_data1 = data1(crop_size, train_data_dir)
#train_data1 = data(crop_size, train_data_dir1, train_data_gt1)
# train_data2 = data(crop_size, train_data_dir2, train_data_gt2)
# train_data3 = data(crop_size, train_data_dir3, train_data_gt3)
# train_data4 = data(crop_size, train_data_dir4, train_data_gt4)
# train_data = ConcatDataset([train_data1, train_data2])

train_data_loader = DataLoader(train_data1, batch_size=train_batch_size, shuffle=True,num_workers=2,pin_memory=True)
print('Data Finished!')
# --- Previous PSNR and SSIM in testing --- #


loss_rec1 = nn.SmoothL1Loss()
loss_rec2 = nn.MSELoss()
num = 0
avg = nn.AvgPool2d(3, stride = 2, padding = 1)
for epoch in range(start_epoch, num_epochs):
    psnr_list = []
    start_time = time.time()
    adjust_learning_rate(optimizer, epoch, category=category)

    for batch_id, train_data in enumerate(train_data_loader):

        optimizer.zero_grad()
        haze, gt, haze_gt = train_data
        haze = haze.to(device)
        gt = gt.to(device)
        haze_gt = haze_gt.to(device)
        gt_quarter_1 = F.interpolate(gt, scale_factor = 0.25,recompute_scale_factor=True)
        gt_quarter_2 = F.interpolate(gt, scale_factor = 0.25,recompute_scale_factor=True)

        # --- Forward + Backward + Optimize --- #

        if train_phrase == 1:
            dehaze_1, feat_extra_1 = net(haze)
            rec_loss1 = loss_rec1(dehaze_1, gt)
            perceptual_loss = loss_network(dehaze_1, gt)
            lap_loss = loss_lap(dehaze_1, gt)
            psnr = to_psnr(dehaze_1, gt)
            psnr_list.extend(to_psnr(dehaze_1, gt))
            train_info = to_psnr(dehaze_1, gt)
        if train_phrase == 2:
            dehaze_1, feat_extra_1 = net(haze)
            dehaze_2, feat, feat_extra_2 = G2(dehaze_1)
            rec_loss1 = (loss_rec1(dehaze_2, gt) + loss_rec1(dehaze_1, gt))/2.0
            rec_loss2 = loss_rec2(dehaze_2, gt)
            perceptual_loss = loss_network(dehaze_2, gt)
            lap_loss = loss_lap(dehaze_2, gt)
            psnr = to_psnr(dehaze_2, gt)
            psnr_list.extend(to_psnr(dehaze_2, gt))
            train_info = to_psnr(dehaze_2, gt)
        if train_phrase == 3:
            dehaze_1, feat_extra_1 = net(F.interpolate(haze, scale_factor = 0.25,recompute_scale_factor=True))
            dehaze_2, feat, feat_extra_2 = G2(dehaze_1)
            dehaze = G3(haze, F.interpolate(dehaze_2, scale_factor = 4,recompute_scale_factor=True), feat)
            rec_loss1 = (loss_rec1(dehaze, gt) + loss_rec1(dehaze_2, gt_quarter_2)+loss_rec1(dehaze_1, gt_quarter_1))/3.0
            rec_loss2 = loss_rec2(dehaze, gt)
            perceptual_loss = (loss_network(dehaze, gt) + loss_network(F.interpolate(dehaze, scale_factor = 0.5,recompute_scale_factor=True), F.interpolate(gt, scale_factor = 0.5,recompute_scale_factor=True)) + loss_network(F.interpolate(dehaze, scale_factor = 0.25,recompute_scale_factor=True), F.interpolate(gt, scale_factor = 0.25,recompute_scale_factor=True)) + loss_network(dehaze_2, gt_quarter_2))/4.0
            lap_loss = loss_lap(dehaze, gt)
            psnr = to_psnr(dehaze, gt)
            psnr_list.extend(to_psnr(dehaze, gt))
            train_info = to_psnr(dehaze, gt)

        loss = (rec_loss1) * 1.2 + 0.04 *perceptual_loss #+ 0.5 * lap_loss

        loss.backward()
        optimizer.step()

        if not (batch_id % 10):
            print('Epoch: {0}, Iteration: {1}'.format(epoch, batch_id))
            print (sum(train_info) / len(train_info))
            writer.add_scalar('scalar/loss_w/ IN', loss, num)
            writer.add_scalar('scalar/psnr_w/ IN', sum(psnr) / len(psnr), num)
            num = num + 1

    # --- Calculate the average training PSNR in one epoch --- #
    train_psnr = sum(psnr_list) / len(psnr_list)
    print_log(epoch+1, train_psnr, category)
    if epoch % 5==0:
        if train_phrase == 1:
            torch.save(net.state_dict(), './checkpoint/'+str(int(train_phrase))+'_'+str(epoch)+'.tar')
        if train_phrase == 2:
            torch.save(net.state_dict(),'./checkpoint/'+str(int(train_phrase))+'-'+str(int(epoch))+'_G1.tar')
            torch.save(G2.state_dict(), './checkpoint/'+str(int(train_phrase))+'_'+str(int(epoch))+'_G2.tar')
        if train_phrase == 3:
            torch.save(net.state_dict(),'./checkpoint/'+str(int(train_phrase))+'-'+str(int(epoch))+'_G1.tar')
            torch.save(G2.state_dict(), './checkpoint/'+str(int(train_phrase))+'_'+str(int(epoch))+'_G2.tar')
            torch.save(G3.state_dict(), './checkpoint/'+str(int(train_phrase))+str(int(train_phrase))+'_'+str(epoch)+'_G3.tar')
        test_psnr = main(val_data, train_phrase, epoch)#test_img, test_gt,
        print_val_log(test_psnr)
        # test_psnr = main(test_img, test_gt, train_phrase, epoch)
        writer.add_scalar('scalar/psnr_test_w/ IN', test_psnr, epoch)
        
