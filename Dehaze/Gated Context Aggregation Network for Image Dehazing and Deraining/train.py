import os
import datetime
import argparse
import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from ImagePairPrefixFolder import ImagePairPrefixFolder, var_custom_collate
from utils import MovingAvg
from tf_visualizer import TFVisualizer

parser = argparse.ArgumentParser()
parser.add_argument('--network', default='GCANet')
parser.add_argument('--name', default='default_exp')
parser.add_argument('--gpu_ids', default='3')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_step', type=int, default=40)
parser.add_argument('--lr_gamma', type=float, default=0.1)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--checkpoints_dir', default='checkpoint')
parser.add_argument('--logDir', default='tblogdir')
parser.add_argument('--resume_dir', default='checkpoint2/default_exp')
parser.add_argument('--resume_epoch', type=int, default=19)
parser.add_argument('--save_epoch', type=int, default=2)
parser.add_argument('--save_latest_freq', type=int, default=5000)
parser.add_argument('--test_epoch', type=int, default=2)
parser.add_argument('--test_max_size', type=int, default=1080)
parser.add_argument('--size_unit', type=int,  default=4)
parser.add_argument('--print_iter', type=int,  default=100)
parser.add_argument('--input_folder', default='examples/train/haze')
parser.add_argument('--gt_folder', default='examples/train/gt')
parser.add_argument('--test_input_folder', default='examples/test/haze')
parser.add_argument('--test_gt_folder', default='examples/test/gt')
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--only_residual', action='store_true', help='regress residual rather than image')
parser.add_argument('--loss_func', default='l2', help='l2|l1')
parser.add_argument('--inc', type=int, default=3)
parser.add_argument('--outc', type=int, default=3)
parser.add_argument('--force_rgb', action='store_true')
parser.add_argument('--no_edge', action='store_true')

opt = parser.parse_args()

opt.input_folder = os.path.expanduser(opt.input_folder)
opt.gt_folder = os.path.expanduser(opt.gt_folder)
opt.test_input_folder = os.path.expanduser(opt.test_input_folder)
opt.test_gt_folder = os.path.expanduser(opt.test_gt_folder)
opt.only_residual = True

if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.name)):
    os.makedirs(os.path.join(opt.checkpoints_dir, opt.name))
opt.resume_dir = opt.resume_dir if opt.resume_dir != '' else os.path.join(opt.checkpoints_dir, opt.name)

visualizer = TFVisualizer(opt)
### Log out
with open(os.path.realpath(__file__), 'r') as fid:
    visualizer.print_logs(fid.read())

## print argument
for key, val in vars(opt).items():
    visualizer.print_logs('%s: %s' % (key, val))

opt.gpu_ids = [int(x) for x in opt.gpu_ids.split(',')]
assert all(0 <= x <= torch.cuda.device_count() for x in opt.gpu_ids), 'gpu id should ' \
                                                      'be 0~{0}'.format(torch.cuda.device_count())
torch.cuda.set_device(opt.gpu_ids[0])


train_dataset = ImagePairPrefixFolder(opt.input_folder, opt.gt_folder, size_unit=opt.size_unit, force_rgb=opt.force_rgb)
train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True,
                              collate_fn=var_custom_collate, pin_memory=True,
                              num_workers=opt.num_workers)

opt.do_test = opt.test_gt_folder != ''
if opt.do_test:
    test_dataset = ImagePairPrefixFolder(opt.test_input_folder, opt.test_gt_folder,
                                         max_img_size=opt.test_max_size, size_unit=opt.size_unit, force_rgb=opt.force_rgb)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                 collate_fn=var_custom_collate, pin_memory=True,
                                 num_workers=1)

total_inc = opt.inc if opt.no_edge else opt.inc + 1
if opt.network == 'GCANet':
    from GCANet import GCANet
    net = GCANet(in_c=total_inc, out_c=3, only_residual=opt.only_residual)
else:
    print('network structure %s not supported' % opt.network)
    raise ValueError

if opt.loss_func == 'l2':
    loss_crit = torch.nn.MSELoss()
elif opt.loss_func == 'l1':
    loss_crit = torch.nn.SmoothL1Loss()
else:
    print('loss_func %s not supported' % opt.loss_func)
    raise ValueError
pnsr_crit = torch.nn.MSELoss()

if len(opt.gpu_ids) > 0:
    net.cuda()
    if len(opt.gpu_ids) > 1:
        net = torch.nn.DataParallel(net)
    loss_crit = loss_crit.cuda()
    pnsr_crit = pnsr_crit.cuda()

optimizer = optim.Adam(net.parameters(), lr=opt.lr)
step_optim_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step, gamma=opt.lr_gamma)
loss_avg = MovingAvg(pool_size=50)

start_epoch = 0
total_iter = 0

if os.path.exists(os.path.join(opt.checkpoints_dir, opt.name, 'latest.pth')):
    print('resuming from latest.pth')
    latest_info = torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'latest.pth'))
    start_epoch = latest_info['epoch']
    total_iter = latest_info['total_iter']
    if isinstance(net, torch.nn.DataParallel):
        net.module.load_state_dict(latest_info['net_state'])
    else:
        net.load_state_dict(latest_info['net_state'])
    optimizer.load_state_dict(latest_info['optim_state'])

if opt.resume_epoch > 0:
    start_epoch = opt.resume_epoch
    total_iter = opt.resume_epoch * len(train_dataloader)
    resume_path = os.path.join(opt.resume_dir, 'net_epoch_%d.pth') % opt.resume_epoch
    print('resume from : %s' % resume_path)
    assert os.path.exists(resume_path), 'cannot find the resume model: %s ' % resume_path
    if isinstance(net, torch.nn.DataParallel):
        net.module.load_state_dict(torch.load(resume_path))
    else:
        net.load_state_dict(torch.load(resume_path))

for epoch in range(start_epoch, opt.epochs):
    visualizer.print_logs("Start to train epoch %d" % epoch)
    net.train()
    for iter, data in enumerate(train_dataloader):
        total_iter += 1
        optimizer.zero_grad()
        step_optim_scheduler.step(epoch)

        batch_input_img, batch_input_edge,  batch_gt = data
        if len(opt.gpu_ids) > 0:
            batch_input_img, batch_input_edge, batch_gt = batch_input_img.cuda(), batch_input_edge.cuda(), batch_gt.cuda()

        if opt.no_edge:
            batch_input = batch_input_img
        else:
            batch_input = torch.cat((batch_input_img, batch_input_edge), dim=1)
        batch_input_v = Variable(batch_input)
        if opt.only_residual:
            batch_gt_v = Variable(batch_gt - (batch_input_img+128))
        else:
            batch_gt_v = Variable(batch_gt)

        pred = net(batch_input_v)

        loss = loss_crit(pred, batch_gt_v)
        avg_loss = loss_avg.set_curr_val(loss.data)

        loss.backward()
        optimizer.step()

        if iter % opt.print_iter == 0:
            visualizer.plot_current_losses(total_iter, { 'loss': loss})
            visualizer.print_logs('%s Step[%d/%d], lr: %f, mv_avg_loss: %f, loss: %f' %
                                    (str(datetime.datetime.now()).split(' ')[1], iter, len(train_dataloader),
                                     step_optim_scheduler.get_lr()[0], avg_loss, loss))

        if total_iter % opt.save_latest_freq == 0:
            latest_info = {'total_iter': total_iter,
                           'epoch': epoch,
                           'optim_state': optimizer.state_dict()}
            if len(opt.gpu_ids) > 1:
                latest_info['net_state'] = net.module.state_dict()
            else:
                latest_info['net_state'] = net.state_dict()
            print('save lastest model.')
            torch.save(latest_info, os.path.join(opt.checkpoints_dir, opt.name, 'latest.pth'))

    if (epoch+1) % opt.save_epoch == 0 :
        visualizer.print_logs('saving model for epoch %d' % epoch)
        if len(opt.gpu_ids) > 1:
            torch.save(net.module.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'net_epoch_%d.pth' % (epoch+1)))
        else:
            torch.save(net.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'net_epoch_%d.pth' % (epoch + 1)))

        if opt.do_test:
            avg_psnr = 0
            task_cnt = 0
            net.eval()
            with torch.no_grad():
                for iter, data in enumerate(test_dataloader):
                    batch_input_img, batch_input_edge,  batch_gt = data
                    if len(opt.gpu_ids) > 0:
                        batch_input_img, batch_input_edge, batch_gt = batch_input_img.cuda(), batch_input_edge.cuda(), batch_gt.cuda()

                    if opt.no_edge:
                        batch_input = batch_input_img
                    else:
                        batch_input = torch.cat((batch_input_img, batch_input_edge), dim=1)
                    batch_input_v = Variable(batch_input)
                    batch_gt_v = Variable(batch_gt)


                    pred = net(batch_input_v)

                    if opt.only_residual:
                        loss = pnsr_crit(pred+Variable(batch_input_img+128), batch_gt_v)
                    else:
                        loss = pnsr_crit(pred, batch_gt_v)
                    avg_psnr += 10 * np.log10(255 * 255 / loss.item())
                    task_cnt += 1

            visualizer.print_logs('Testing for epoch: %d' % epoch)
            visualizer.print_logs('Average test PNSR is %f for %d images' % (avg_psnr/task_cnt, task_cnt))