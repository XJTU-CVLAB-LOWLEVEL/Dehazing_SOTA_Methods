import numpy as np
import os
import ntpath
import time
import utils
# from scipy.misc import imresize
from tensorboardX import SummaryWriter


class TFVisualizer():
    def __init__(self, opt):
        self.tf_visualizer = SummaryWriter(os.path.join(opt.logDir, opt.name))
        self.opt = opt
        self.saved = False
        self.ncols = 4

        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        self.saved = False

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, iter_mark, epoch, save_result):
        for label, image in visuals.items():
            img_gid = utils.tensor2imgrid(image)
            self.tf_visualizer.add_image(label, img_gid, iter_mark)

    # losses: dictionary of error labels and values
    def plot_current_losses(self, iter_mark, losses):
        # for label, loss in losses.items():
        #     self.tf_visualizer.add_scalar(label, loss, iter_mark)
        self.tf_visualizer.add_scalars('training loss', losses, iter_mark)

    def print_logs(self, message):
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

