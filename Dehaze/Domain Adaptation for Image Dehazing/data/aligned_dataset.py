import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import util.util as util


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.dir_AB = os.path.join(opt.dataroot, 'train_syn')
        self.dir_C = os.path.join(opt.dataroot, 'train_re')
        self.dir_D = os.path.join(opt.dataroot, 'train_syn_depth')
        self.dir_E = os.path.join(opt.dataroot, 'train_re_depth')
        self.dir_FG = os.path.join(opt.dataroot, 'test_syn')
        self.dir_H = os.path.join(opt.dataroot, 'test_re')

        self.AB_paths = sorted(make_dataset(self.dir_AB))
        self.C_paths = sorted(make_dataset(self.dir_C))
        self.D_paths = sorted(make_dataset(self.dir_D))
        self.E_paths = sorted(make_dataset(self.dir_E))
        self.FG_paths = sorted(make_dataset(self.dir_FG))
        self.H_paths = sorted(make_dataset(self.dir_H))

        self.transformPIL = transforms.ToPILImage()
        transform_list1 = [transforms.ToTensor()]
        transform_list2 = [transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]
        transform_list3 = [transforms.Normalize((0.5),
                                                (0.5))]

        self.transform1 = transforms.Compose(transform_list1)
        self.transform2 = transforms.Compose(transform_list2)
        self.transform3 = transforms.Compose(transform_list3)

    def __getitem__(self, index):
        # A, B is the image pair, hazy, gt respectively
        if self.opt.phase == 'train':
            AB_path = self.AB_paths[index]
            D_path = self.D_paths[index]
            # and C is the unlabel hazy image
            C_ind = random.randint(0, int(len(self.AB_paths)-1))
            C_path = self.C_paths[C_ind]
            E_path = self.E_paths[C_ind]
            # C_path = self.C_paths[random.randint(0, len(self.AB_paths)-2200)]
            AB = Image.open(AB_path).convert('RGB')
            C = Image.open(C_path).convert('RGB')
            D = Image.open(D_path)
            E = Image.open(E_path)

            ori_w = AB.width
            ori_h = AB.height
            AB = AB.resize((ori_w, ori_h), Image.BICUBIC)
            D = D.resize((D.width, D.height), Image.BICUBIC)

            C_w = C.width
            C_h = C.height
            ## resize the real image without label
            C = C.resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC)
            E = E.resize((self.opt.fineSize, self.opt.fineSize), Image.BICUBIC)

            AB = self.transform1(AB)
            C = self.transform1(C)
            D = self.transform1(D)
            E = self.transform1(E)

            ######### crop the training image into fineSize ########
            w_total = AB.size(2)
            w = int(w_total / 2)
            h = AB.size(1)
            w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

            A = AB[:, h_offset:h_offset + self.opt.fineSize,
                   w_offset:w_offset + self.opt.fineSize]
            B = AB[:, h_offset:h_offset + self.opt.fineSize,
                   w + w_offset:w + w_offset + self.opt.fineSize]
            D = D[:, h_offset:h_offset + self.opt.fineSize,
                w_offset:w_offset + self.opt.fineSize]

            w = C.size(2)
            h = C.size(1)
            w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))


            if (not self.opt.no_flip) and random.random() < 0.5:
                idx = [i for i in range(A.size(2) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                A = A.index_select(2, idx)
                B = B.index_select(2, idx)
                D = D.index_select(2, idx)


            A = self.transform2(A)
            B = self.transform2(B)
            C = self.transform2(C)
            D = self.transform3(D)
            E = self.transform3(E)

            if random.random()<0.5:
                noise = torch.randn(3, self.opt.fineSize, self.opt.fineSize) / 100
                #A = A + noise

            return {'A': A, 'B': B, 'C': C, 'D': D, 'E': E,'E_paths': E_path, 'D_paths': D_path,'C_paths': C_path,
                    'A_paths': AB_path, 'B_paths': AB_path}

        elif self.opt.phase == 'test':
            if self.opt.test_type == 'syn':
                FG_path = self.FG_paths[index]
                FG = Image.open(FG_path).convert('RGB')
                ori_w = FG.width
                ori_h = FG.height



                new_w = 1024
                new_h = 512
                FG = FG.resize((int(new_w), int(new_h)), Image.BICUBIC)
                FG = self.transform1(FG)
                FG = self.transform2(FG)
                F = FG[:,:,0:int(new_w/2)]
                G = FG[:,:,int(new_w/2):new_w]
                return {'A': F, 'B': G, 'A_paths': FG_path, 'B_paths': FG_path}


            elif self.opt.test_type == 'real':
                H_path = self.H_paths[index]
                H = Image.open(H_path).convert('RGB')
                H_w = H.width
                H_h = H.height

                new_w = int(np.floor(H_w / 16) * 16)
                new_h = int(np.floor(H_h / 16) * 16)
                H = H.resize((int(new_w), int(new_h)), Image.BICUBIC)
                H = self.transform1(H)
                H = self.transform2(H)
                return {'C': H, 'C_paths': H_path}



    def __len__(self):

        if self.opt.phase == 'train':
            return len(self.AB_paths)
        elif self.opt.phase == 'test':
            if self.opt.test_type == 'syn':
                return len(self.AB_paths)
            elif self.opt.test_type == 'real':
                return len(self.C_paths)

    def name(self):
        return 'AlignedDataset'
