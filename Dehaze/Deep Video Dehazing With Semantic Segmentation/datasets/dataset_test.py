import torch.utils.data as data
import numpy as np
from PIL import Image
from os.path import join
from PIL import ImageOps
import os
import random
import torch
import cv2
from Resolution import Repair1,Repair2
import visualization as vl
import matplotlib.pyplot as plt

# Import relevant codes
# from utility.utility import is_image_file

def is_image_file(filename):
    if filename.startswith('._'):
        return
    return any(filename.endswith(extension) for extension in [".png", ".JPG", ".jpeg"])

# ---------------------
#    Dataset_train
# ---------------------
class Video_dataset_train(data.Dataset):
    def __init__(self, train_dir,gt_dir,seg_dir ,cropSize=128, frames=5):
        super(Video_dataset_train, self).__init__()
        self.gt_dir = gt_dir
        self.input_dir = train_dir
        self.seg_dir = seg_dir

        self.cropSize = cropSize
        self.fineSize = self.cropSize
        self.frames = frames
        # Load gt list
        self.gt_list = []
        # gt_folders = [x for x in sorted(os.listdir(self.gt_dir))]
        # for folder in gt_folders:
        #     folder_dir = join(self.gt_dir, folder)
        for img in sorted(os.listdir(gt_dir)):
            if is_image_file(img):
                self.gt_list.append(join(gt_dir, img))

        # Load blur list
        self.input_list = []
        # input_folders = [x for x in sorted(os.listdir(self.input_dir))]
        # # for folder in input_folders:
        # #     folder_dir = join(self.input_dir, folder)
        # for subfolder in subdirs:
        # sub_folder_dir = join(train_dir,str(subfolder))
        for img in sorted(os.listdir(train_dir)):
            if is_image_file(img):
                self.input_list.append(join(train_dir, img))

        self.seg_list = []
        for img in sorted(os.listdir(seg_dir)):
            if is_image_file(img):
                self.seg_list.append(join(seg_dir, img))

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, index):
        # Arrange frames
        folder_id = int(str(self.input_list[index])[-9:-7])
        num_input = len(self.input_list)
        num_gt = len(self.gt_list)
        if num_input != num_gt:
            raise ValueError("wrong dataset")

        if index == (len(self.input_list)-1):
            folder_id_next = None
            folder_id_next_next = None
        elif index == (len(self.input_list)-2):
            folder_id_next = int(str(self.input_list[index+1])[-9:-7])
            folder_id_next_next = None
        else:
            folder_id_next = int(str(self.input_list[index+1])[-9:-7])
            folder_id_next_next = int(str(self.input_list[index+2])[-9:-7])

        if index == 0:
            folder_id_previous = None
            folder_id_previous_previous = None
        elif index == 1:
            folder_id_previous = int(str(self.input_list[index-1])[-9:-7])
            folder_id_previous_previous = None
        else:
            folder_id_previous = int(str(self.input_list[index-1])[-9:-7])
            folder_id_previous_previous = int(str(self.input_list[index-2])[-9:-7])

        # Get image ID
        interval = 1
        if self.frames ==5:
            if folder_id != folder_id_previous:
                self.input_frame_lists = [index, index, index, index + interval, index + interval*2] # [n, n, n, n+1, n+2]
                self.seg_frame_lists = [index, index , index, index + interval, index + interval * 2]
                self.gt_frame_lists = [index , index, index + interval]
            elif folder_id == folder_id_previous and folder_id != folder_id_previous_previous:
                self.input_frame_lists = [index - interval, index - interval, index, index + interval, index + interval * 2] # [n-1, n-1, n, n+1, n+2]
                self.seg_frame_lists = [index - interval, index - interval, index, index + interval, index + interval * 2]
                self.gt_frame_lists = [index - interval, index, index + interval]
            elif folder_id != folder_id_next:
                self.input_frame_lists = [index - interval*2, index - interval, index, index, index] # [n-2, n-1, n, n, n]
                self.seg_frame_lists = [index - interval * 2, index - interval, index, index,index]
                self.gt_frame_lists = [index - interval, index, index ]
            elif folder_id == folder_id_next and folder_id != folder_id_next_next:
                self.input_frame_lists = [index - interval*2, index - interval, index, index + interval, index + interval] # [n-2, n-1, n, n+1, n+1]
                self.seg_frame_lists = [index - interval * 2, index - interval, index, index + interval,index + interval ]
                self.gt_frame_lists = [index - interval, index, index + interval ]
            else:
                self.input_frame_lists = [index - interval*2, index - interval, index, index + interval, index + interval*2]
                self.seg_frame_lists = [index - interval*2, index - interval, index, index + interval, index + interval*2]# [n-2, n-1, n, n+1, n+2]
                self.gt_frame_lists = [ index - interval, index, index + interval]
        elif self.frames == 3:
            if folder_id != folder_id_previous:
                self.frame_lists = [index, index, index + interval]
            elif folder_id != folder_id_next:
                self.frame_lists = [index - interval, index, index]
            elif index == (len(self.input_list)-1):
                self.frame_lists = [index - interval, index, index]
            else:
                self.frame_lists = [index - interval, index, index + interval]

        elif self.frames == 1:
            self.frame_lists = [index]

        else:
            raise ValueError("only support frames == 1 & 3 & 5")

        # Open images
        input_imgs = [Image.open((self.input_list[i])).convert('RGB') for i in self.input_frame_lists]
        gt_imgs = [Image.open((self.gt_list[i])).convert('RGB') for i in self.gt_frame_lists]#.convert('L')
        seg_imgs = [Image.open((self.seg_list[i])).convert('L') for i in self.seg_frame_lists]
        #seg_imgs = [Image.open((self.set_list[i])) for i in self.set_frame_lists]
        #seg_imgs =torch.zeros(3, 256, 256)
        # gt_img = Image.open(self.gt_list[index]).convert('RGB')

        # Set parameters
        left_top_w = random.randint(0, gt_imgs[0].size[0] - self.fineSize - 1)
        left_top_h = random.randint(0, gt_imgs[0].size[1] - self.fineSize - 1)
        random_flip_h = random.random()
        random_flip_v = random.random()
        random_rot = random.random()

        seg = []
        for seg_img in seg_imgs:
            input_patch = seg_img.crop(
                (left_top_w, left_top_h, left_top_w + self.fineSize, left_top_h + self.fineSize))
            #input_patch = input_patch.resize((self.cropSize, self.cropSize), Image.BICUBIC)
            if random_flip_h < 0.5:
                input_patch = ImageOps.flip(input_patch)
            if random_flip_v < 0.5:
                input_patch = ImageOps.mirror(input_patch)
            if random_rot < 0.5:
                input_patch = input_patch.rotate(180)
            input_patch = np.array(input_patch, dtype=np.float32) / 255
            # input_patch = input_patch.transpose((2, 0, 1))
            input_patch = torch.from_numpy(input_patch.copy()).float()
            input_patch = torch.unsqueeze(input_patch, 0)
            seg.append(input_patch)
        seg = torch.cat(seg, 0)
        seg = np.array(seg, dtype=np.float32)


        gts = []
        for gt_img in gt_imgs:
            input_patch = gt_img.crop(
                (left_top_w, left_top_h, left_top_w + self.fineSize, left_top_h + self.fineSize))
            #input_patch = input_patch.resize((self.cropSize, self.cropSize), Image.BICUBIC)
            if random_flip_h < 0.5:
                input_patch = ImageOps.flip(input_patch)
            if random_flip_v < 0.5:
                input_patch = ImageOps.mirror(input_patch)
            if random_rot < 0.5:
                input_patch = input_patch.rotate(180)
            input_patch = np.array(input_patch, dtype=np.float32) / 255
            #input_patch = input_patch.transpose((2, 0, 1))
            input_patch = torch.from_numpy(input_patch.copy()).float()
            input_patch = torch.unsqueeze(input_patch, 0)
            gts.append(input_patch)

        gts = torch.cat(gts, 0)
        gts = np.array(gts, dtype=np.float32)
        # Processing input images
        inputs = []
        for input_img in input_imgs:
            input_patch = input_img.crop(
                (left_top_w, left_top_h, left_top_w + self.fineSize, left_top_h + self.fineSize))
            #input_patch = input_patch.resize((self.cropSize, self.cropSize), Image.BICUBIC)
            if random_flip_h < 0.5:
                input_patch = ImageOps.flip(input_patch)
            if random_flip_v < 0.5:
                input_patch = ImageOps.mirror(input_patch)
            if random_rot < 0.5:
                input_patch = input_patch.rotate(180)
            input_patch = np.array(input_patch, dtype=np.float32) / 255
            input_patch = input_patch.transpose((2, 0, 1))
            input_patch = torch.from_numpy(input_patch.copy()).float()
            inputs.append(input_patch)
        inputs = torch.cat(inputs, 0)
        inputs = np.array(inputs, dtype=np.float32)



        # Processing gt image
        # gt_patch = gt_img.crop(
        #     (left_top_w, left_top_h, left_top_w + self.fineSize, left_top_h + self.fineSize))
        # gt_patch = gt_patch.resize((self.cropSize, self.cropSize), Image.BICUBIC)
        # if random_flip_h < 0.5:
        #     gt_patch = ImageOps.flip(gt_patch)
        # if random_flip_v < 0.5:
        #     gt_patch = ImageOps.mirror(gt_patch)
        # if random_rot < 0.5:
        #     gt_patch = gt_patch.rotate(180)
        # gt_patch = np.array(gt_patch, dtype=np.float32) / 255
        # gt_patch = gt_patch.transpose((2, 0, 1))

        return inputs.copy(), \
               gts.copy(), \
                seg.copy()


# ---------------------
#  Dataset_validation
# ---------------------
class Video_dataset_valid(data.Dataset):
    def __init__(self, root_dir, frames=1):
        super(Video_dataset_valid, self).__init__()
        self.gt_dir = join(root_dir, 'test/gt')
        self.input_dir = join(root_dir, 'test/result')
        self.frames = frames

        # Load gt list
        self.gt_list = []
        gt_folders = [x for x in sorted(os.listdir(self.gt_dir))]
        for folder in gt_folders:
            folder_dir = join(self.gt_dir, folder)
            for img in sorted(os.listdir(folder_dir)):
                if is_image_file(img):
                    self.gt_list.append(join(folder_dir, img))

        # Load blur list
        self.input_list = []
        input_folders = [x for x in sorted(os.listdir(self.input_dir))]
        for folder in input_folders:
            folder_dir = join(self.input_dir, folder)
            for img in sorted(os.listdir(folder_dir)):
                if is_image_file(img):
                    self.input_list.append(join(folder_dir, img))

    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, index):
        # Arrange frames
        folder_id = int(str(self.input_list[index])[-9:-7])
        num_input = len(self.input_list)
        num_gt = len(self.gt_list)
        if num_input != num_gt:
            raise ValueError("wrong dataset")

        if index == (len(self.input_list)-1):
            folder_id_next = None
            folder_id_next_next = None
        elif index == (len(self.input_list)-2):
            folder_id_next = int(str(self.input_list[index+1])[-9:-7])
            folder_id_next_next = None
        else:
            folder_id_next = int(str(self.input_list[index+1])[-9:-7])
            folder_id_next_next = int(str(self.input_list[index+2])[-9:-7])

        if index == 0:
            folder_id_previous = None
            folder_id_previous_previous = None
        elif index == 1:
            folder_id_previous = int(str(self.input_list[index-1])[-9:-7])
            folder_id_previous_previous = None
        else:
            folder_id_previous = int(str(self.input_list[index-1])[-9:-7])
            folder_id_previous_previous = int(str(self.input_list[index-2])[-9:-7])

        # Get image ID
        interval = 1
        if self.frames == 5:
            if folder_id != folder_id_previous:
                self.frame_lists = [index, index, index, index + interval, index + interval * 2]  # [n, n, n, n+1, n+2]

            elif folder_id == folder_id_previous and folder_id != folder_id_previous_previous:
                self.frame_lists = [index - interval, index - interval, index, index + interval,
                                    index + interval * 2]  # [n-1, n-1, n, n+1, n+2]

            elif folder_id != folder_id_next:
                self.frame_lists = [index - interval * 2, index - interval, index, index, index]  # [n-2, n-1, n, n, n]

            elif folder_id == folder_id_next and folder_id != folder_id_next_next:
                self.frame_lists = [index - interval * 2, index - interval, index, index + interval,
                                    index + interval]  # [n-2, n-1, n, n+1, n+1]

            else:
                self.frame_lists = [index - interval * 2, index - interval, index, index + interval,
                                    index + interval * 2]  # [n-2, n-1, n, n+1, n+2]

        elif self.frames == 3:
            if folder_id != folder_id_previous:
                self.frame_lists = [index, index, index + interval]
            elif folder_id != folder_id_next:
                self.frame_lists = [index - interval, index, index]
            elif index == (len(self.input_list) - 1):
                self.frame_lists = [index - interval, index, index]
            else:
                self.frame_lists = [index - interval, index, index + interval]

        elif self.frames == 1:
            self.frame_lists = [index]

        else:
            raise ValueError("only support frames == 1 & 3 & 5")

        # Open images
        input_imgs = [Image.open((self.input_list[i])).convert('RGB') for i in self.frame_lists]
        gt_img = Image.open(self.gt_list[index])

        # Set parameters
        left_top_w = 0
        left_top_h = 0
        right_top_w = gt_img.size[0]//4 * 4
        right_top_h = gt_img.size[1]//4 * 4

        # Processing input images
        inputs = []
        for input_img in input_imgs:
            input_img = input_img.crop(
                (left_top_w, left_top_h, right_top_w, right_top_h))
            a = np.array(input_img, dtype=np.float32)
            input_img = np.array(input_img, dtype=np.float32) / 255
            input_img = input_img.transpose((2, 0, 1))
            a = a.transpose((2, 0, 1))
            input_img = torch.from_numpy(input_img.copy()).float()
            a = torch.from_numpy(a.copy()).float()
            inputs.append(input_img)
        inputs = torch.cat(inputs, 0)
        inputs = np.array(inputs, dtype=np.float32)

        # Processing gt image
        gt_img = gt_img.crop(
            (left_top_w, left_top_h, right_top_w, right_top_h))
        b = np.array(gt_img, dtype=np.float32)
        gt_img = np.array(gt_img, dtype=np.float32) / 255
        gt_img = gt_img.transpose((2, 0, 1))
        b = b.transpose((2, 0, 1))

        return inputs.copy(), \
               gt_img.copy(), \
                a, b


# ---------------------
#    Dataset_test
# ---------------------
class Video_dataset_test(data.Dataset):
    def __init__(self, input_dir_test,gt_dir_test,seg_dir_test,cropSize, frames=5):
        super(Video_dataset_test, self).__init__()
        self.gt_dir = gt_dir_test
        self.input_dir = input_dir_test
        self.seg_dir = seg_dir_test
        self.frames = frames
        self.cropSize = cropSize
        self.fineSize = self.cropSize
        # Load gt list
        self.gt_list = []
        for img in sorted(os.listdir(self.gt_dir)):
            if is_image_file(img):
                self.gt_list.append(join(self.gt_dir, img))

        # Load input list
        self.input_list = []
        for img in sorted(os.listdir(self.input_dir)):
            if is_image_file(img):
                self.input_list.append(join(self.input_dir, img))

        self.seg_list = []
        for img in sorted(os.listdir(self.seg_dir)):
            if is_image_file(img):
                self.seg_list.append(join(self.seg_dir, img))



    def __len__(self):
        return len(self.gt_list)

    def __getitem__(self, index):
        # Arrange frames
        folder_id = int(str(self.input_list[index])[-9:-7])
        num_input = len(self.input_list)
        num_gt = len(self.gt_list)
        if num_input != num_gt:
            raise ValueError("wrong dataset")

        if index == (len(self.input_list)-1):
            folder_id_next = None
            folder_id_next_next = None
        elif index == (len(self.input_list)-2):
            folder_id_next = int(str(self.input_list[index+1])[-9:-7])
            folder_id_next_next = None
        else:
            folder_id_next = int(str(self.input_list[index+1])[-9:-7])
            folder_id_next_next = int(str(self.input_list[index+2])[-9:-7])

        if index == 0:
            folder_id_previous = None
            folder_id_previous_previous = None
        elif index == 1:
            folder_id_previous = int(str(self.input_list[index-1])[-9:-7])
            folder_id_previous_previous = None
        else:
            folder_id_previous = int(str(self.input_list[index-1])[-9:-7])
            folder_id_previous_previous = int(str(self.input_list[index-2])[-9:-7])

        # Get image ID

        interval=1
        if self.frames ==5:
            if folder_id != folder_id_previous:
                self.input_frame_lists = [index, index, index, index + interval, index + interval*2] # [n, n, n, n+1, n+2]
                self.seg_frame_lists = [index, index , index, index + interval, index + interval * 2]
                self.gt_frame_lists = [index , index, index + interval]
            elif folder_id == folder_id_previous and folder_id != folder_id_previous_previous:
                self.input_frame_lists = [index - interval, index - interval, index, index + interval, index + interval * 2] # [n-1, n-1, n, n+1, n+2]
                self.seg_frame_lists = [index - interval, index - interval, index, index + interval, index + interval * 2]
                self.gt_frame_lists = [index - interval, index, index + interval]
            elif folder_id != folder_id_next:
                self.input_frame_lists = [index - interval*2, index - interval, index, index, index] # [n-2, n-1, n, n, n]
                self.seg_frame_lists = [index - interval * 2, index - interval, index, index,index]
                self.gt_frame_lists = [index - interval, index, index ]
            elif folder_id == folder_id_next and folder_id != folder_id_next_next:
                self.input_frame_lists = [index - interval*2, index - interval, index, index + interval, index + interval] # [n-2, n-1, n, n+1, n+1]
                self.seg_frame_lists = [index - interval * 2, index - interval, index, index + interval,index + interval ]
                self.gt_frame_lists = [index - interval, index, index + interval ]
            else:
                self.input_frame_lists = [index - interval*2, index - interval, index, index + interval, index + interval*2]
                self.seg_frame_lists = [index - interval*2, index - interval, index, index + interval, index + interval*2]# [n-2, n-1, n, n+1, n+2]
                self.gt_frame_lists = [ index - interval, index, index + interval]
        elif self.frames == 3:
            if folder_id != folder_id_previous:
                self.frame_lists = [index, index, index + interval]
            elif folder_id != folder_id_next:
                self.frame_lists = [index - interval, index, index]
            elif index == (len(self.input_list)-1):
                self.frame_lists = [index - interval, index, index]
            else:
                self.frame_lists = [index - interval, index, index + interval]

        elif self.frames == 1:
            self.frame_lists = [index]

        else:
            raise ValueError("only support frames == 1 & 3 & 5")

        # Open images
        input_imgs = [Image.open((self.input_list[i])).convert('RGB') for i in self.input_frame_lists]
        #gt_img = Image.open(self.gt_list[index]).convert('RGB')
        gt_img = [Image.open((self.gt_list[i])) .convert('RGB') for i in self.gt_frame_lists]
        seg_imgs = [Image.open((self.seg_list[i])) .convert('L') for i in self.seg_frame_lists]


        # Set parameters
        left_top_w = random.randint(0, gt_img[0].size[0] - self.fineSize - 1)
        left_top_h = random.randint(0, gt_img[0].size[1] - self.fineSize - 1)
        random_flip_h = random.random()
        random_flip_v = random.random()
        random_rot = random.random()

        # Processing input images
        inputs = []
        for input_img in input_imgs:
            input_img = np.array(input_img, dtype=np.float32) / 255
            # plt.imshow(input_img)
            # plt.show()
            input_img = input_img.transpose((2, 0, 1))
            input_img = torch.from_numpy(input_img.copy()).float()

        #     a,b,c=input_img.size()
        #     input_img=input_img.resize(a,b*0.6,c*0.6)
        #     # size=(a,b*0.6,c*0.6)
        #     # input_img=cv2.resize(input_img, size, interpolation=cv2.INTER_AREA)
        #     vl.show_image(input_img)
            inputs.append(input_img)
        inputs = torch.cat(inputs, 0)#.resize(a,,b*0.6,c*0.6)
        inputs = np.array(inputs, dtype=np.float32)
        # for input_img in input_imgs:
        #     input_patch = input_img.crop(
        #         (left_top_w, left_top_h, left_top_w + self.fineSize, left_top_h + self.fineSize))
        #     #input_patch = input_patch.resize((self.cropSize, self.cropSize), Image.BICUBIC)
        #     if random_flip_h < 0.5:
        #         input_patch = ImageOps.flip(input_patch)
        #     if random_flip_v < 0.5:
        #         input_patch = ImageOps.mirror(input_patch)
        #     if random_rot < 0.5:
        #         input_patch = input_patch.rotate(180)
        #     input_patch = np.array(input_patch, dtype=np.float32) / 255
        #     input_patch = input_patch.transpose((2, 0, 1))
        #     input_patch = torch.from_numpy(input_patch.copy()).float()
        #     inputs.append(input_patch)
        # inputs = torch.cat(inputs, 0)
        # inputs = np.array(inputs, dtype=np.float32)

        seg = []
        # for seg_img in seg_imgs:
        #     input_patch = seg_img.crop(
        #         (left_top_w, left_top_h, left_top_w + self.fineSize, left_top_h + self.fineSize))
        #     #input_patch = input_patch.resize((self.cropSize, self.cropSize), Image.BICUBIC)
        #     if random_flip_h < 0.5:
        #         input_patch = ImageOps.flip(input_patch)
        #     if random_flip_v < 0.5:
        #         input_patch = ImageOps.mirror(input_patch)
        #     if random_rot < 0.5:
        #         input_patch = input_patch.rotate(180)
        #     input_patch = np.array(input_patch, dtype=np.float32) / 255
        #     # input_patch = input_patch.transpose((2, 0, 1))
        #     input_patch = torch.from_numpy(input_patch.copy()).float()
        #     input_patch = torch.unsqueeze(input_patch, 0)
        #     #input_patch = torch.from_numpy(input_patch.copy()).float()
        #     seg.append(input_patch)
        for seg_img in seg_imgs:
            seg_img = np.array(seg_img, dtype=np.float32) / 255
            seg_img = np.expand_dims(seg_img, 2)
            seg_img = seg_img.transpose((2, 0, 1))
            seg_img = torch.from_numpy(seg_img.copy()).float()
            seg.append(seg_img)

        seg = torch.cat(seg, 0)

        seg = np.array(seg, dtype=np.float32)

        # Processing gt image

        gts = []
        # for gt_img in gt_img:
        #     input_patch = gt_img.crop(
        #         (left_top_w, left_top_h, left_top_w + self.fineSize, left_top_h + self.fineSize))
        #     input_patch = input_patch.resize((self.cropSize, self.cropSize), Image.BICUBIC)
        #     if random_flip_h < 0.5:
        #         input_patch = ImageOps.flip(input_patch)
        #     if random_flip_v < 0.5:
        #         input_patch = ImageOps.mirror(input_patch)
        #     if random_rot < 0.5:
        #         input_patch = input_patch.rotate(180)
        #     input_patch = np.array(input_patch, dtype=np.float32) / 255
        #     input_patch = input_patch.transpose((2, 0, 1))
        #     input_patch = torch.from_numpy(input_patch.copy()).float()
        #     #input_patch = torch.unsqueeze(input_patch,0)
        #
        #     gts.append(input_patch)
        for gt_img in gt_img:
            gt_img = np.array(gt_img, dtype=np.float32) / 255
            gt_img = gt_img.transpose((2, 0, 1))
            gt_img = torch.from_numpy(gt_img.copy()).float()
            gts.append(gt_img)
        gts = torch.cat(gts, 0)#.resize(3,256,256)
        gts = np.array(gts, dtype=np.float32)





        return inputs.copy(), \
               gts.copy(), \
               seg.copy(), \
               self.input_list[index]


