import imgaug.augmenters as iaa
import cv2
from random import randrange
import numpy as np
import glob


seq = iaa.Sequential([
    #iaa.imgcorruptlike.Contrast(severity=1),
    #iaa.imgcorruptlike.Fog(severity=5),
    iaa.Fog(),
])

#aug = iaa.color.MultiplyBrightness((0.5, 1.5))

files = glob.glob('./data/train/ntire/*.png')
for f in files:
    fname = f.split('/')[-1].split('.')[0]
    imglist=[]
    img = cv2.imread(f)
    for i in range(400):
        imglist.append(img)
    img_aug = seq.augment_images(imglist)
    #print ('succ')
    crop_width = 320
    crop_height = 320
    height, width,_ = img.shape
    result = []
    weight_map = np.zeros([320, 320, 3])

    for h in range(320):
        for w in range(320):
            weight_map[h, w, :] = 1 - np.abs((h - 160) * (w - 160) * (h - 160) * (w - 160) * (h - 160) * (w - 160)) / (160 * 160 * 160 * 160 * 160 * 160)
    #imglist = []
    for img in img_aug:
        x, y = randrange(0, width - crop_width + 1), randrange(0, height - crop_height + 1)
        crop_img = img[y:y+crop_height, x:x+crop_width,:]
        #print (crop_img.shape)
        imglist=[]
        imglist.append(crop_img)
        img_aug_new = seq.augment_images(imglist)
        #img[y:y+crop_height, x:x+crop_width,:] = 255
        img[y:y+crop_height, x:x+crop_width,:] = img_aug_new[0]
        result.append(img)

    for i in range(40):
        cv2.imwrite('./data/train/ntire_hom/{}_aug_{}.png'.format(fname, str(i)), img_aug[i])
        print('save!')
#cv2.imshow('111', img_aug[0])
#cv2.waitKey(0)
