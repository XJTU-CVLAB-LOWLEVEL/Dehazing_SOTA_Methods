# x = input()
# y = input()
#
# x1 = x.split(" ")[0]
# x2 = int(x.split(' ')[1])
#
#
# def is_symmetrical(str):
#     length = len(str)
#     count =  0
#     for i in range(length // 2):
#         if str[i] == str[length - i - 1]:
#             count = count + 1
#     return count
#
# num = is_symmetrical(y)
#
# y1 = y[num:]
#
# for i in range(x2 - 1):
#     y += y1
#
# print(y)

'''

5 10
3 9 5 7 6
'''
import os

folder_path = r'/home/lry/video_dehazing/datasets/VideoHazy_v2_re/Test/Results1'
folders = os.listdir(folder_path)
for folder in folders:
    folder_name = os.path.join(folder_path, folder)
    img_names = os.listdir(folder_name)
    for img_name in img_names:
        old = os.path.join(folder_name, img_name)
        #new_name=str('0')+img_name[3:5]+'.jpg'
        new_name = img_name[0:5]+'_VDHNet.jpg'
        new = os.path.join(folder_name, new_name)
        os.rename(old, new)

