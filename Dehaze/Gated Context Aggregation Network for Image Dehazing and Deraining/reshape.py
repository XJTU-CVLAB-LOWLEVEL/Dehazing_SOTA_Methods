import cv2
import os

# path = '/home/hang/PytorchProjects/ggca-net/examples/train_resize/
path = '/home/hang/PytorchProjects/ggca-net/examples/train/haze'
names = os.listdir(path)
for name in names:
    img = os.path.join(path, name)
    new_path = '/home/hang/PytorchProjects/ggca-net/examples/train_resize/haze'
    img_input = cv2.imread(img)
    w, h = img_input.shape[:2]
    w, h = int(w/2), int(h/2)
    new_img = cv2.resize(img_input, (h, w))
    new_name = os.path.join(new_path, name)
    cv2.imwrite(new_name, new_img)