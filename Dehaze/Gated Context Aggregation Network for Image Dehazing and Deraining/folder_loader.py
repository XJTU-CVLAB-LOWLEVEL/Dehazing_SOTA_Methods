import io
import os
import utils
import struct
from PIL import Image

class FolderLoader(object):
    def __init__(self, fold_path):
        super(FolderLoader, self).__init__()
        self.fold_path = fold_path
        self.img_paths = utils.make_dataset(self.fold_path)
        self.img_names = [os.path.basename(x) for x in self.img_paths]

    def __getitem__(self, index):
        img = Image.open(self.img_paths[index])#.convert('RGB')
        return self.img_names[index], img

    def __len__(self):
        return len(self.img_names)
