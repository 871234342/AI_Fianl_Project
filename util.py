import torch
import pandas as pd
import os
import copy
import random
import numpy as np
from skimage import io
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import time


class Rotate90:
    '''
    Rotate an image by 90, 180 or 270 degree
    '''
    def __init__(self, p=None):
        '''
        :param p: List of prob. of each rotation
                  e.g. [0.1, 0.2, 0.3, 0.4]
        '''
        if p:
            self.p = [0, 0, 0, 0]
            for i in range(4):
                for j in range(i + 1):
                    self.p[i] += p[j]
        else:
            self.p = [0.25, 0.5, 0.75, 1]

    def __call__(self, x):
        choice = random.uniform(0, 1)
        if choice < self.p[0]:
            return x
        elif choice < self.p[1]:
            return TF.rotate(x, 90)
        elif choice < self.p[2]:
            return TF.rotate(x, 180)
        elif choice < self.p[3]:
            return TF.rotate(x, 270)

            
class DataWithLabel(Dataset):
    def __init__(self, annot, img_dir, transforms=None):
        '''
        :param annot: csv annotation file
        :param img_dir: folder path containing images
        '''
        df = pd.read_csv(annot)
        self.imgs = []
        self.labels = []
        self.img_dir = img_dir
        self.transforms = transforms

        for i in range(len(df)):
            self.imgs.append(df.iloc[i, 0])
            self.labels.append(df.iloc[i, 1])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img_path = os.path.join(self.img_dir, self.imgs[item] + ".tif")
        img = Image.open(img_path)
        img = np.array(img)
        label = self.labels[item]

        if self.transforms:
            img = self.transforms(img)

        return (img, label)
