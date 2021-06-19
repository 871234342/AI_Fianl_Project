import torch
import pandas as pd
import os
import copy
import random
import numpy as np
from skimage import io
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import time
import albumentations as A


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
    def __init__(self, imgs, labels, img_dir, transforms=None):
        '''
        :param imgs: list of image ids
        :param labels: list of labels
        :param img_dir: directory of images
        :param transforms: albumentations transform
        '''
        self.imgs = imgs
        self.labels = labels
        self.img_dir = img_dir
        self.transforms = transforms

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
            img = self.transforms(image = img)['image']

        img = transforms.ToTensor()(img)
        return img, label


class DataNoLabel(Dataset):
    def __init__(self, img_dir, transform=None):
        '''
        :param img_dir: directory of testing images
        :param transform: albumentations transform
        '''
        self.img_dir = img_dir
        self.transform = transform
        self.img_ids = []
        for filename in os.listdir(img_dir):
            self.img_ids.append(filename)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        img_path = os.path.join(self.img_dir, self.img_ids[item])
        img = Image.open(img_path)
        img = np.array(img)
        if self.transform:
            img = self.transform(image = img)['image']
        if img.ndim == 4:
            new_img = []
            for i in range(len(img)):
                new_img.append(transforms.ToTensor()(img[i]))
            img = torch.stack(new_img)
        else:
            img = transforms.ToTensor()(img)
        img_id = self.img_ids[item].split('.')[0]

        return img, img_id
