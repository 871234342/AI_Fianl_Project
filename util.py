import torch
import pandas as pd
import os
import copy
import numpy as np
from skimage import io
from PIL import Image
from torch.utils.data import Dataset
import time


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
