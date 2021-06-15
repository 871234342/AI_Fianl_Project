import pandas as pd
import numpy as np
from PIL import Image
from util import DataWithLabel


df = pd.read_csv('train_labels.csv')

dataset = DataWithLabel("train_labels.csv", "train/")

sample = dataset[1]

print(sample)
print(sample[0].shape)