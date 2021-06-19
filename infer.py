import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import albumentations as A

from util import DataNoLabel, Rotate90

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
use_cuda = torch.cuda.is_available()
SERIAL_NUMBER = 8
NUM_WORKER = 8
BATCH_SIZE = 8
MODEL_WEIGHT = "best_weights_" +str(SERIAL_NUMBER) + ".pth"


img_ids = []
labels = []
test_folder = 'test'

# TTA
tta1 = A.Compose([
    A.HorizontalFlip(p=1)
])
tta2 = A.Compose([
    A.VerticalFlip(p=1)
])
tta3 = A.Compose([
    A.RandomRotate90(p=1)
])

test_transforms = [tta1, tta2, tta3]

def tta(image, **kwargs):
    imgs = [image]
    for test_transform in test_transforms:
        imgs.append(test_transform(image=image)['image'])
    imgs = np.array(imgs)
    return imgs

transform = A.Compose([
    A.HueSaturationValue(),
    A.Normalize(p=1),
    A.Lambda(name='tta', image=tta)
])

infer_set = DataNoLabel(test_folder, transform)
infer_loader = DataLoader(infer_set, batch_size=BATCH_SIZE,
                          num_workers=8, shuffle=False)

# Construct model
model = models.resnext50_32x4d()
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Linear(num_features, 1),
    nn.Sigmoid()
)
model.load_state_dict(torch.load(MODEL_WEIGHT)['model_state_dict'])
print("Model loaded.")

# Use GPU
device = torch.device('cuda' if use_cuda else 'cpu')
print("number of GPU(s): {}".format(torch.cuda.device_count()))
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
print(device)
model = model.to(device)
model.eval()
'''
# Inference w/o tta
with torch.no_grad():
    for inputs, ids in infer_loader:
        batch_size = inputs.size()[0]
        inputs = inputs.to(device)
        outputs = model(inputs)
        pred = outputs > 0.5
        for i in range(batch_size):
            img_ids.append(ids[i])
            labels.append(int(pred[i].item()))
'''

# Inference with tta
with torch.no_grad():
    for inputs, ids in infer_loader:
        batch_size, n_tta, c, h, w = inputs.size()
        inputs = inputs.view(-1, c, h, w)
        inputs = inputs.to(device)
        outputs = model(inputs)
        outputs = outputs.view(batch_size, n_tta, -1)
        outputs = outputs.mean(1)
        pred = outputs > 0.5
        for i in range(batch_size):
            img_ids.append(ids[i])
            labels.append(int(pred[i].item()))


result_path = 'submission_' + str(SERIAL_NUMBER) + '.csv'
df = pd.DataFrame(data={'id': img_ids, 'label': labels})
df.to_csv(result_path, index=False)
