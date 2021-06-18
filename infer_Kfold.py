import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import albumentations as A

from util import DataNoLabel

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
use_cuda = torch.cuda.is_available()
SERIAL_NUMBER = 10
NUM_WORKER = 8
BATCH_SIZE = 8
FOLD = 10
PARALLEL = True

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

# Use GPU
device = torch.device('cuda' if use_cuda else 'cpu')
print("number of GPU(s): {}".format(torch.cuda.device_count()))
if PARALLEL:
    model = nn.DataParallel(model)
print(device)

for i in range(FOLD):
    # Load weight
    weight_path = "fold/best_weights_" + str(SERIAL_NUMBER) +"_"+str(i)+'.pth'
    model.load_state_dict(torch.load(weight_path)['model_state_dict'])
    print(i, "-th Model loaded.")
    model = model.to(device)
    model.eval()

    # Inference with tta
    with torch.no_grad():
        for inputs, ids in infer_loader:
            batch_size, n_tta, c, h, w = inputs.size()
            inputs = inputs.view(-1, c, h, w)
            inputs = inputs.to(device)
            outputs = model(inputs)
            outputs = outputs.view(batch_size, n_tta, -1)
            outputs = outputs.mean(1)

            #print("i: ", i)

            if i == 0:  # first fold
                for j in range(batch_size):
                    img_ids.append(ids[j])
                    labels.append(outputs[j].item())
            else:
                for j in range(batch_size):
                    index = img_ids.index(ids[j])
                    labels[index] += outputs[j].item()

# average all outputs
for i in range(len(labels)):
    labels[i] /= FOLD

# predict
pred = []
for score in labels:
    pred.append(int(score > 0.5))

result_path = 'submission_' + str(SERIAL_NUMBER) + '.csv'
df = pd.DataFrame(data={'id': img_ids, 'label': pred})
df.to_csv(result_path, index=False)
