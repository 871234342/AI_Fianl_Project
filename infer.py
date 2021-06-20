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
from ResNext import se_resnext101


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
use_cuda = torch.cuda.is_available()
SERIAL_NUMBER = 11
NUM_WORKER = 8
BATCH_SIZE = 16
FOLD = 5
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
    A.Resize(224, 224),
    A.HueSaturationValue(),
    A.Normalize(p=1),
    A.Lambda(name='tta', image=tta)
])

infer_set = DataNoLabel(test_folder, transform)
infer_loader = DataLoader(infer_set, batch_size=BATCH_SIZE,
                          num_workers=8, shuffle=False)

# Construct model
model = se_resnext101(1000, pretrained='imagenet')
in_features = model.last_linear.in_features
model.last_linear = nn.Sequential(
    nn.Linear(in_features, 1),
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
    weight_path = "best_weights_" + str(SERIAL_NUMBER) +"_"+str(2)+'.pth'
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

            if i == 0:  # first fold
                for j in range(batch_size):
                    img_ids.append(ids[j])
                    labels.append(outputs[j].item())
            else:
                for j in range(batch_size):
                    index = img_ids.index(ids[j])
                    labels[index] += outputs[j].item()

    '''
    # Inference without tta
    with torch.no_grad():
        for inputs, ids in infer_loader:
            batch_size, c, h, w = inputs.size()
            inputs = inputs.to(device)
            outputs = model(inputs)

            if i == 0:  # first fold
                for j in range(batch_size):
                    img_ids.append(ids[j])
                    labels.append(outputs[j].item())
            else:
                for j in range(batch_size):
                    index = img_ids.index(ids[j])
                    labels[index] += outputs[j].item()
    '''
    #break       # no ensemble


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
