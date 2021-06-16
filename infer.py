import torch
import torch.nn as nn
from torchvision import transforms, models
import pandas as pd
import os
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

from util import DataNoLabel, Rotate90

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_cuda = torch.cuda.is_available()
SERIAL_NUMBER = 3
NUM_WORKER = 8
BATCH_SIZE = 8
MODEL_WEIGHT = "best_weights.pth"



img_ids = []
labels = []
test_folder = 'test'

# TTA
tta1 = transforms.Compose([
    transforms.RandomVerticalFlip(p=1),
    Rotate90()
])
tta2 = transforms.Compose([
    transforms.RandomHorizontalFlip(p=1),
    Rotate90()
])
tta3 = transforms.Compose([
    Rotate90([0, 0.3, 0.4, 0.3])
])

transform = transforms.Compose([
    transforms.CenterCrop((48, 48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.Lambda(lambda image: torch.stack([
        tta1(image), tta2(image), tta3(image), image
    ]))
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

# Inference
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

result_path = 'submission' + str(SERIAL_NUMBER) + 'csv'
df = pd.DataFrame(data={'id': img_ids, 'label': labels})
df.to_csv(result_path, index=False)