import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, models
import pandas as pd
import os
import time
import random
import numpy as np
import copy
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold
import albumentations as A

from util import DataWithLabel, Rotate90

SERIAL_NUMBER = 10
LOG_FILE = "log" + str(SERIAL_NUMBER) + ".txt"
# Hyper-parameters
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
BATCH_SIZE = 1024
WEIGHT_DECAY = 0
LEARNING_RATE = 0.01
NUM_WORKER = 8
CHECKPOINT = None
NUM_EPOCH = 30
CHECKPOINT_INTERVAL = 2
LR_STEP = 12
VAL_SIZE = 0.1
FOLDS = 10

print("EXPERIMENT {}".format(SERIAL_NUMBER))

# data prerocess
df = pd.read_csv("train_labels.csv")
data_ids = df['id'].tolist()
data_labels = df['label'].tolist()

# use GPU
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print("Number of GPU(s) being used: {}".format(torch.cuda.device_count()))
print("Using ", device)

# data augmentation
transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Flip(p=0.5),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.IAASharpen(),
        A.IAAEmboss(),
        A.RandomBrightnessContrast(),
        A.ImageCompression(),
        A.Blur(),
        A.GaussNoise()
    ], p=0.5),
    A.HueSaturationValue(p=0.5),
    A.Normalize(p=1)
])

# keep experiment setting
with open(LOG_FILE, mode='w') as file:
    file.write("Test case: " + str(SERIAL_NUMBER))
    file.write(" Batch size: " + str(BATCH_SIZE))
    file.write(" Number of epochs: " + str(NUM_EPOCH))
    file.write(" Learning rate: " + str(LEARNING_RATE))
    file.write(" Weight decay: " + str(WEIGHT_DECAY))
    file.write(" LR_step: " + str(LR_STEP) + "\n")
    file.write(str(transform) + "\n")

# K Fold - start training
kf = KFold(n_splits=FOLDS)
for i, indexes in enumerate(kf.split(data_ids)):
    print("training fold ", i)
    with open(LOG_FILE, mode='a+') as file:
        file.write("Fold " + str(i) +"\n")

    data_ids = np.array(data_ids)
    data_labels = np.array(data_labels)
    x_train, x_val = data_ids[indexes[0]], data_ids[indexes[1]]
    y_train, y_val = data_labels[indexes[0]], data_labels[indexes[1]]

    # data set construction
    train_set = DataWithLabel(x_train, y_train, "train/", transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                              num_workers=NUM_WORKER, shuffle=True)
    val_set = DataWithLabel(x_val, y_val, "train/", transform)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE,
                            num_workers=NUM_WORKER, shuffle=True)
    loaders = {'train': train_loader, 'val': val_loader}

    # model construction
    model = models.resnext50_32x4d(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 1),
        nn.Sigmoid()
    )

    # model to device
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Criterion, optimizer, scheduler
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE,
                          weight_decay=WEIGHT_DECAY)
    scheduler = lr_scheduler.StepLR(optimizer, LR_STEP, 0.1)

    # Train model
    best_model_weights = copy.deepcopy(model.state_dict())
    best_accuracy = 0.0
    best_epoch = 0
    training_time = 0

    # Iterate through all epochs
    for epoch in range(NUM_EPOCH):
        print("Epoch {}/{}".format(epoch + 1, NUM_EPOCH))
        train_accuracy = 0
        val_accuracy = 0
        train_loss= 0.0
        val_loss = 0.0

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            training_start = time.time()
            epoch_loss = 0.0
            epoch_correct = 0

            # Iterate through all batches
            for inputs, labels in loaders[phase]:
                inputs = inputs.to(device)
                labels = labels.type(torch.FloatTensor).to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = torch.squeeze(model(inputs))
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_loss += loss.item()
                pred = outputs > 0.5
                epoch_correct += torch.sum(pred == labels)

            scheduler.step()

            # print result on screen
            if phase == 'train':
                train_loss = epoch_loss / len(train_set)
                train_accuracy += float(epoch_correct) / len(train_set)
                print("{} Loss: {:.8f}, Acc: {:.4f} in {}s".format(
                    phase, train_loss, train_accuracy,
                    time.time() - training_start
                ))
            else:
                val_loss = epoch_loss / len(val_set)
                val_accuracy += float(epoch_correct) / len(val_set)
                print("{} Loss: {:.8f}, Acc: {:.4f}".format(
                    phase, val_loss, val_accuracy,
                    time.time() - training_start
                ))
                # keep best model
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                    best_model_weights = copy.deepcopy(model.state_dict())
                    best_epoch = epoch + 1

            training_time += (time.time() - training_start)

        # keep results
        if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
            with open(LOG_FILE, mode="a+") as file:
                file.write("{}: train loss: {:.8f} acc: {:.4f}, val loss: {:.8f} acc:{:.4f}\n".format(
                    str(epoch + 1).zfill(3), train_loss,
                    train_accuracy, val_loss, val_accuracy
                ))
            PATH = 'checkpoint_' + str(epoch + 1) + '.pth'
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'schedular_State_dict': scheduler.state_dict()
            }, PATH)

    print("Best accuracy: {:.4f} at epoch {}, total time: {}s".format(
        best_accuracy, best_epoch, training_time
    ))
    with open(LOG_FILE, mode='a+') as file:
        file.write("Best accuracy: {:.4f} at epoch{}\n".format(
            best_accuracy, best_epoch
        ))
    PATH = "fold/best_weights_" + str(SERIAL_NUMBER) + "_" + str(i) + '.pth'
    model.load_state_dict(best_model_weights)
    torch.save({
        'model_state_dict': model.state_dict()
    }, PATH)
