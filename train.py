import os
import gc
import random
from pathlib import Path

import numpy as np
import pandas as pd

from PIL import Image
from tqdm.notebook import tqdm
from sklearn.model_selection import StratifiedKFold

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
import torch.nn.functional as F

import timm

from Dataset import *
from losses import *
from models import *
from parameters import args



def train(args, model, train_loader, criterion, optimizer, scheduler, epoch):
    model.train()
    running_loss = 0.

    for batch_idx, (anchor, da_anchor, label) in enumerate(train_loader):

        images = torch.cat([anchor[0], da_anchor[1]], dim=0)
        images = images
        target = label
        
        optimizer.zero_grad()
        f1 = model(anchor)
        f2 = model(da_anchor)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, target)

        loss.backward()
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        if batch_idx % 15 == 14:
            print(f'epoch{epoch}, batch{batch_idx+1} loss: {running_loss / 15}')
            train_loss = running_loss / 15
            running_loss = 0.

    return train_loss


if __name__ == '__main__':

    train_dic = make_datapath_dic(phase='train')
    transform = ImageTransform(224)
    train_dataset = MyDataset(train_dic, transform=transform, phase='train')
    batch_size = 32

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 30
    model = SupConModel("resnet18",128)
    model = model.to(DEVICE)

    criterion = SupConLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-03, weight_decay=1.0e-02)
    scheduler = lr_scheduler.OneCycleLR(optimizer, epochs=EPOCHS, steps_per_epoch=len(train_loader),
                                        max_lr=1.0e-3, pct_start=0.1, anneal_strategy='cos',
                                        div_factor=1.0e+3, final_div_factor=1.0e+3
                                        )

    torch.autograd.set_detect_anomaly(True)

    x_epoch_data = []
    y_train_loss_data = []

    for epoch in range(EPOCHS):

        loss_epoch = train(args, model, tra