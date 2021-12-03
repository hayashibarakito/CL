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


class ConLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def calculate_euclidean(self, x1, x2):
        return (x1 - x2).pow(2).sum(1)

    def forward(self, anchor, da_anchor, size_average=True):
        print("anchor:",anchor.shape)
        print("da",da_anchor.shape)
        distance_positive = self.calculate_euclidean(anchor, da_anchor)
        print("D:",distance_positive)

        losses = 2

        return losses.mean() if size_average else losses.sum()


def train(args, model, train_loader, criterion, optimizer, scheduler, epoch):
    model.train()
    running_loss = 0.

    for batch_idx, (anchor, da_anchor, _) in enumerate(train_loader):

        print(batch_idx)
        print(anchor.shape)
        
        optimizer.zero_grad()

        f1 = model(anchor)
        f2 = model(da_anchor)
        loss = criterion(f1, f2)

    return loss


if __name__ == '__main__':

    train_dic = make_datapath_dic(phase='train')
    print(train_dic)
    transform = ImageTransform(224)
    train_dataset = MyDataset(train_dic, transform=transform, phase='train')
    batch_size = 3

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 30
    model = SupConModel("resnet18",128)

    criterion = ConLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-03, weight_decay=1.0e-02)
    scheduler = lr_scheduler.OneCycleLR(optimizer, epochs=EPOCHS, steps_per_epoch=len(train_loader),
                                        max_lr=1.0e-3, pct_start=0.1, anneal_strategy='cos',
                                        div_factor=1.0e+3, final_div_factor=1.0e+3
                                        )

    torch.autograd.set_detect_anomaly(True)

    x_epoch_data = []
    y_train_loss_data = []

    for epoch in range(EPOCHS):
        
        loss_epoch = train(args, model, train_loader, criterion, optimizer, scheduler, epoch)
        break
        # scheduler.step()
        x_epoch_data.append(epoch)
        y_train_loss_data.append(loss_epoch)


    plt.plot(x_epoch_data, y_train_loss_data, color='blue', label='train_loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    plt.title('loss')
    plt.show()

    if args.save_model:
        model_name = str(y_train_loss_data[-1]) + '.pth'
        torch.save(model.state_dict(), model_name)
        print(f'Saved model as {model_name}')
