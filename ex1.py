import os
import gc
import random
from pathlib import Path
import glob

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

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms as T
from models import *
from losses import *

def make_datapath_dic(phase='train'):
    root_path = './CL/data_ex/' + phase
    class_list = os.listdir(root_path)
    class_list = [class_name for class_name in class_list if not class_name.startswith('.')]
    datapath_dic = {}
    for i, class_name in enumerate(class_list):
        data_list = []
        target_path = os.path.join(root_path, class_name, '*.jpg')
        for path in glob.glob(target_path):
            data_list.append(path)
        datapath_dic[i] = data_list

    return datapath_dic
    
class SupConDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        paths,
        labels,
        transform: A.Compose,
    ):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index: int):
        path, label = self.paths[index], self.labels[index]
        img = Image.open(path)
        img = np.array(img)
        # 参考
        # https://github.com/HobbitLong/SupContrast/blob/8d0963a7dbb1cd28accb067f5144d61f18a77588/util.py#L9
        img_1 = self.transform(image=img)["image"]
        img_2 = self.transform(image=img)["image"]
        img = [img_1, img_2]
        return {"image": img, "target": label}

SupConDataset()