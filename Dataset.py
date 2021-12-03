import os
import glob
import random
from timm import models

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image

from torchvision.transforms import transforms
from gaussian_blur import GaussianBlur
from torch.utils.data import Dataset,DataLoader

from models import *
from losses import *

def make_datapath_dic(phase='train'):
    root_path = './CL/data/' + phase
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
    
#dic = make_datapath_dic(phase='train')
#print(dic)
#print(dic.keys())
#print(dic[0])
#print(dic.values())

class ImageTransform():
    def __init__(self, size, s=1):
            """Return a set of data augmentation transformations as described in the SimCLR paper."""
            color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
            self.data_transform  = {'train':    transforms.Compose([transforms.RandomResizedCrop(size=size),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.RandomApply([color_jitter], p=0.8),
                                                transforms.RandomGrayscale(p=0.2),
                                                GaussianBlur(kernel_size=int(0.1 * size)),
                                                transforms.ToTensor()
                                    ]),
                                    'test':     transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2))
                                    ])
                                    }
    def __call__(self, img, phase='train'):
        return self.data_transform[phase](img)

class MyDataset(Dataset):
    def __init__(self, datapath_dic, transform=None, phase='train'):
        self.datapath_dic = datapath_dic
        self.transform = transform
        self.phase = phase

        all_datapath = []
        for data_list in self.datapath_dic.values():
            all_datapath += data_list
            
        self.all_datapath = all_datapath

    def __len__(self):
        return len(self.all_datapath)

    def __getitem__(self, idx):
        image_path = self.all_datapath[idx]
        #print(image_path)

        image = self.transform(Image.open(image_path), self.phase)
        da_image = self.transform(Image.open(image_path), self.phase)

        if 'airplane' in image_path:
            image_label = 0
        elif 'bird' in image_path:
            image_label = 1
        elif 'car' in image_path:
            image_label = 2
        elif 'cat' in image_path:
            image_label = 3
        elif 'deer' in image_path:
            image_label = 4
        elif 'dog' in image_path:
            image_label = 5
        elif 'frog' in image_path:
            image_label = 6
        elif 'horse' in image_path:
            image_label = 7
        elif 'ship' in image_path:
            image_label = 8
        elif 'truck' in image_path:
            image_label = 9

        #print(image_label)
        
        return image, da_image, image_label

"""
train_dic = make_datapath_dic(phase='train')
transform = ImageTransform(224)
train_dataset = MyDataset(train_dic, transform=transform, phase='train')
batch_size = 5

model = SupConModel("resnet18",128)
criterion = SupConLoss()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

for batch_idx, (anchor, da_anchor, label) in enumerate(train_loader):

        images = torch.cat([anchor[0], da_anchor[1]], dim=0)
        images = images
        target = label
        bsz = target.shape[0]

        f1 = model(anchor)
        f2 = model(da_anchor)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, target)
        break
"""
    