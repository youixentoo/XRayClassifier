# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:53:07 2019

@author: gebruiker
"""
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets, models
from skimage import io, transform

import time
import os



def getTraining(dataset):
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=4, shuffle=True,
                                       num_workers=0)
    
def getTesting(dataset):
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=4, shuffle=False,
                                       num_workers=0)



def dataloader(folderPath, mean, std):
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
        #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #transforms.Normalize(mean=[0.485, 0.456, 0.406],
         #                    std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.ImageFolder(root=folderPath, 
                                   transform=data_transform)
    
    return dataset 


# Work in progress
def modelInit(device):
    model = models.alexnet(pretrained=False)
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)
    return criterion, optimizer, model




















###