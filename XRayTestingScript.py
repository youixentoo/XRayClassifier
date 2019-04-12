# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 12:51:07 2019

@author: gebruiker

Tests the network
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

from sklearn.model_selection import KFold, RepeatedKFold
from sklearn.metrics import confusion_matrix

import time
import os

import DataUtils as DU
import DatasetCreation as DC
import NetworkMains as NM
import configs

import CMPlot

def main():
    # Configs
    config = configs.config()

    # Reading the selected data
    DataCSVFrame = pd.read_csv("DataFrame.csv", usecols=["Image_Index","Finding_Labels"], index_col=False)
    
    labelsSet = set(DataCSVFrame["Finding_Labels"].values)
#    labelsFreq = np.unique(list(DataCSVFrame["Finding_Labels"].values), return_counts=True)
#    print(labelsFreq)

    # Dictionary with the label as key and the index in the set as value
    labelsDict = {}
    # Dictionary that is the reverse of the one above, to change back from value to the corresponding label
    labelDictClassify = {}

    # Filling the dictionaries
    for index, label in enumerate(labelsSet):
        labelsDict[label] = index
        labelDictClassify[index] = label

    # Path where all the images are stored
    imgPath = config.getImagePath()
    # Creating the dataset
    xrayDataset = DC.XRayDataset(DataCSVFrame, imgPath, labelsDict)
    
    #print(xrayDataset.xrayClassFrame)
    
    device = DU.getDevice()

    # Gets the ranges of training and test data
    training, testing = DU.splitTrainTest(xrayDataset, config)
    
    # Initialize the model
    model = models.alexnet(pretrained=False, num_classes=4)
    
    #print(model)
    
    # Load the trained model
    cwd = os.path.dirname(os.path.realpath(__file__))
    model.load_state_dict(torch.load("%s%s%s.pth" % (cwd, os.sep, config.getModelName())))
    model.eval()
    
    # Testing the model
    wrongLabels, labelsCM, predsCM = NM.testing(xrayDataset, testing, model, device, labelDictClassify)
    
    # Confusion Matrix
    CMPlot.plot_confusion_matrix(labelsCM, predsCM, list(labelsSet), normalize=False, title="Confusion Matrix")
    
    plt.show()
    
    print(labelsDict)
    
    
    
    
    
if __name__ == "__main__":
    main()