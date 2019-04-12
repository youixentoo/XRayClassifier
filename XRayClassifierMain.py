# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:53:07 2019

@author: gebruiker

Data is created in the csv_test.py script

Script used to train the network

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

import time
import os

import DataUtils as DU
import DatasetCreation as DC
import NetworkMains as NM
import configs


def main():
    # Configs
    config = configs.config()

    # Reading the selected data
    DataCSVFrame = pd.read_csv("DataFrame.csv", usecols=["Image_Index","Finding_Labels"], index_col=False)
    labelsSet = set(DataCSVFrame["Finding_Labels"].values)

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


#    # Getting the first image from the dataset
#    imgs, labs = xrayDataset.__getitem__([8307])
#    print(len(imgs))
#    print(imgs)


    # Get the device (cpu/gpu) to run the model on
    device = DU.getDevice()

    # Gets the ranges of training and test data
    training, testing = DU.splitTrainTest(xrayDataset, config)

    # Get the train and validation sets
    trainSets, valSets = DU.trainValSets(training, config)

    # Initialize the criterion, optimizer and model
    criterion, optimizer, model = NM.modelInit(device)
    
    # Get the batchsize
    batchsize = config.getBatchSize()
    
    
    # Train the model
    trainedModel = NM.trainNetwork(device, xrayDataset, trainSets, valSets, config, model, criterion, optimizer, batchsize)
    
    # Save the model to be used for testing
    NM.save_model(trainedModel, config.getModelName())
    


if __name__ == "__main__":
    main()

