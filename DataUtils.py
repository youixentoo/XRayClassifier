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
from sklearn.model_selection import KFold

import time
import os


def getDevice(cudaAllowed=False):
    if cudaAllowed:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = "cpu"

    print("Running on: %s" % device)
    return device

# Returns index ranges
def splitTrainTest(dataset, config):
    trainSplit = config.getTrainsplit()
    datasetSize = dataset.__len__()

    training = round(datasetSize * (trainSplit / 100))
    testing = training + 1

    trainingRange = range(training)
    testingRange = range(testing,datasetSize)

    return trainingRange, testingRange

# Splits the training data into training and validation data via
# SKLearn KFold, returns a list of indexes.    
def trainValSets(trainingRange, config):
    valSplit = config.getValidationSplit()
    trainingIndexes = [*trainingRange]
    percSpl = round(100 / valSplit)
    kf = KFold(percSpl)

    trainSets, valSets = [], []

    for tr, ts in kf.split(trainingIndexes):
        trainSets.append(tr)
        valSets.append(ts)

    return trainSets, valSets




















###
