# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:53:07 2019

@author: gebruiker

    ##############################################
    Checklist:

    Nog te doen:
        - Matrix aanpassen van 1024x1024 naar 224x224
            - Graag als functie
        - AlexNet epoch
            - Voornamelijk die van Sigma als richtlijn
        - Confusion Matrix is in principe hetzelfde als wat ik eigenlijk printte bij Sigma



    ##############################################


Data is created in the csv_test.py script


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

import DataUtils as DU
import DatasetCreation as DC
import NetworkMains as NM


def main():
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
    imgPath = os.path.dirname(os.path.realpath(__file__))+os.sep+"version 1 data"+os.sep+"Images"
    # Creating the dataset
    xrayDataset = DC.XRayDataset(DataCSVFrame, imgPath, labelsDict)

    # Getting the first image from the dataset
    item = xrayDataset.__getitem__(0)
    lab = item["image"]
    print(lab)














if __name__ == "__main__":
    main()
