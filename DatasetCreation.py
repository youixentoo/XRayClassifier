# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:24:31 2019

@author: gebruiker

Creates the dataset object
"""

import os
import io
import sys
import shutil
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms, utils, datasets, models

from PIL import Image

#def createDataset(testTrainSplit, trainValidationSplit, classes, sourceDirectory, targetDirectory):
    #_createDatasetDirs(classes, targetDirectory)

class XRayDataset(Dataset):
    """
    XRay Dataset

    datasetDF should only contain images and labels, not indexes or headers
    """

    def __init__(self, datasetDF, imagesPath, labelsDict):
        # Dataset values
        self.xrayClassFrame = datasetDF
        self.imagesPath = imagesPath
        self.labelsDict = labelsDict

    def __len__(self):
        return len(self.xrayClassFrame)

    def __getitem__(self, indexes):
        # Lists to put the data in
        imagesL = []
        labelsL = []
        
        for index in indexes:
            # Get the image name from the index
            imgName = os.path.join(self.imagesPath, self.xrayClassFrame.iloc[index, 0])
            
            # Convert the image to an array and resize it
            image = np.array(Image.open(imgName).resize((224, 224)).convert("RGB"))
            
            # Swap color axis because
            # numpy image: H x W x C
            # torch image: C X H X W
    
            image = image.transpose((2, 0, 1))
            # The label name from the image
            xrayClass = self.xrayClassFrame.iloc[index, 1]
            
            # Get the label int from the xrayClass
            label = self.labelsDict.get(xrayClass)
            
            # Add the data to the lists
            imagesL.append(image)
            labelsL.append(label)
            

        # Convert the numpy array to a pytorch tensor
        imgs = torch.from_numpy(np.asarray(imagesL))
        labs = torch.from_numpy(np.asarray(labelsL))
            
        return imgs, labs




# Returns a dataframe with only the classification from the images in the folder.
# Also replaces the " " in the column names with a "_"
# iloc[index, 1] = imageName
# iloc[index, 2] = classification
def getDatasetFrame(imagesDir):
    directory = os.path.dirname(os.path.realpath(__file__))
    currentDirItems = os.listdir(directory+os.sep+imagesDir)

    DataCSVFrame = pd.read_csv("Data_Entry_2017.csv", usecols=["Image Index","Finding Labels"], index_col=False)
    DataCSVFrame = DataCSVFrame.rename(columns=lambda x: x.strip().replace(' ','_'))

    DatasetFrame = DataCSVFrame[DataCSVFrame["Image_Index"].isin(currentDirItems)]
    DatasetFrame = DatasetFrame.reset_index()

    return DatasetFrame


# Saves the dataframe as a .csv file
def saveDFToCSV(dataFrame, filePath):
    test = dataFrame.to_csv(filePath, index=False)
    return test


