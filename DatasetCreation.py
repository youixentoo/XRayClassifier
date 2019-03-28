# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:24:31 2019

@author: gebruiker
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
from skimage import io, color

#def createDataset(testTrainSplit, trainValidationSplit, classes, sourceDirectory, targetDirectory):
    #_createDatasetDirs(classes, targetDirectory)

class XRayDataset(Dataset):
    """
    XRay Dataset

    datasetDF should only contain images and labels, not indexes or headers
    """

    def __init__(self, datasetDF, imagesPath, labelsDict):
        self.xrayClassFrame = datasetDF
        self.imagesPath = imagesPath
        self.labelsDict = labelsDict

    def __len__(self):
        return len(self.xrayClassFrame)

    def __getitem__(self, index):
        imgName = os.path.join(self.imagesPath, self.xrayClassFrame.iloc[index, 0])
        imageG = io.imread(imgName)

        # Convert the gray-scaled image to RGB (3 layers)
        image = color.gray2rgb(imageG)

        xrayClass = self.xrayClassFrame.iloc[index, 1]
        #xrayClass = xrayClass.astype('float')#.reshape(-1,2)


        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))

        imgT = torch.from_numpy(image)

        label = self.labelsDict.get(xrayClass)
        labelTensor = torch.from_numpy(np.asarray([label]))


        sample = {'image':imgT,  'label':labelTensor}


        return sample




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






"""

def createDataset():
    testCSV()



def testCSV():
    csv_frame = pd.read_csv("Data_Entry_2017.csv")
    print(csv_frame.as_matrix())

def _createDatasetDirs(classes, targetDir):
    mainDirs = ["Training","Validation","Testing"]
    for mains in mainDirs:
        try:
            os.mkdir(targetDir+os.sep+mains)
        except Exception:
            pass
        for classification in classes:
            try:
                os.mkdir(targetDir+os.sep+mains+os.sep+classification)
            except Exception:
                pass






def getTraining(dataset):
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=4, shuffle=True,
                                       num_workers=0)

def getTesting(dataset):
    return torch.utils.data.DataLoader(dataset,
                                       batch_size=4, shuffle=False,
                                       num_workers=0)

def _createTrainValidationData(trainValidationSplit, classes, sourceDir, targetDir):
    return None

"""



if __name__ == "__main__":
    createDataset()
