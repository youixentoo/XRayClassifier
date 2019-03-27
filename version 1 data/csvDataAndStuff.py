# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 12:24:31 2019

@author: gebruiker
"""
import os
import pandas as pd
    
    
def getDatasetFrame(imagesDir):
    directory = os.path.dirname(os.path.realpath(__file__))
    currentDirItems = os.listdir(directory+os.sep+imagesDir)
    
    DataCSVFrame = pd.read_csv("Data_Entry_2017.csv", usecols=["Image Index","Finding Labels"])
    DataCSVFrame = DataCSVFrame.rename(columns=lambda x: x.strip().replace(' ','_'))
    
    DatasetFrame = DataCSVFrame[DataCSVFrame["Image_Index"].isin(currentDirItems)]
    
    return DatasetFrame