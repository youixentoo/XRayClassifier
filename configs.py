# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 12:33:28 2019

@author: thijs

Configs
"""
import os

class config():
    
    def __init__(self):
        # Configs
        ###########################################################################
        # How much training data to testing data (%)
        self._trainingSplit = 80
        # How much of the training data is validation (%)
        self._validationSplit = 5
        # Number of epochs
        self._epochs = 5
        # Batch size
        self._batches = 50
        # Model name
        self._modelName = "XRAY_1"
        # Path where the images are stored (/ = os.sep)
        self._imagePath = os.path.dirname(os.path.realpath(__file__))+os.sep+"version 1 data"+os.sep+"Images"
   
    
    def getTrainsplit(self):
        return self._trainingSplit
     
    def getValidationSplit(self):
        return self._validationSplit
     
    def getEpochs(self):
        return self._epochs
    
    def getBatchSize(self):
        return self._batches
 
    def getModelName(self):
        return self._modelName
        
    def getImagePath(self):
        return self._imagePath