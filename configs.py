# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 12:33:28 2019

@author: gebruiker
"""

class config():
    
    def __init__(self):
        # Configs
        ###########################################################################
        # How much training data to testing data (%)
        self._trainingSplit = 80
        # How much of the training data is validation (%)
        self._validationSplit = 5
        # Number of epochs
        self._epochs = 1
        # Batch size
        self._batches = 10
        # Model name
        self._modelName = "XRAY_1"
   
    
    def getTrainsplit(self):
        return self._trainingSplit
     
    def getValidationSplit(self):
        return self._validationSplit
     
    def getEpochs(self):
        return self._epochs
    
    def getBatchSize(self):
        return self._batches
 
    def getModelName(self, useBestModel=True):
        if useBestModel:
            return self._modelName
        else:
            return self._modelName + "_other"