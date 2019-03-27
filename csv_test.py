# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 22:36:02 2019

@author: gebruiker
"""
import pandas as pd
import numpy as np
import os
import DatasetCreation as DC

def saveToCSVFile():
    classes = ["cardiomegaly", "effusion" , "mass", "no finding", "other"]
    classes2 = ["Cardiomegaly", "Effusion" , "Mass", "No Finding"]#, "Other"]
    
    fileP = os.path.dirname(os.path.realpath(__file__))+os.sep+"DataFrame.csv"
    df = DC.getDatasetFrame("version 1 data"+os.sep+"Images")
   # any(types in classification.lower() for types in classes):
    #df['Finding_Labels'] = np.where("|" in df['Finding_Labels'], "Other", df['Finding_Labels'])
    df = df[df['Finding_Labels'].isin(classes2)]
    
    
    test = DC.saveDFToCSV(df, fileP)
#    print(test)
#    saveToFile(test, "DataFrame.csv")
    
    
def saveToFile(data, fileName):
    with open("%s" % fileName, "w") as file:
        file.write(data)
    file.close()
    

saveToCSVFile()