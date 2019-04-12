# XrayClassifier

Uses Pytorch to classify X-Ray images into 4 classes:
- Cardiomegaly
- Effusion
- Mass
- No Finding

No Finding doesn't mean that the person is healthy.

# Installation:

Go to https://pytorch.org/ on how to install both pytorch and torchvision

# Notes:

csv_test.py is used to create the dataframe.csv file for the dataset, run this file first if you run for the first time.
XRayClassifierMain is the script used to train the network, for testing use XRayTestingScript.
Config options are found in the configs.py script.