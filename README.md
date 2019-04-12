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

csv_test.py was used to create the dataframe.csv file for the dataset, run this file first if you run for the first time and you're missing the dataframe.csv file.
XRayClassifierMain is the script used to train the network, for testing use XRayTestingScript.
Config options are found in the configs.py script.

# Data:

It *should* work with the folder containing all images as the images source, but I had no way to test this. 
If it doesn't work, download the used dataset from here:
https://drive.google.com/file/d/1qqb9sBdm_cx1KBhjbLT1ZZcojXYLnsHz/view?usp=sharing
If the link has expired or doesn't work, please contact me.
