# -*- coding: utf-8 -*-
"""
Network functions
"""
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time
import copy

from torch.utils import data
from torch.utils.data import BatchSampler, SequentialSampler
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
from torchvision import utils, datasets, models
from skimage import io, transform



#class Net(nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        self.conv1 = nn.Conv2d(3, 6, 5)
#        self.pool = nn.MaxPool2d(2, 2)
#        self.conv2 = nn.Conv2d(6, 16, 5)
#        self.fc1 = nn.Linear(16 * 5 * 5, 120)
#        self.fc2 = nn.Linear(120, 84)
#        self.fc3 = nn.Linear(84, 4) # 4 classes
#
#    def forward(self, x):
#        x = self.pool(F.relu(self.conv1(x)))
#        x = self.pool(F.relu(self.conv2(x)))
#        x = x.view(-1, 16 * 5 * 5)
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        return x

# Getting the model and other needed stuff
def modelInit(device):
    model = models.alexnet(pretrained=False, num_classes=4)

    criterion = nn.CrossEntropyLoss() #nn.NLLLoss()
#    optimizer = optim.Adam(model.parameters(), lr=0.003)
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    model.to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        
    return criterion, optimizer, model




def trainNetwork(device, dataset, trainSets, valSets, config, model, criterion, optimizer, batchsize):
    since = time.time()

    num_epochs = config.getEpochs()

    datasetSize = dataset.__len__()
    
    # To save the best model later on
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    index = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Loops through the kfolds and resets the index after the last item in the kfold data
        dataSets = {"train":trainSets[index], "val":valSets[index]}
        if index % 20 == 19:
            index = 0
        else:
            index += 1

        #Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Create the dataset batches
            dataSet = list(BatchSampler(SequentialSampler(dataSets[phase]), batch_size=batchsize, drop_last=False))
            dSSize = len(dataSet)

            # Iterate over data.
            for i, datasetIndexes in enumerate(dataSet):
                print("{:.4f}% Done".format((i/dSSize)*100))
                images, labels = dataset.__getitem__(datasetIndexes)
                
                images = images.to(device)
                labels = labels.to(device)

                images = images.float()
                labels = labels.float()
                
                #images = images.unsqueeze(0)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = (model(images)).float()
                    preds = round(sum(abs(outputs.detach().numpy()[0])) / len(outputs.detach().numpy()[0]))
                    #_, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels.long())

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * images.size(0)
                running_corrects += torch.sum(preds == (labels.long()).data)

            epoch_loss = running_loss / len(dataSets[phase])
            epoch_acc = running_corrects.double() / len(dataSets[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)

    return model
   



# Testing of the model using test data
# Returns a list of the wrongly guessed labels
def testing(dataset, testingIndexes, model, device, labelDictClassify):
    
    # Variables needed in the function
    correct = 0
    total = 0
    wrongLabels = []
    labelsForCM = []
    predsForCM = []
    
    # Runs the model without gradient calculations as that is not needed for testing.
    # It also saves on memory.
    with torch.no_grad():
        # Looping over the data stored in testData to be tested on the model.
        for index in testingIndexes:
            # Predicting the label based on the image
            image, label = dataset.__getitem__([index])
            image, label = image.to(device), label.to(device)
            label.float()
            output = model(image.float())
            
            # Prediction
            predicted = round(sum(abs(output.detach().numpy()[0])) / len(output.detach().numpy()[0]))

            label = label.long()
            
            # Counts the number of correct guesses
            correct += (predicted == label).sum().item()

            # Adds all wrongly guessed label to a list
            if not predicted == label:
                wrongLabels.append([label.item(),predicted.item()])
                
            labelsForCM.append(labelDictClassify.get(label.item()))
            predsForCM.append(labelDictClassify.get(predicted.item()))
                
            total += 1 
            
    print('Accuracy of the network on the test data: %1.4f %%' % (
        100 * correct / total))
    
    return wrongLabels, labelsForCM, predsForCM





# Saves the model as file to be used elsewhere
def save_model(model, name):
    torch.save(model.state_dict(), name+".pth") 










###
