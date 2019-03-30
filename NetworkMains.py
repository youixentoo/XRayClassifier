# -*- coding: utf-8 -*-
"""
Moet nog kopieren van sigma, dit werkt waarschijnlijk nog voor geen meter
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

# Work in progress
def modelInit(device):
    model = models.alexnet(pretrained=False)

    criterion = nn.CrossEntropyLoss() #nn.NLLLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
    model.to(device)
    return criterion, optimizer, model




def trainNetwork(device, dataset, config, model, criterion, optimizer):
    since = time.time()

    num_epochs = config.getEpochs()
    valSplit = config.getValidationSplit()

    datasetSize = dataset.__len__()
    
    print(datasetSize)


    """
    #dataset_sizes = {"train": len(dataloaders["train"]), "val": len(dataloaders["val"])}


    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)



        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                #scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                inputs = inputs.float()
                labels = labels.float()

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = (model(inputs)).float()
                    _, preds = torch.max(outputs, 0)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == (labels.long()).data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

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
    """



def testNetwork(testData, model, device, lowX, highX):

    correct = 0
    total = 0
    wrongLabels = []

    with torch.no_grad():
        for dataTensor, labels in testData:
            dataTensor, labels = dataTensor.to(device), labels.to(device)
            labels.float()
            output = model(dataTensor.float())

            #print("Label: ",str(labels.item()),"\nPredict: ",str(output.item()))

#            if output.item() > lowX and output.item() < highX:
#                with torch.no_grad():
#                    output = updatedValue

            _, predicted = torch.max(output, 0)
            #print(str(labels.item()), "::", str(output.item()))
            total += 1 #labels.size(0)
            labels = labels.long()
            correct += (predicted == labels).sum().item()
            if not predicted == labels:
                wrongLabels.append(labels)


    print('Accuracy of the network on the test data: %1.4f %%' % (
        100 * correct / total))
    return wrongLabels















###
