#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 14:15:49 2019

@author: tim
"""

import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, log_loss
from torch.utils.data import Dataset
import PIL
import torchvision.transforms as transforms 

classes = 4
class SliceCNN(nn.Module):
    def __init__(self):
        super(SliceCNN, self).__init__()
    
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1025,1), stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            #nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.fc = nn.Linear(64 * 126 *23, classes)

    def forward(self, x):
        out = self.layer1(x)
        print(out.shape)
        
        
        #out = self.layer2(out)
        #print(out.shape)
        #out = self.layer3(out)
        #print(out.shape)
        #out = self.layer4(out)
        #print(out.shape)

        #out = out.reshape(out.size(0), -1)
        #print(out.shape)

        #out = self.fc(out)
        return out
    
# Always check your model are you able to make a forward pass and shapes match your expectations?
image = torch.randn(1, 1, 1025, 200)
cnn = SliceCNN()
output = cnn(image)
print("input shape:")
print(image.shape)
print("output shape:")
print(output.shape)