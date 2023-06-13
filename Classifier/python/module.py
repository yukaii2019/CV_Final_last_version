#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: rakshit
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b7

class efficientnet(nn.Module):
    def __init__(self):
        super(efficientnet, self).__init__()

        self.model = efficientnet_b7(pretrained=True) 
        list(self.model.features.children())[0][0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.model.classifier = nn.Sequential(self.model.classifier[0], 
                                            nn.Linear(in_features=2560, out_features=100, bias=True),
                                            nn.Linear(in_features=100, out_features=2, bias=True))
        
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = efficientnet()

    print(model.model.classifier)
    #print(list(model.model.features.0)[0])
    print(list(model.model.features.children())[0][0])
    #print(list(list(model.model.children())[0].children()))

    B = 3
    H = 192
    W = 256

    x = torch.rand(B, 1, H, W)

    x= model.forward(x)

    print(x.shape)