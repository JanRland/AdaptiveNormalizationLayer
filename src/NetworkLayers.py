#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 18:50:07 2024

@author: jan
"""
import torch 
import torch.nn as nn


class AdaptiveNorm(nn.Module):
    def __init__(self, dimensions):
        super(AdaptiveNorm, self).__init__()
        
        self.gamma=torch.nn.Parameter(torch.ones(*dimensions))
        self.beta=torch.nn.Parameter(torch.zeros(*dimensions))
        self.eps=0.00001

        self.gamma.requires_grad = True
        self.beta.requires_grad = True

        self.m=torch.nn.Parameter(torch.zeros(*dimensions))
        self.v=torch.nn.Parameter(torch.ones(*dimensions))

        self.m.requires_grad = True
        self.v.requires_grad = True
        
    def forward(self, x):
        x=torch.add(torch.mul(torch.div(torch.subtract(x,self.m),torch.sqrt(self.v+self.eps)), self.gamma), self.beta)
       
        return x   
    
class InvAdaptiveNorm(nn.Module):
    def __init__(self, dimensions):
        super(InvAdaptiveNorm, self).__init__()
        
        self.gamma=torch.nn.Parameter(torch.ones(*dimensions))
        self.beta=torch.nn.Parameter(torch.zeros(*dimensions))
        self.eps=0.00001

        self.gamma.requires_grad = True
        self.beta.requires_grad = True

        self.m=torch.nn.Parameter(torch.zeros(*dimensions))
        self.v=torch.nn.Parameter(torch.ones(*dimensions))

        self.m.requires_grad = True
        self.v.requires_grad = True
        
    def forward(self, x):
        x=torch.add(torch.mul(torch.div(torch.subtract(x,self.beta),self.gamma), torch.sqrt(self.v+self.eps)), self.m)
        
        return x    
        

class ResNetBlock1D(nn.Module):
    def __init__(self, inputF, outputF):
        super(ResNetBlock1D, self).__init__()
        self.l1=nn.Linear(outputF, outputF)
        self.norm1=AdaptiveNorm((outputF,))
        self.dropout1=nn.Dropout(0.2)
        
        self.l2=nn.Linear(outputF, outputF)
        self.norm2=AdaptiveNorm((outputF,))
        self.dropout2=nn.Dropout(0.2)
        self.scale=None
        if inputF!=outputF:
            self.scale=nn.Linear(inputF, outputF)
        
        self.l3=nn.Linear(outputF, outputF)
        self.relu=nn.ReLU()
        
    def forward(self, x):
        if self.scale!=None:
            x=self.scale(x)
        xRes=x
        x=self.l1(x)
        x=self.relu(x)
        x=self.dropout1(x)
        x=self.norm1(x)
        
        x=self.l2(x)
        x=self.relu(x)
        x=self.dropout2(x)
        x=self.norm2(x)

        x=self.l3(x)
        
        x=x+xRes
        
        return x
    
    
class ResNetBlock2D(nn.Module):
    def __init__(self, inputF, outputF):
        super(ResNetBlock2D, self).__init__()
        self.relu=nn.ReLU()
        self.conv1=nn.Conv1d(outputF, outputF, 9, 1, 4)
        self.conv2=nn.Conv1d(outputF, outputF, 5, 1, 2)
        self.conv3=nn.Conv1d(outputF, outputF, 3, 1, 1)
        
        self.dropout1=nn.Dropout(0.2)
        self.dropout2=nn.Dropout(0.2)
        self.dropout3=nn.Dropout(0.2)

        self.scale=None
        if inputF!=outputF:
            self.scale=nn.Conv1d(inputF, outputF, 3, 1, 1)

        self.norm1=AdaptiveNorm((outputF,1))
        self.norm2=AdaptiveNorm((outputF,1))
        self.norm3=AdaptiveNorm((outputF,1))
        
    def forward(self, x):
        if self.scale!=None:
            x=self.scale(x)
        xRes=x
        
        x=self.conv1(x)
        x=self.norm1(x)
        x=self.relu(x)
        x=self.dropout1(x)
        
        x=self.conv2(x)
        x=self.norm2(x)
        x=self.relu(x)
        x=self.dropout2(x)
        
        x=self.conv3(x)
        x=self.norm3(x)
        x=self.relu(x)
        
        x=x+xRes

        return x   
    
    
class ResNetBlock3D(nn.Module):
    def __init__(self, inputF, outputF):
        super(ResNetBlock2D, self).__init__()
        self.relu=nn.ReLU()
        self.conv1=nn.Conv2d(outputF, outputF, 9, 1, 4)
        self.conv2=nn.Conv2d(outputF, outputF, 5, 1, 2)
        self.conv3=nn.Conv2d(outputF, outputF, 3, 1, 1)
        
        self.dropout1=nn.Dropout(0.2)
        self.dropout2=nn.Dropout(0.2)
        self.dropout3=nn.Dropout(0.2)

        self.scale=None
        if inputF!=outputF:
            self.scale=nn.Conv2d(inputF, outputF, 3, 1, 1)

        self.norm1=AdaptiveNorm((outputF,1,1))
        self.norm2=AdaptiveNorm((outputF,1,1))
        self.norm3=AdaptiveNorm((outputF,1,1))
        
    def forward(self, x):
        if self.scale!=None:
            x=self.scale(x)
        xRes=x
        
        x=self.conv1(x)
        x=self.norm1(x)
        x=self.relu(x)
        x=self.dropout1(x)
        
        x=self.conv2(x)
        x=self.norm2(x)
        x=self.relu(x)
        x=self.dropout2(x)
        
        x=self.conv3(x)
        x=self.norm3(x)
        x=self.relu(x)
        
        x=x+xRes

        return x  
    
