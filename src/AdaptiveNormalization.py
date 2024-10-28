#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains the adaptive normalization layer used in the paper.

@author: Iraj Masoudian
@author: Jan Benedikt Ruhland
@author: Dominik Heider
@maintainer: Jan Benedikt Ruhland - jan.ruhland@hhu.de
"""

import torch 
import torch.nn as nn


class AdaptiveNorm(nn.Module):
    def __init__(self, dimensions, statisticWeights=False):
        """
        

        Parameters
        ----------
        dimensions : Tuple
            Define the dimensions for which the normalization should be 
            performed.
        statisticWeights : Boolean, optional
            If the initialized statistics is too high to update with gradient
            descent, we introduce the scaling factor m_w, v_w. 
            The default is False.

        """
        super(AdaptiveNorm, self).__init__()
        
        self.statWeights=statisticWeights
        
        self.gamma=torch.nn.Parameter(torch.ones(*dimensions))
        self.beta=torch.nn.Parameter(torch.zeros(*dimensions))
        self.eps=0.0000001

        self.gamma.requires_grad = True
        self.beta.requires_grad = True

        self.m=torch.nn.Parameter(torch.zeros(*dimensions))
        self.v=torch.nn.Parameter(torch.ones(*dimensions))

        self.m.requires_grad = True
        self.v.requires_grad = True
        
        self.m_w=torch.nn.Parameter(torch.ones(*dimensions))
        self.v_w=torch.nn.Parameter(torch.ones(*dimensions))
        
        self.m_w.requires_grad = True
        self.v_w.requires_grad = True
        
        self.relu=nn.ReLU()
        
    def forward(self, x):
        if self.statWeights:
            x=torch.add(torch.mul(torch.div(torch.subtract(x,self.m*self.m_w),torch.sqrt(self.relu(self.v*self.v_w)+self.eps)), self.gamma), self.beta)
        else:
            x=torch.add(torch.mul(torch.div(torch.subtract(x,self.m),torch.sqrt(self.relu(self.v)+self.eps)), self.gamma), self.beta)
        return x   