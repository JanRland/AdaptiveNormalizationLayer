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
    def __init__(self, dimensions, statisticWeights=False, offset=0.):
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
        offset : Float
            If an accurate estimate of the initial statistical moment is 
            unavailable, you can tune it as a hyperparameter avoiding 
            unstable training convergence. 

        """
        
        
        super(AdaptiveNorm, self).__init__()
        
        self.statWeights=statisticWeights
        
        self.gamma=torch.nn.Parameter(torch.ones(*dimensions))
        self.beta=torch.nn.Parameter(torch.zeros(*dimensions))
        self.eps=0.0000001

        self.gamma.requires_grad = True
        self.beta.requires_grad = True
        

        self.m=torch.nn.Parameter(torch.ones(*dimensions)*offset)
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
            x=torch.add(torch.mul(torch.div(torch.subtract(x,self.beta),self.gamma), torch.sqrt(self.relu(self.v*self.v_w)+self.eps)), self.m*self.m_w)
        else:
            x=torch.add(torch.mul(torch.div(torch.subtract(x,self.beta),self.gamma), torch.sqrt(self.relu(self.v)+self.eps)), self.m)
        return x  