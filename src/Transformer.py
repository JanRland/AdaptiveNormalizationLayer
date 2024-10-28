#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This script contains the transformer network used in the paper.

@author: Iraj Masoudian
@author: Jan Benedikt Ruhland
@author: Dominik Heider
@maintainer: Jan Benedikt Ruhland - jan.ruhland@hhu.de
"""

import torch.nn as nn
import torch
import numpy as np
from AdaptiveNormalization import AdaptiveNorm


    
class PositionalEncoding(nn.Module):
    def __init__(self, N_e, N_s):
        """
        

        Parameters
        ----------
        N_e : Int
            Embedding length.
        N_s : Int
            Length of sequence. 


        """
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(N_s, N_e)
        position = torch.arange(0, N_s, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, N_e, 2).float() * -(np.log(10000.0) / N_e))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        
        return x + self.pe[:, :x.size(1)] 



class Attention(nn.Module):
    def __init__(self, N_s, N_e, N_h):
        """
        

        Parameters
        ----------
        N_s : Int
            Length of sequence. 
        N_e : Int
            Embedding length.
        N_h : Int
            Number of attention heads.

        For more information, please refer to this paper https://arxiv.org/abs/1706.03762.
        """
        super(Attention, self).__init__()
        
        self.N_s=N_s
        self.N_h=N_h
        self.N_e=N_e

        

        self.softmax=nn.Softmax(3)
        self.fc=nn.Linear(N_e, N_e)
        self.f_scale=1/np.sqrt(N_e/N_h)
        self.l_Q=nn.Linear(N_e, N_e)
        self.l_K=nn.Linear(N_e, N_e)
        self.l_V=nn.Linear(N_e, N_e)
            
        
        
    def forward(self, Q, K, V):
        """
        The dimension of Q,K,V are N x N_s x N_h x N_e/N_h
        
        N=batch size
        N_s=length of signal
        N_h=number of heads
        N_e=embedding dimension

        Parameters
        ----------
        Q : torch.tensor
            Query tensor.
        K : torch.tensor
            Key tensor.
        V : torch.tensor
            Value tensor.

        For more information, please refer to this paper https://arxiv.org/abs/1706.03762.
        """
        
        Q=self.l_Q(Q).view(-1, self.N_s, self.N_h, int(self.N_e/self.N_h))
        K=self.l_K(K).view(-1, self.N_s, self.N_h, int(self.N_e/self.N_h))
        V=self.l_V(V).view(-1, self.N_s, self.N_h, int(self.N_e/self.N_h))
        
        
        # transform Key tensor to Nx N_h x N_e/N_h x N_s
        k=K.transpose(1,3).transpose(1,2)
        
        # transform Query tensor to N x N_h x N_s x N_e/N_h
        q=Q.transpose(1,2)
        
        # transform Value tensor to N x N_h x N_s x N_e/N_h
        v=V.transpose(1,2)
        
        # dimension N x N_h x N_s x N_s
        x=torch.matmul(q,k)
        x=x*self.f_scale
        
        x=self.softmax(x)
        
        # dimension N x N_h x N_s x N_e/N_h
        x=torch.matmul(x, v)
        
        # dimension N x N_s x N_h x N_e/N_h
        x=x.transpose(1,2)
        
        # dimension N x N_s x N_e
        x=x.flatten(2,3)
        
        # dimension N x N_s x N_e
        x=self.fc(x)
        
        return x
    
    
    
class TransformerBlock(nn.Module):
    def __init__(self, N_s, N_e, N_h):
        """
        

        Parameters
        ----------
        N_s : Int
            Length of sequence. 
        N_e : Int
            Embedding length.
        N_h : Int
            Number of attention heads.

        For more information, please refer to this paper https://arxiv.org/abs/1706.03762.
        """
        super(TransformerBlock, self).__init__()
        
        self.attention_1=Attention(N_s, N_e, N_h)
    
        self.norm1=nn.LayerNorm(N_e) 
        self.norm2=nn.LayerNorm(N_e) 
        
        self.batchnorm1 = nn.BatchNorm2d(N_e)
        self.batchnorm2 = nn.BatchNorm2d(N_e)
        
        self.adaptivenorm1 = AdaptiveNorm((1,N_s,1))
        self.adaptivenorm2 = AdaptiveNorm((1,N_s,1))
        
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        
        self.l1=nn.Linear(N_e, 2*N_e)
        self.l2=nn.Linear(2*N_e, N_e)
        self.relu=nn.ReLU()
        
        
    def forward(self, x):
        
        res=x
        x=self.attention_1(x, x, x)
        x=x+res
        
        x=self.adaptivenorm1(x)
        x=self.dropout1(x)
        res2=x
        
        x=self.l1(x)
        x=self.relu(x)
        x=self.l2(x)
        
        x=x+res2
        
        x=self.adaptivenorm2(x)
        x=self.dropout2(x)
        
        return res
    
class CustomTransformer(nn.Module):
    def __init__(self, N_v, N_s, N_e, N_h):
        """
        

        Parameters
        ----------
        N_v : Int
            Vocabulary size.
        N_s : Int
            Length of sequence. 
        N_e : Int
            Embedding length.
        N_h : Int
            Number of attention heads.

        For more information, please refer to this paper https://arxiv.org/abs/1706.03762.
        """
        super(CustomTransformer, self).__init__()
        
    
        self.embd = nn.Embedding(N_v, N_e, max_norm=1)
        self.pos=PositionalEncoding(N_e, N_s)
        
        self.e1=TransformerBlock(N_s, N_e, N_h)
        self.e2=TransformerBlock(N_s, N_e, N_h)
        self.e3=TransformerBlock(N_s, N_e, N_h)
        self.e4=TransformerBlock(N_s, N_e, N_h)
        
        self.fc2=nn.Linear(N_e, 20)
        self.fc=nn.Linear(N_s*20, 5)
        self.softmax=nn.Softmax(1)

        self.adaptivenorm1 = AdaptiveNorm((1,N_s,1))
        self.adaptivenorm2 = AdaptiveNorm((1,N_s,1))
        self.adaptivenorm3 = AdaptiveNorm((1,N_s,1))
        
        self.dropout1 = nn.Dropout(0.2)

        
    def forward(self, x):


        x=self.embd(x)
        x=self.pos(x)
        
        
        x=self.e1(x)
        x=self.adaptivenorm1(x)

        x=self.e2(x)
        x=self.adaptivenorm2(x)

        x=self.e3(x)
        x=self.adaptivenorm3(x)

        x=self.e4(x)
        
        x=self.fc2(x)
        x=self.dropout1(x)
        x=x.flatten(1,2)
        x=self.fc(x)
    
        x=self.softmax(x)
        return x