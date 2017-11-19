#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import torch
from torch import nn, LongTensor
from torch.nn import functional as F
import numpy as np


class GLU(nn.Module):
    def __init__(self, ksz, dim, dropout=0):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(dim, dim * 2, ksz, dilation=5, padding=ksz//2*5),
                nn.BatchNorm2d(dim * 2),
                )
        self.drop = nn.Dropout(dropout)
        self.nor = nn.BatchNorm2d(dim)

    def forward(self, i):
        x = self.conv(i)
        x = self.drop(x)
        x, g = torch.chunk(x, 2, dim=1)
        x = x * F.sigmoid(g)
        x = self.nor(x)
        return x
