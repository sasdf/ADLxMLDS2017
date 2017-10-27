#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import torch
from torch import nn, LongTensor
from torch.nn import functional as F
import numpy as np


class GLU(nn.Module):
    def __init__(self, ksz, dim):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv1d(dim, dim * 2, ksz, padding=ksz//2),
                nn.BatchNorm1d(dim * 2),
                )
        self.nor = nn.BatchNorm1d(dim)
    def forward(self, i):
        x = self.conv(i)
        x, g = torch.chunk(x, 2, dim=1)
        x = x * F.sigmoid(g)
        x = self.nor(x)
        return (i + x) / np.sqrt(2)
