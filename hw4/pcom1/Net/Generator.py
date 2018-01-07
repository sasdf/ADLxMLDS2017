#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import torch
from torch import nn
from torch.nn import functional as F
from Components import StackedRnn, Stack, Residual, LastDim, InspectSize, SiLU, ForceDropout
from torch.autograd import Variable
import random
from utils import createVariable

"""
Input(164)
Linear(65536)
Reshape(64, 32, 32)
Stack(8,
  Residual(
    Conv(64, 3, 1),
    ReLU(),
    Conv(64, 3, 1),
  )
)
# Shape (64, 32, 32)
Stack(3,
  Conv(64, 3, 1),
  ReLU(),
  Unpooling(2, 2)
  Conv(64, 3, 1),
  ReLU(),
)
# Shape (64, 256, 256)
Conv(64, 3, 1),
ReLU(),
# Shape (64, 256, 256)
Conv(3, 1, 1),
# Shape (3, 256, 256)
"""

#  hdim = 8
#  up = [2, 3, 2]
hdim = 24
up = [2, 2]
noiseDim = 128
dropout = 0.5

class Generator(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()

        #  self.noise = nn.Sequential(
                #  nn.Linear(128, 48 * hdim * hdim)
                #  )

        #  self.hair = nn.Embedding(12, 8 * hdim * hdim)
        #  self.eyes = nn.Embedding(11, 8 * hdim * hdim)
        self.inp = nn.Sequential(
                nn.Linear(noiseDim + 1, 64 * hdim * hdim),
                )

        self.conv = nn.Sequential(
                #  nn.BatchNorm2d(64),
                nn.ReLU(),
                Stack(8, lambda i: (
                    Residual(
                        nn.Conv2d(64, 64, 3, padding=1),
                        #  ForceDropout(dropout if i % 2 == 1 else 0),
                        #  nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Conv2d(64, 64, 3, padding=1),
                        #  nn.BatchNorm2d(64),
                        #  ForceDropout(dropout),
                        ),
                    )),
                Stack(len(up), lambda i: (
                    nn.Conv2d(64, 64, 3, padding=1),
                    #  ForceDropout(dropout),
                    #  nn.ReLU(),
                    #  nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Upsample(scale_factor=up[i], mode='nearest'),
                    nn.Conv2d(64, 64, 3, padding=1),
                    #  nn.BatchNorm2d(64),
                    #  ForceDropout(dropout),
                    #  nn.ReLU(),
                    nn.ReLU(),
                    )),
                nn.Conv2d(64, 64, 3, padding=1),
                #  nn.BatchNorm2d(64),
                #  ForceDropout(dropout),
                #  nn.ReLU(),
                nn.ReLU(),
                nn.Conv2d(64, 1, 1),
                #  nn.Sigmoid(),
                #  nn.Tanh(),
                )

        #  for p in self.inp.parameters():
            #  p.requires_grad = False

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)


    def forward(self, noise, illum):
        illum = illum.unsqueeze(1)
        x = torch.cat([noise, illum], 1)
        x = self.inp(x).view(x.size(0), 64, hdim, hdim)

        #  noise = self.noise(noise).view(noise.size(0), 48, hdim, hdim)
        #  hair = self.hair(hair).view(hair.size(0), 8, hdim, hdim)
        #  eyes = self.eyes(eyes).view(eyes.size(0), 8, hdim, hdim)
        #  x = torch.cat([noise, hair, eyes], 1)
        y = self.conv(x)
        return y
