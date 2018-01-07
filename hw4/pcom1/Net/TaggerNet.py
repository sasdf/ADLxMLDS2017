#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import torch
from torch import nn
from torch.nn import functional as F
from Components import StackedRnn, Stack, Residual, LastDim, InspectSize, SiLU
from torch.autograd import Variable
import random

channel = [3, 32, 64, 128, 128]
stride  = [0, 1,  1,  1,   1, ]
kernel  = [0, 3,  3,  3,   3, ]
padding = [0, 2,  2,  2,   2, ]
cdim = channel[-1] * 3 * 3
odim = 12
dropout = 0.2

class TaggerNet(nn.Module):
    def __init__(self, dropout=0, tag='hair'):
        super().__init__()
        self.tag = tag

        self.conv = nn.Sequential(
                Stack(len(channel) - 2, lambda i: (
                    nn.Conv2d(channel[i], channel[i+1], kernel[i+1], stride[i+1], padding[i+1]),
                    nn.MaxPool2d(2),
                    nn.Dropout(dropout),
                    nn.LeakyReLU(),
                    #  Stack(2, lambda _: (
                        #  Residual(
                            #  nn.Conv2d(channel[i+1], channel[i+1], 3, padding=1),
                            #  nn.Dropout(dropout),
                            #  nn.LeakyReLU(),
                            #  nn.Conv2d(channel[i+1], channel[i+1], 3, padding=1),
                            #  nn.Dropout(dropout),
                            #  ),
                        #  nn.LeakyReLU(),
                        #  )),
                    )),
                nn.Conv2d(channel[-2], channel[-1], kernel[-1], stride[-1], padding=kernel[-1] // 2),
                nn.AdaptiveMaxPool2d(3),
                nn.Dropout(dropout),
                nn.LeakyReLU(),
                )

        self.out = nn.Sequential(
                nn.Linear(cdim, cdim),
                nn.Dropout(dropout),
                nn.LeakyReLU(),
                nn.Linear(cdim, cdim),
                nn.Dropout(dropout),
                nn.LeakyReLU(),
                nn.Linear(cdim, odim),
                )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)


    def forward(self, x):
        if self.tag == 'hair':
            x = x[:, :, 8:-40, 12:-12]
        elif self.tag == 'eyes':
            x = x[:, :, 8:-40, 12:-12]
        y = self.conv(x)
        y = y.view(y.size(0), -1)
        y = self.out(y)
        return y
