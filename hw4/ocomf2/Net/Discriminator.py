#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import torch
from torch import nn
from torch.nn import functional as F
from Components import StackedRnn, Stack, Residual, LastDim, InspectSize, SiLU
from torch.autograd import Variable
import random
from utils import createVariable

channel = [1, 32, 64, 128, 256, 512, 1024]
stride  = [0, 2,  2,  2,   2,   2,   2   ]
kernel  = [0, 4,  4,  4,   3,   3,   3   ]
padding = [0, 1,  1,  1,   1,   1,   1   ]
cdim = channel[-1] * 2 * 2
odim = 1
dropout = 0.3

class Discriminator(nn.Module):
    def __init__(self, dropout=0):
        super().__init__()

        self.conv = nn.Sequential(
                Stack(len(channel) - 2, lambda i: (
                    nn.Conv2d(channel[i], channel[i+1], kernel[i+1], stride[i+1], padding[i+1]),
                    #  nn.Dropout(dropout),
                    nn.LeakyReLU(),
                    Stack(2, lambda _: (
                        Residual(
                            nn.Conv2d(channel[i+1], channel[i+1], 3, padding=1),
                            #  nn.Dropout(dropout),
                            nn.LeakyReLU(),
                            nn.Conv2d(channel[i+1], channel[i+1], 3, padding=1),
                            #  nn.Dropout(dropout),
                            ),
                        nn.LeakyReLU(),
                        )),
                    )),
                nn.Conv2d(channel[-2], channel[-1], kernel[-1], stride[-1], padding=kernel[-1] // 2),
                #  nn.Dropout(dropout),
                nn.LeakyReLU(),
                )

        self.isReal = nn.Sequential(
                #  nn.Linear(cdim, cdim),
                #  nn.Dropout(dropout),
                #  nn.LeakyReLU(),
                #  nn.Linear(cdim, cdim),
                #  nn.Dropout(dropout),
                #  nn.LeakyReLU(),
                nn.Linear(cdim, 1),
                )

        self.illum = nn.Sequential(
                #  nn.Linear(cdim, cdim),
                #  #  nn.Dropout(dropout),
                #  nn.LeakyReLU(),
                #  nn.Linear(cdim, cdim),
                #  #  nn.Dropout(dropout),
                #  nn.LeakyReLU(),
                nn.Linear(cdim, odim),
                )

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.02)


    def forward(self, x):
        #  n = createVariable(torch.randn(x.size()), x.is_cuda)
        #  n = n * 0.02
        #  x = x + n
        y = self.conv(x)
        y = y.view(y.size(0), -1)
        isReal = self.isReal(y).squeeze(-1)
        illum = self.illum(y).squeeze(1)
        return isReal, illum
