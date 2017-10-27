#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import torch
from torch import nn
from torch.nn import functional as F
from Components import StackedRnn


hdim = 40
idim = 39
odim = 40
ksz = 7
nrnn = 4
rnnopt = hdim*2*nrnn + idim
rnnopt = hdim*2
class PredNet(nn.Module):
    def __init__(self, dropout=0, padding=False):
        super().__init__()
        self.padding = padding
        self.conv = nn.Sequential(
                nn.Conv1d(idim, hdim, ksz, padding=ksz//2),
                nn.SELU(),
                nn.BatchNorm1d(hdim),
                )
        self.rnn = StackedRnn(hdim, hdim, nrnn, dropout_rate=dropout, padding=padding,
                concat_layers=False)
        self.out = nn.Sequential(
                nn.BatchNorm1d(rnnopt),
                nn.Linear(rnnopt, odim),
                nn.BatchNorm1d(odim),
                )

    def forward(self, x, l):
        x = x.transpose(-1, -2).contiguous()
        x = self.conv(x)
        x = x.transpose(-1, -2).contiguous()
        x = self.rnn(x, l)
        s = list(x.size())[:-1]
        x = x.view(-1, rnnopt)
        x = self.out(x)
        x = x.view(s + [odim])
        return x