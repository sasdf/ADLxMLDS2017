#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

from torch import nn
from . import StackedRnn, ResNet


class QRNN(nn.Module):
    def __init__(self, ksz, dim, nrnn, ncnn, padding=False):
        super().__init__()
        self.rnn = StackedRnn(dim, dim, nrnn, concat_layers=True, padding=padding, dropout_rate=0.0)
        self.nor = nn.BatchNorm1d(dim*2*nrnn+dim)
        self.conv = nn.Sequential(
                nn.Conv1d(dim*2*nrnn+dim, dim, ksz, padding=ksz//2),
                nn.SELU(),
                nn.BatchNorm1d(dim),
                ResNet(ksz, dim, ncnn),
                )

    def forward(self, x):
        m, ml = x
        x = self.rnn(m, ml)
        x = x.transpose(-1, -2)
        x = x.contiguous()
        x = self.nor(x)
        x = self.conv(x)
        x = x.transpose(-1, -2)
        x = x.contiguous()
        return x, ml
