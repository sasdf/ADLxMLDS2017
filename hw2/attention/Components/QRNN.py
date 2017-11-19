#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

from torch import nn
from . import StackedRnn, ResNetCell


class QRNN(nn.Module):
    def __init__(self, ksz, dim, nrnn, ncnn,
            concat_layers=True, dropout=0, padding=False):
        super().__init__()
        self.rnn = StackedRnn(dim, dim, nrnn, concat_layers=concat_layers,
                padding=padding, dropout_rate=dropout)
        self.nor = nn.BatchNorm1d(dim*2*nrnn+dim)
        self.conv = nn.Sequential(
                nn.Conv1d(dim*2*nrnn+dim, dim, ksz, padding=ksz//2),
                nn.SELU(),
                nn.BatchNorm1d(dim),
                Stack(ncnn, lambda i:
                    ResNetCell(ksz, dim, dropout))
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
