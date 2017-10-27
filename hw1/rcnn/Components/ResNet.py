#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

from torch import nn
from . import Residual


class ResNet(nn.Module):
    def __init__(self, ksz, dim, nlayers, dropout=0):
        super().__init__()
        layers = []
        for i in range(nlayers):
            layers.append(Residual(
                nn.Conv1d(dim, dim, ksz, padding=ksz//2),
                nn.Dropout(dropout),
                nn.SELU(),
                nn.BatchNorm1d(dim),
                nn.Conv1d(dim, dim, ksz, padding=ksz//2),
                nn.Dropout(dropout),
                nn.SELU(),
                nn.BatchNorm1d(dim),
                ))

        self.layer = nn.Sequential(*layers)
        self.forward = self.layer.forward
