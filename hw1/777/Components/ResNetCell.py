#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

from torch import nn
from . import Residual


class ResNetCell(nn.Module):
    def __init__(self, ksz, dim, dropout=0):
        super().__init__()
        self.layer = Residual(
                nn.Conv1d(dim, dim, ksz, padding=ksz//2),
                nn.Dropout(dropout),
                nn.SELU(),
                nn.BatchNorm1d(dim),
                nn.Conv1d(dim, dim, ksz, padding=ksz//2),
                nn.Dropout(dropout),
                nn.SELU(),
                nn.BatchNorm1d(dim),
                )
        self.forward = self.layer.forward
