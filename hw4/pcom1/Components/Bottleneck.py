#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

from torch import nn
from . import Residual


class Bottleneck(nn.Module):
    def __init__(self, layer, dim, btlnek, dropout=0):
        super().__init__()
        self.layer = Residual(
                nn.Conv1d(dim, btlnek, 1, padding=0),
                nn.Dropout(dropout),
                nn.SELU(),
                nn.BatchNorm1d(btlnek),
                layer,
                nn.Conv1d(btlnek, dim, 1, padding=0),
                nn.Dropout(dropout),
                nn.SELU(),
                nn.BatchNorm1d(dim),
                )
        self.forward = self.layer.forward
