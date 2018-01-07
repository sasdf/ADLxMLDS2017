#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

from torch import nn
from torch.nn import functional as F


class ForceDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, x):
        if self.dropout > 0:
            return F.dropout(x, p=self.dropout, training=True)
        return x
