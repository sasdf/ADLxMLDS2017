#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

from torch import nn
from . import GLU


class GatedResNet(nn.Module):
    def __init__(self, ksz, dim, nlayers):
        super().__init__()
        layers = []
        for i in range(nlayers):
            layers.append(GLU(ksz, dim))

        self.layer = nn.Sequential(*layers)
        self.forward = self.layer.forward
