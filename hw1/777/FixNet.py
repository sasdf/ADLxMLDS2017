#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import torch
from torch import nn
from torch.nn import functional as F
from Components import Stack, ResNetCell, GLU


idim = 40
edim = 40
odim = 40
ksz = 7
class FixNet(nn.Module):
    def __init__(self, padding=False, dropout=0):
        super().__init__()
        self.padding = padding
        self.emb = nn.Embedding(edim, edim)
        self.conv = nn.Sequential(
                nn.Conv1d(idim, odim, ksz, padding=ksz//2),
                nn.SELU(),
                nn.BatchNorm1d(odim),
                Stack(9, lambda i:
                    GLU(ksz, odim, dropout=dropout)),
                )

    def forward(self, c, x, l):
        x = self.emb(x)
        #  x = x * c.unsqueeze(-1)
        #  x = torch.cat((x, c.unsqueeze(-1)), 2)
        x = x.transpose(-1, -2)
        x = x.contiguous()
        x = self.conv(x)
        x = x.transpose(-1, -2)
        x = x.contiguous()
        return x
