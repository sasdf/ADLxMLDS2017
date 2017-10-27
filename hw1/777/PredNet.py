#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import torch
from torch import nn
from torch.nn import functional as F
from Components import Stack, ResNetCell, GLU, Bottleneck


hdim = 64
idim = 39
odim = 40
ksz = 7
class PredNet(nn.Module):
    def __init__(self, dropout=0, padding=False):
        super().__init__()
        self.padding = padding
        self.conv = nn.Sequential(
                nn.Conv1d(idim, hdim, ksz, padding=ksz//2),
                nn.SELU(),
                nn.BatchNorm1d(hdim),
                Stack(9, lambda i:
                    ResNetCell(ksz, hdim, dropout)),
                nn.Conv1d(hdim, odim, ksz, padding=ksz//2),
                )
        self.emb = nn.Linear(odim, odim)
        self.fixlayers = nn.ModuleList()
        for i in range(3):
            fix = nn.Sequential(
                    Stack(3, lambda i:
                        ResNetCell(ksz, odim, dropout)),
                    )
            self.fixlayers.append(fix)

    def forward(self, x, l):
        y = []
        x = x.transpose(-1, -2).contiguous()
        x = self.conv(x)
        x = x.transpose(-1, -2).contiguous()
        y.append(x)

        for fix in self.fixlayers:
            r = x
            x = F.softmax(x)
            s = x.size()
            x = self.emb(x.view(-1, odim)).view(s)
            x = x.transpose(-1, -2).contiguous()
            x = fix(x)
            x = x.transpose(-1, -2).contiguous()
            x = (r + x) /  1.414
            y.append(x)

        return y
