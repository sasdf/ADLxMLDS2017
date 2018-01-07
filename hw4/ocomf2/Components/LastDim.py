#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

from torch import nn

def last_dim(op, x):
    s = x.size()
    x = x.contiguous().view(-1, s[-1]).contiguous()
    y = op(x)
    y = y.view(s[:-1] + (y.size(-1),)).contiguous()
    return y

class LastDim(nn.Sequential):
    def forward(self, x):
        return last_dim(super().forward, x)
