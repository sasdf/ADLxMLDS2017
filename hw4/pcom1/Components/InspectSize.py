#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

from torch import nn


class InspectSize(nn.Module):
    def forward(self, *args):
        print([x.size() for x in args])
        if len(args) > 1:
            return args
        return args[0]
