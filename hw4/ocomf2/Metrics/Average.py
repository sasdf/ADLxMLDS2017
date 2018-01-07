#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

from torch import nn


class Average(object):
    def __init__(self, name, num=None):
        super().__init__()
        self.name = name
        self.num = num
        self.data = []

    def value(self):
        if len(self.data) == 0:
            return 0
        return sum(self.data) / len(self.data)

    def append(self, elem):
        self.data.append(elem)
        if self.num is not None and self.num < len(self.data):
            self.data = self.data[-self.num:]

    def extend(self, iters):
        for elem in iters:
            self.append(elem)

    def __repr__(self):
        return '{}: {:3.2f}'.format(self.name, self.value())
