#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0


def trim_phone(x, c=None):
    r = []
    l = -1
    n = 0
    for i, p in enumerate(x):
        if c and c[i] < 0.5:
            continue
        if p != l:
            l = p
            n = 0
        else:
            n += 1
            if n == 1:
                r.append(p)
    return r
