#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0


def trim_phone(x):
    r = []
    l = -1
    c = 0
    for p in x:
        if p != l:
            l = p
            c = 0
        else:
            c += 1
            if c == 1:
                r.append(p)
    return r
