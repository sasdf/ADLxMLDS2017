#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import re


def toSent(vocab_dict, x):
    p = ''
    for i in x:
        if i not in vocab_dict or i < 2:
            continue
        c = vocab_dict[i]
        if c.startswith('\'') or c in ',.!?)]}' or c == "''":
            p += c
        else:
            p += ' ' + c
        p = re.sub(r'(``|\[|\(|\{) ', r'\1', p)
        p = p.lstrip()
    return p
