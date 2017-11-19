#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import torch
from torch import nn
from torch.nn import functional as F
from Components import StackedRnn, Stack, Residual, LastDim
from torch.autograd import Variable
import random


hdim = 128
edim = 300
nrnn = 1
rnnopt = hdim * 1
bi = False
ndir = 2 if bi else 1
nhid = 2

class LmNet(nn.Module):
    def __init__(self, embedding, dropout=0, padding=False):
        odim = len(embedding)
        super().__init__()
        self.padding = padding

        self.emb = nn.Embedding(odim, edim)
        self.emb.weight.data = torch.from_numpy(embedding).float()
        self.emb.weight.requires_grad = False

        self.lmrnn = nn.LSTM(edim, hdim, nrnn, dropout=dropout,
                bidirectional=False)

        self.out = LastDim(
                #  nn.BatchNorm1d(rnnopt),
                Stack(3, lambda i: (
                    nn.Linear(hdim if i else rnnopt, hdim),
                    nn.BatchNorm1d(hdim),
                    nn.Dropout(dropout),
                    nn.SELU(),
                    )),
                nn.Linear(hdim, odim),
                #  nn.BatchNorm1d(odim),
                )

    def forward(self, x, y):
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)
        p = self.emb(x)
        p, h = self.lmrnn(p)
        p = p.permute(1, 0, 2).contiguous()
        # p (batch, dec_len, hidden)
        s = list(p.size())
        p = p.view(-1, s[-1])
        p = self.out(p)
        p = p.view(s[:-1]+[p.size(-1)])
        # p (batch, dec_len, hidden)
        return p
