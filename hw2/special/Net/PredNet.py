#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import torch
from torch import nn
from torch.nn import functional as F
from Components import StackedRnn, Stack
from torch.autograd import Variable


hdim = 128
idim = 4096
edim = 100
nrnn = 1
rnnopt = hdim + hdim
bi = False
encopt = nrnn * hdim * 2 if bi else nrnn * hdim

class PredNet(nn.Module):
    def __init__(self, odim, dropout=0, padding=False):
        super().__init__()
        self.padding = padding

        self.encoder = nn.LSTM(idim, hdim, nrnn, dropout=dropout,
                bidirectional=bi)
        self.emb = nn.Embedding(odim, edim)
        self.att = nn.Linear(hdim, hdim)
        self.decoder = nn.LSTM(edim, hdim, nrnn, dropout=dropout,
                bidirectional=bi)
        #  self.bridge = nn.Sequential(
                #  nn.BatchNorm1d(encopt),
                #  Stack(2, lambda i: (
                    #  nn.Linear(nrnn * hdim if i else encopt, nrnn * hdim),
                    #  nn.BatchNorm1d(nrnn * hdim),
                    #  nn.SELU(),
                    #  )),
                #  )
        self.out = nn.Sequential(
                nn.BatchNorm1d(rnnopt),
                Stack(2, lambda i: (
                    nn.Linear(hdim if i else rnnopt, hdim),
                    nn.BatchNorm1d(hdim),
                    nn.SELU(),
                    )),
                nn.Linear(hdim, odim),
                nn.BatchNorm1d(odim),
                )

    def forward(self, *args, **kwargs):
        if self.training:
            return self.forward_teach(*args, **kwargs)
        else:
            return self.forward_eval(*args, **kwargs)

    def forward_teach(self, x, ol, y):
        x = x.transpose(0, 1)
        y = y.transpose(0, 1)
        e, h = self.encoder(x)
        # e (enc_len, batch, hidden_size * num_directions)
        # h (num_layers * num_directions, batch, hidden_size)
        
        #  h = h.transpose(0, 1)
        #  s = h.size()
        #  h = self.bridge(h.view(h.size(0), -1)).view(s).transpose(0, 1)

        s = e.size()
        e = self.att(e.view(-1, e.size(-1))).view(s).transpose(0, 1)
        # e (batch, enc_len, hidden)
        if torch.cuda.is_available():
            p = Variable(torch.LongTensor(1, x.size(1)).cuda().fill_(1))
        else:
            p = Variable(torch.LongTensor(1, x.size(1)).fill_(1))
        # p (dec_len, batch)
        p = torch.cat([p, y[:-1,:]], 0)
        p = self.emb(p)
        p, h = self.decoder(p, h)
        p = p.permute(1, 2, 0)
        # p (batch, hidden, dec_len)
        a = torch.bmm(e, p)
        # a (batch, enc_len, dec_len)
        a = torch.bmm(e.transpose(1, 2), a)
        # a (batch, hidden, dec_len)
        p = torch.cat([p, a], 1).transpose(1, 2).contiguous()
        # p (batch, dec_len, hidden)
        s = list(p.size())
        p = p.view(-1, s[-1])
        p = self.out(p)
        p = p.view(s[:-1]+[p.size(-1)])
        # p (batch, dec_len, hidden)
        return p

    def forward_eval(self, x, ol, y=None):
        x = x.transpose(0, 1)
        e, h = self.encoder(x)
        # e (enc_len, batch, hidden_size * num_directions)
        # h (num_layers * num_directions, batch, hidden_size)
        # WTFFFFFFFFFFFFFFFFFFF
        #  h = [hi[:,-1:].contiguous() for hi in h]
        # WTFFFFFFFFFFFFFFFFFFF
        s = e.size()
        e = self.att(e.view(-1, e.size(-1))).view(s).transpose(0, 1)
        # e (batch, enc_len, hidden)
        if torch.cuda.is_available():
            p = Variable(torch.LongTensor(1, x.size(1)).cuda().fill_(1))
        else:
            p = Variable(torch.LongTensor(1, x.size(1)).fill_(1))
        # p (dec_len, batch, hidden)
        y = []
        for i in range(ol.data[0]):
            p = self.emb(p)
            p, h = self.decoder(p, h)
            p = p.permute(1, 2, 0)
            # p (batch, hidden, dec_len)
            a = torch.bmm(e, p)
            # a (batch, enc_len, dec_len)
            a = torch.bmm(e.transpose(1, 2), a)
            # a (batch, hidden, dec_len)
            p = torch.cat([p, a], 1).squeeze(-1)
            # p (batch, hidden)
            p = self.out(p)
            y.append(p)
            # p (batch, hidden)
            p = torch.max(p, 1)[1]
            p = p.unsqueeze(0)
        y = torch.stack(y, 1)
        return y
