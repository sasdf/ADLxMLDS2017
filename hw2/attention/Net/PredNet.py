#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import torch
from torch import nn
from torch.nn import functional as F
from Components import StackedRnn, Stack, Residual, LastDim, last_dim
from torch.autograd import Variable
import random


hdim = 128
idim = 4096
edim = 300
mdim = 128
nrnn = 1
rnnopt = hdim * 2
bi = False
ndir = 2 if bi else 1
nhid = 2
encopt = nrnn * hdim * nhid

seed = 42
def rand():
    global seed
    seed = (seed * 42871) % 0x7fffffff
    return seed / 0x7fffffff

class PredNet(nn.Module):
    def __init__(self, embedding, dropout=0, padding=False):
        self.odim = odim = len(embedding)
        super().__init__()
        self.padding = padding

        self.attV = LastDim(
                nn.Linear(idim, hdim),
                nn.Dropout(dropout),
                )

        self.attH = LastDim(
                nn.Linear(hdim, hdim),
                nn.Dropout(dropout),
                )

        self.attOut = LastDim(
                nn.Linear(hdim, 1),
                nn.Dropout(dropout),
                )

        self.ctx = LastDim(
                nn.Linear(hdim, hdim),
                nn.Dropout(dropout),
                nn.SELU(),
                )

        self.emb = nn.Embedding(odim, edim)
        self.emb.weight.data = torch.from_numpy(embedding).float()
        self.emb.weight.requires_grad = False

        self.encoder = nn.Sequential(
                nn.BatchNorm1d(idim),
                Stack(2, lambda i: (
                    nn.Linear(idim, idim),
                    nn.BatchNorm1d(idim),
                    nn.Dropout(dropout),
                    nn.SELU(),
                    )),
                nn.Linear(idim, encopt),
                nn.BatchNorm1d(encopt),
                nn.Dropout(dropout),
                nn.Tanh(),
                )

        self.decoder = nn.LSTM(edim, hdim, nrnn, dropout=dropout,
                bidirectional=False)

        opt = hdim + hdim
        self.out = LastDim(
                nn.BatchNorm1d(opt),
                Stack(3, lambda i: (
                    nn.Linear(hdim if i else opt, hdim),
                    nn.BatchNorm1d(hdim),
                    nn.Dropout(dropout),
                    nn.SELU(),
                    )),
                nn.Linear(hdim, odim),
                nn.BatchNorm1d(odim),
                )
        self.cnt = 0

    def forward(self, *args, **kwargs):
        self.cnt += 1
        #  r = random.random()
        s = max(50 * 50 - self.cnt + 30 * 50, 0) / (50 * 50) * 0.01 + 0.99
        #  if (True or self.cnt < 30 * 50 or r < s) and self.training:
            #  return self.forward_teach(*args, **kwargs)
        #  elif self.training:
            #  return self.forward_ss(*args, **kwargs)
        #  else:
            #  return self.forward_eval(*args, **kwargs)
        r = rand()
        if (True or self.cnt < 30 * 50 or r < s) and self.training:
            return self.forward_teach(*args, **kwargs)
        else:
            return self.forward_eval(*args, **kwargs)

    def forward_teach(self, v, ol, y):
        self.train()
        x = v.mean(1)
        y = y.transpose(0, 1)

        v = self.attV(v)
        v = v.unsqueeze(1).expand(v.size(0), y.size(0), v.size(1), v.size(2)).contiguous()
        # v (batch, dec_len, enc_len, hidden)

        h = self.encoder(x)
        # h (batch, nrnn * hidden_size)
        h = h.view(-1, nrnn, hdim * nhid).transpose(0, 1).contiguous()
        h = [t.contiguous() for t in torch.chunk(h, 2, 2)]
        # h (num_layers * num_directions, batch, hidden_size) * nhid
        
        if x.is_cuda:
            p = Variable(torch.LongTensor(1, x.size(0)).cuda().fill_(1))
        else:
            p = Variable(torch.LongTensor(1, x.size(0)).fill_(1))
        p = torch.cat([p, y[:-1,:]], 0)
        # p (dec_len, batch)
        e = self.emb(p)
        p, h = self.decoder(e, h)

        p = p.permute(1, 0, 2).contiguous()
        # p (batch, dec_len, hidden)
        u = self.attH(p)
        u = u.unsqueeze(2).expand(u.size(0), u.size(1), v.size(2), u.size(2)).contiguous()
        # u (batch, dec_len, enc_len, hidden)
        
        a = self.attOut(F.tanh(u + v)).squeeze(-1)
        # a (batch, dec_len, enc_len)
        a = last_dim(F.softmax, a).view(-1, 1, a.size(-1))
        # a (batch * dec_len, 1, enc_len)
        s = v.size()
        w = v.view(-1, v.size(2), v.size(3))
        c = torch.bmm(a, w).view(s[0], s[1], s[3])
        # c (batch, dec_len, hidden)

        c = self.ctx(c)
        # c (batch, dec_len, hidden)

        #  e = e.transpose(0, 1)
        p = torch.cat([c, p], 2)
        p = self.out(p)
        # p (batch, dec_len, hidden)
        return p

    def forward_ss(self, x, ol, y):
        self.eval()
        y = y.transpose(0, 1)
        if x.is_cuda:
            p = Variable(torch.LongTensor(1, x.size(0)).cuda().fill_(1))
        else:
            p = Variable(torch.LongTensor(1, x.size(0)).fill_(1))
        # p (1, batch, hidden)
        t = torch.cat([p, y[:-1,:]], 0)
        y = []
        lh = None
        for i in range(ol.data[0]):
            if i < t.size(0):
                r = random.random()
                s = max(300 * 50 - self.cnt + 30 * 50, 0) / (300 * 50) * 0.2 + 0.8
                if r < s:
                    p = t[i:i+1]
            p = self.emb(p)
            p, lh = self.lmrnn(p, lh)
            p = p.squeeze(0)
            # p (batch, hidden)
            p = torch.cat([x, e, p, m], 1)
            # p (batch, hidden)
            p = self.out(p)
            y.append(p)
            # p (batch, hidden)
            p = torch.max(p, 1)[1].detach()
            p = p.unsqueeze(0)
        y = torch.stack(y, 1)
        return y

    def forward_eval(self, v, ol, y=None):
        self.eval()
        x = v.mean(1)

        v = self.attV(v)
        # v (batch, enc_len, hidden)

        h = self.encoder(x)
        # h (batch, nrnn * hidden_size)
        h = h.view(-1, nrnn, hdim * nhid).transpose(0, 1).contiguous()
        h = [t.contiguous() for t in torch.chunk(h, 2, 2)]
        # h (num_layers * num_directions, batch, hidden_size) * nhid
        
        if x.is_cuda:
            p = Variable(torch.LongTensor(1, x.size(0)).cuda().fill_(1))
        else:
            p = Variable(torch.LongTensor(1, x.size(0)).fill_(1))
        # p (1, batch)
        y = []
        h2 = None
        accu = Variable(torch.ones([x.size(0), self.odim]))
        if x.is_cuda:
            accu = accu.cuda()
        for i in range(ol.data[0]):
            e = self.emb(p)
            p, h = self.decoder(e, h)
            p = p.squeeze(0)
            # d (batch, hidden)

            u = self.attH(p)
            u = u.unsqueeze(1).expand(u.size(0), v.size(1), u.size(1)).contiguous()
            # u (batch, enc_len, hidden)
            
            a = self.attOut(F.tanh(u + v)).squeeze(-1)
            # a (batch, enc_len)
            a = F.softmax(a).unsqueeze(1)
            # a (batch, 1, enc_len)
            c = torch.bmm(a, v).squeeze(1)
            # a (batch, hidden)

            c = self.ctx(c)
            # p (batch, hidden)

            #  e = e.squeeze(0)
            p = torch.cat([c, p], 1)
            p = self.out(p)
            p = p * accu.clone().detach()
            y.append(p)
            # p (batch, hidden)
            p = torch.max(p, 1)[1].detach()
            accu[list(range(accu.size(0))), p.cpu().data.numpy().tolist()] = 0
            accu[0] = 1
            p = p.unsqueeze(0)
        y = torch.stack(y, 1)
        return y
