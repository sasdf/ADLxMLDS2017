#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm, trange
import itertools as it
from collections import Counter
from Net import TaggerNet
from utils import toList, createVariable
from Metrics import Average
from logger import logging

class TaggerModel(object):
    def __init__(self, *args, **kwargs):
        self.model = TaggerNet(*args, **kwargs)
        self.optimizer = optim.Adam((p for p in self.model.parameters() if p.requires_grad), lr=2e-4, betas=(0.5, 0.999))
        self.smooth = 1e-2
        self.use_cuda = False

    def fit(self, dataset):
        self.model.train()
        bar = tqdm(dataset, smoothing=0)
        avgLoss = Average('Loss', num=20)
        acc = Average('TAcc')
        for i, (x, y) in enumerate(bar):
            x, y = [createVariable(z, self.use_cuda) for z in [x, y]]

            prob = self.model(x)

            loss = F.cross_entropy(prob, y)
            avgLoss.append(toList(loss)[0])

            pred = torch.max(prob.data, 1)[1]
            corr = (pred == y.data).sum()
            total = y.size(0)
            acc.append(corr / total)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 10)
            self.optimizer.step()

            logs = logging((avgLoss, acc))
            bar.desc = logs

        bar.close()
        return [avgLoss, acc]

    def score(self, dataset):
        self.model.eval()
        bar = tqdm(dataset, smoothing=0)
        acc = Average('Acc')
        for i, (x, y) in enumerate(bar):
            x, y = [createVariable(z, self.use_cuda) for z in [x, y]]

            prob = self.model(x)

            pred = torch.max(prob.data, 1)[1]
            corr = (pred == y.data).sum()
            total = y.size(0)
            acc.append(corr / total)

            logs = logging((acc,))
            bar.desc = logs

        bar.close()
        return [acc]

    def predict(self, dataset):
        self.model.eval()
        bar = tqdm(dataset, smoothing=0)
        r = []
        for i, (x, y) in enumerate(bar):
            x, y = [createVariable(z, self.use_cuda) for z in [x, y]]

            prob = self.model(x)

            prob = F.softmax(prob)
            conf, pred = torch.max(prob, 1)
            r.extend(zip(x.data.cpu().numpy(), toList(pred), toList(y), toList(conf)))
        bar.close()
        return r

    def cuda(self):
        self.use_cuda = True
        self.model = self.model.cuda()
        return self

    def cpu(self):
        self.use_cuda = False
        self.model = self.model.cpu()
        return self
