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
from .bleu_eval_new import BLEU

class LmModel(object):
    def __init__(self, net, *args, optimizer=optim.Adam, **kwargs):
        self.model = net(*args, **kwargs)
        self.optimizer = optimizer((p for p in self.model.parameters() if p.requires_grad), lr=1e-2)
        self.smooth = 1e-2
        self.use_cuda = False

    def fit(self, dataset):
        self.optimizer.zero_grad()
        self.model.train()
        bar = tqdm(dataset, smoothing=0)
        avg_loss = None
        correct = 0
        total = 0
        for i, (x, y, l) in enumerate(bar):
            self.model.train()
            x, y, l = [Variable(z) for z in [x, y, l]]
            if self.use_cuda:
                x, y, l = [z.cuda() for z in [x, y, l]]

            prob = self.model(x, y)

            y = y.view(-1)
            prob = prob.view(-1, prob.size(-1))
            loss = F.cross_entropy(prob, y)

            pred = torch.max(prob.data, 1)[1]
            corr = ((pred == y.data) * (y.data >= 2)).sum()
            correct += corr
            total += (y.data >= 2).sum()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(self.model.parameters(), 10)
            self.optimizer.step()

            if avg_loss is None: avg_loss = loss.data[0]
            avg_loss = avg_loss * (1-self.smooth) + loss.data[0] * self.smooth
            bar.desc = "Loss: {:5.4f}, Acc: {:6.3f}".format(avg_loss, correct / total * 100)

        bar.close()
        return (avg_loss, correct / total)

    def score(self, dataset):
        self.model.eval()
        bar = tqdm(dataset, smoothing=0)
        correct = 0
        total = 0
        r = []
        for i, (x, y, l) in enumerate(bar):
            x, y, l = [Variable(z, volatile=True) for z in [x, y, l]]
            if self.use_cuda:
                x, y, l = [z.cuda() for z in [x, y, l]]

            prob = self.model(x, y)

            s = prob.size()
            prob = prob.view(-1, s[-1])
            prob = F.softmax(prob).view(s)
            conf, pred = torch.max(prob.data, 2)
            corr = ((pred == y.data) * (y.data >= 1)).sum()
            correct += corr
            total += (y.data >= 1).sum()

            bar.desc = "Acc: {:6.3f}, BLEU: {:6.3f}".format(correct / total * 100, 0)
        bar.close()

        return (correct / total, correct / total)

    def predict(self, dataset):
        bar = tqdm(dataset, smoothing=0)
        r = []
        for i, (x, y, l, ls) in enumerate(bar):
            x, = [Variable(z, volatile=True) for z in [x,]]
            if self.use_cuda:
                x, = [z.cuda() for z in [x,]]

            ol = Variable(torch.LongTensor([50]))
            prob = noRepeat(self.model(x, ol))

            s = prob.size()
            prob = prob.view(-1, s[-1])
            prob = F.softmax(prob).view(s)
            conf, pred = torch.max(prob.data, 2)

            for xi, ci, lsi in zip(pred, conf.cpu().numpy(), ls):
                xl = 0
                for xl in range(0, len(xi)):
                    if xi[xl] < 2:
                        break
                xi, ci = xi.tolist()[:xl], ci.tolist()[:xl]
                r.append((ci, xi, lsi))
        return r

    def cuda(self):
        self.use_cuda = True
        self.model = self.model.cuda()
        return self

    def cpu(self):
        self.use_cuda = False
        self.model = self.model.cpu()
        return self
