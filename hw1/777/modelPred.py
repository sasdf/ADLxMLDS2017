#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import torch
from torch.autograd import Variable
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm, trange
import editdistance
from utils import trim_phone


class ModelPred(object):
    def __init__(self, net, optimizer=optim.Adam, *args, **kwargs):
        self.model = net(*args, **kwargs)
        self.model = self.model.cuda()
        self.optimizer = optimizer((p for p in self.model.parameters() if p.requires_grad))
        self.smooth = 1e-2

    def fit(self, dataset):
        self.optimizer.zero_grad()
        self.model.train()
        bar = tqdm(dataset, smoothing=0)
        avg_loss = None
        correct = 0
        total = 0
        for i, (x, l, y, n) in enumerate(bar):
            x, l, y = [Variable(z).cuda() for z in [x, l, y]]

            probs = self.model(x, l)
            loss = 0

            y = y.view(-1)
            for i, prob in enumerate(probs):
                prob = prob.view(-1, prob.size(-1))
                loss += F.cross_entropy(prob, y) * (i+1)

            prob = probs[-1]
            prob = prob.view(-1, prob.size(-1))
            pred = torch.max(prob.data, 1)[1]
            corr = ((pred == y.data) * (y.data != 0)).sum()
            correct += corr
            total += (y.data != 0).sum()

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
        ed = 0
        et = 0
        for i, (x, l, y, n) in enumerate(bar):
            x, l, y = [Variable(z, volatile=True).cuda() for z in [x, l, y]]

            probs = self.model(x, l)

            prob = probs[-1]
            s = prob.size()
            prob = prob.view(-1, prob.size(-1))
            prob = F.softmax(prob).view(s)
            conf, pred = torch.max(prob.data, 2)
            corr = ((pred == y.data) * (y.data != 0)).sum()
            correct += corr
            total += (y.data != 0).sum()

            for ci, xi, yi, li in zip(conf, pred, y.squeeze(-1).data.cpu().numpy(), l.data.cpu().numpy()):
                ci, xi, yi = ci[:li].tolist(), xi[:li].tolist(), yi[:li].tolist()
                xi = trim_phone(xi, ci)
                yi = [yi[i] for i in range(li) if i == 0 or yi[i-1] != yi[i]]
                ed += editdistance.eval(xi, yi)
                et += 1

            bar.desc = "Acc: {:6.3f}, Dis: {:6.3f}".format(correct / total * 100, ed / et)

        bar.close()
        return (correct / total, ed / et)

    def predict(self, dataset):
        self.model.eval()
        bar = tqdm(dataset, smoothing=0)
        r = []
        for i, (x, l, y, n) in enumerate(bar):
            x, l = [Variable(z, volatile=True).cuda() for z in [x, l]]

            probs = self.model(x, l)

            prob = probs[-1]
            s = prob.size()
            prob = prob.view(-1, prob.size(-1))
            prob = F.softmax(prob).view(s)
            conf, pred = torch.max(prob.data, 2)
            for ci, xi, yi, li, ni in zip(conf, pred, y.squeeze(-1).numpy(), l.data.cpu().numpy(), n):
                ci, xi, yi = ci[:li].tolist(), xi[:li].tolist(), yi[:li].tolist()
                r.append((ci, xi, yi, ni))
        bar.close()
        return r

    def predict_proba(self, dataset):
        self.model.eval()
        bar = tqdm(dataset, smoothing=0)
        r = []
        for i, (x, l, y, n) in enumerate(bar):
            x, l = [Variable(z, volatile=True).cuda() for z in [x, l]]

            probs = self.model(x, l)

            prob = probs[-1]
            s = prob.size()
            prob = prob.view(-1, prob.size(-1))
            prob = F.softmax(prob).view(s)
            for xi, yi, li, ni in zip(prob.data.cpu().numpy(), y.squeeze(-1).numpy(), l.data.cpu().numpy(), n):
                xi, yi = xi[:li].tolist(), yi[:li].tolist()
                r.append((xi, yi, ni))
        bar.close()
        return r