#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import torch
from torch.autograd import Variable, grad
from torch import nn, optim
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm, trange
import itertools as it
from collections import Counter
from Net import Colorizer
from utils import createVariable, toList
from Metrics import Average
from logger import logging
import random
from PIL import Image
import os
import numpy as np
from utils import toImage

class ColorModel(object):
    def __init__(self, *args, **kwargs):
        self.G = Colorizer()
        #  self.optimG = optim.RMSprop((p for p in self.G.parameters() if p.requires_grad), lr=2e-3)
        self.optimG = optim.Adam((p for p in self.G.parameters() if p.requires_grad), lr=6e-4, betas=(0.5, 0.999))
        self.use_cuda = False
        self.step = 0
        self.memory = []

    def fit(self, dataset):
        bar = tqdm(dataset, smoothing=0)
        avgGLoss = Average('GL', num=4)
        for i, (x, y) in enumerate(bar):
            self.step += 1
            batchSZ = y.size(0)
            x, y = [createVariable(z, self.use_cuda) for z in [x, y]]
            true = createVariable(torch.ones(batchSZ).float(), self.use_cuda)
            false = createVariable(torch.zeros(batchSZ).float(), self.use_cuda)


            # b/w
            #  coff = torch.rand(3)
            #  coff /= coff.sum()
            coff = [0.299, 0.587, 0.114]
            #  sign = torch.rand(3)
            #  bw = sum(x[:, i] * coff[i] if sign[i] > 0.5 else (1.0 - x[:, i]) * coff[i] for i in range(3))
            bw = sum(x[:, i] * coff[i] for i in range(3))
            bw = bw.unsqueeze(1)
            c = x


            # lr decay
            if self.step % 10000 == 0:
                for param_group in self.optimG.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

            self.optimG.zero_grad()
            self.G.train()
            gloss = 0
            
            # l1
            x = self.G(y[:, 0], y[:, 1], bw)

            if self.step % 15 == 0:
                imb = bw.data[0].repeat(3, 1, 1)
                img = c.data[0]
                img = torch.cat([x.data[0], img, imb], 1)
                img = img.cpu().numpy()
                img, org = toImage(img)
                img.save(os.path.join('output', 'training', 'cnorm', '%d-0.jpg' % (self.step)))
                org.save(os.path.join('output', 'training', 'corig', '%d-0.jpg' % (self.step)))

            loss = F.mse_loss(x, c)

            gloss += loss.data.cpu().numpy().tolist()[0]
            loss.backward()

            avgGLoss.append(gloss)
            torch.nn.utils.clip_grad_norm(self.G.parameters(), 1)
            self.optimG.step()
            logs = logging((avgGLoss,))
            bar.desc = logs

        bar.close()
        return [avgGLoss,]

    def predict(self, tags, data):
        # FIXME:
        self.G.train()
        bar = tqdm(tags, smoothing=0)
        r = []
        l = 0
        c = []
        while l < len(tags):
            for x, y in data:
                c.append(x)
                l += x.size(0)
                if l >= len(tags):
                    break
        c = torch.cat(c, 0) if len(c) > 1 else c[0]
        bw = c[:, 0] * 0.299 + c[:, 1] * 0.587 + c[:, 2] * 0.114
        bw = bw.unsqueeze(1)
        for (i, tag), bw in zip(enumerate(bar), bw):
            tag = torch.LongTensor(tag).unsqueeze(0)
            bw = bw.unsqueeze(0)

            # Training Generator
            bw = createVariable(bw, self.use_cuda, True)
            hair = createVariable(tag[:, 0], self.use_cuda, True)
            eyes = createVariable(tag[:, 1], self.use_cuda, True)

            x = self.G(hair, eyes, bw)
            #  x = torch.clamp(x, 0, 1)
            r.append((x.data.cpu().numpy()[0],))

        bar.close()
        return r

    def cuda(self):
        self.use_cuda = True
        self.G = self.G.cuda()
        return self

    def cpu(self):
        self.use_cuda = False
        self.G = self.G.cpu()
        return self
