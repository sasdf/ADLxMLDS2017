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
from Net import Generator, Discriminator
from utils import createVariable, toList
from Metrics import Average
from logger import logging
import random
from PIL import Image
import os
import numpy as np
from utils import toImage

noiseDim = 128

class AnimeModel(object):
    def __init__(self, *args, **kwargs):
        self.G = Generator()
        self.D = Discriminator()
        #  self.optimG = optim.RMSprop((p for p in self.G.parameters() if p.requires_grad), lr=2e-3)
        self.optimG = optim.Adam((p for p in self.G.parameters() if p.requires_grad), lr=2e-4, betas=(0.5, 0.999))
        self.optimD = optim.Adam((p for p in self.D.parameters() if p.requires_grad), lr=2e-4, betas=(0.5, 0.999))
        self.smooth = 1e-2
        self.use_cuda = False
        self.step = 0
        self.memory = []

    def fit(self, dataset):
        bar = tqdm(dataset, smoothing=0)
        avgDLoss = Average('DL', num=4)
        realRealAcc = Average('DR', num=4)
        avgGLoss = Average('GL', num=4)
        fakeRealAcc = Average('GR', num=4)
        realIlluAcc = Average('TI', num=4)
        fakeIlluAcc = Average('GI', num=4)
        for i, (x, y) in enumerate(bar):
            self.step += 1
            batchSZ = y.size(0)
            x, y = [createVariable(z, self.use_cuda) for z in [x, y]]
            true = createVariable(torch.ones(batchSZ).float(), self.use_cuda)
            false = createVariable(torch.zeros(batchSZ).float(), self.use_cuda)

            # lr decay
            if self.step % 50000 == 0:
                for param_group in self.optimD.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5
                for param_group in self.optimG.param_groups:
                    param_group['lr'] = param_group['lr'] * 0.5

            #  tagger pretrain
            #  if self.step < 4000:
                #  self.G.eval()
                #  self.D.train()
                #  self.optimD.zero_grad()
                #  dloss = 0

                #  # Real data
                #  isReal, tags = self.D(x)
                #  lossHair = F.cross_entropy(tags[:, 0, :], y[:, 0])
                #  lossEyes = F.cross_entropy(tags[:, 1, :], y[:, 1])
                #  realHairAcc.append(toList((torch.max(tags[:, 0, :], 1)[1] == y[:, 0]).sum())[0] / batchSZ)
                #  realEyesAcc.append(toList((torch.max(tags[:, 1, :], 1)[1] == y[:, 1]).sum())[0] / batchSZ)
                #  lossRealTags = lossHair * 0.6 + lossEyes
                #  loss = lossRealTags
                #  dloss += loss.data.cpu().numpy().tolist()[0]
                #  loss.backward()

                #  # Gradient penalty
                #  alpha = createVariable(torch.rand(batchSZ, 1, 1, 1), self.use_cuda) 
                #  beta = createVariable(torch.randn(x.size()), self.use_cuda) 
                #  gradientPenalty = 0

                #  x = alpha * x + (1 - alpha) * (x + 0.5 * x.std() * beta)
                #  x = x.detach()
                #  x.requires_grad = True
                #  isReal, tags = self.D(x)
                #  hair = tags[:,0,:12]
                #  eyes = tags[:,1,:11]

                #  hairGrad = createVariable(torch.ones(batchSZ, 12).float(), self.use_cuda)
                #  hairGrad = grad(hair, x, hairGrad, create_graph=True,
                        #  retain_graph=True, only_inputs=True)[0].view(batchSZ, -1)
                #  gradientPenalty += ((hairGrad.norm(p=2, dim=1) - 1)**2).mean()

                #  eyesGrad = createVariable(torch.ones(batchSZ, 11).float(), self.use_cuda)
                #  eyesGrad = grad(eyes, x, eyesGrad, create_graph=True,
                        #  retain_graph=True, only_inputs=True)[0].view(batchSZ, -1)
                #  gradientPenalty += ((eyesGrad.norm(p=2, dim=1) - 1)**2).mean()

                #  gradientPenalty *= 0.5
                #  dloss += gradientPenalty.data.cpu().numpy().tolist()[0]
                #  gradientPenalty.backward()

                #  avgDLoss.append(dloss)
                #  torch.nn.utils.clip_grad_norm(self.D.parameters(), 1)
                #  self.optimD.step()
                #  logs = logging((avgDLoss, avgGLoss, realRealAcc, fakeRealAcc, realHairAcc, fakeHairAcc, realEyesAcc, fakeEyesAcc))
                #  bar.desc = logs
                #  continue


            lambdaAdvMax = 1
            #  lambdaAdv = min(1, self.step / 4000) ** 2
            #  lambdaAdv = lambdaAdv * 0.8 + 0.2
            #  lambdaAdv = lambdaAdv * lambdaAdvMax
            lambdaAdv= lambdaAdvMax

            skipD = False

            if lambdaAdv >= lambdaAdvMax - 1e-10:
                # gap skip
                gap = max(realRealAcc.value() - fakeRealAcc.value(), 0)
                gap = min(1, gap * 2)
                r = random.random()
                if r > 1 - gap * 0.9:
                    skipD = True
                pass

            if not skipD:
                for _ in range(1):
                    # Training Discriminator
                    self.G.eval()
                    self.D.train()
                    self.optimD.zero_grad()
                    self.optimG.zero_grad()
                    dloss = 0

                    # Real data
                    isReal, illum = self.D(x)
                    lossRealLabel = F.binary_cross_entropy_with_logits(isReal, true)
                    realRealAcc.append(toList(F.sigmoid(isReal).mean())[0])
                    lossIllu = F.mse_loss(illum, y)
                    realIlluAcc.append(toList(lossIllu)[0])
                    loss = lossRealLabel * lambdaAdv + lossIllu
                    dloss += loss.data.cpu().numpy().tolist()[0]
                    loss.backward()


                    # Gradient penalty
                    alpha = createVariable(torch.rand(batchSZ, 1, 1, 1), self.use_cuda) 
                    beta = createVariable(torch.randn(x.size()), self.use_cuda) 
                    gradientPenalty = 0

                    x = alpha * x + (1 - alpha) * (x + 0.5 * x.std() * beta)
                    x = x.detach()
                    x.requires_grad = True
                    isReal, illum = self.D(x)
                    #  isReal = F.sigmoid(isReal)

                    realGrad = grad(isReal, x, true, create_graph=True,
                            retain_graph=True, only_inputs=True)[0].view(batchSZ, -1)
                    gradientPenalty += ((realGrad.norm(p=2, dim=1) - 1)**2).mean()

                    gradientPenalty *= 0.5
                    dloss += gradientPenalty.data.cpu().numpy().tolist()[0]
                    gradientPenalty.backward()


                    # Fake data
                    noise = createVariable(torch.randn(batchSZ, noiseDim), self.use_cuda)
                    illum = createVariable(torch.FloatTensor(batchSZ).uniform_(0.3, 1), self.use_cuda)

                    x = self.G(noise, illum)
                    #  x = torch.clamp(x, 0, 1)
                    x = x.detach()

                    isReal, illum = self.D(x)
                    lossRealLabel = F.binary_cross_entropy_with_logits(isReal, false)

                    loss = lossRealLabel * lambdaAdv
                    loss = loss / 2
                    dloss += loss.data.cpu().numpy().tolist()[0]
                    loss.backward()

                    # Fake data history
                    if len(self.memory) > batchSZ:
                        x = random.sample(self.memory, batchSZ)
                        x = createVariable(torch.stack(x, 0), self.use_cuda)

                        isReal, illum = self.D(x)
                        lossRealLabel = F.binary_cross_entropy_with_logits(isReal, false)

                        loss = lossRealLabel * lambdaAdv
                        loss = loss / 2
                        dloss += loss.data.cpu().numpy().tolist()[0]
                        loss.backward()


                    avgDLoss.append(dloss)
                    torch.nn.utils.clip_grad_norm(self.D.parameters(), 1)
                    self.optimD.step()

            # Training Generator
            for i in range(1):
                self.optimD.zero_grad()
                self.optimG.zero_grad()
                self.D.eval()
                self.G.train()
                noise = createVariable(torch.randn(batchSZ, noiseDim), self.use_cuda)
                illum = createVariable(torch.FloatTensor(batchSZ).uniform_(0.3, 1), self.use_cuda)
                gloss = 0

                x = self.G(noise, illum)
                isReal, _illum = self.D(x)

                self.memory.append(x[0].data.cpu())
                if len(self.memory) > 65535: self.memory = self.memory[-65535:]

                if self.step % 15 == 0 and i == 0:
                    img = x.data[0].cpu().numpy()
                    img, org = toImage(img)
                    img.save(os.path.join('output', 'training', 'norm', '%d-0.jpg' % (self.step)))
                    org.save(os.path.join('output', 'training', 'orig', '%d-0.jpg' % (self.step)))

                lossRealLabel = F.binary_cross_entropy_with_logits(isReal, true)
                fakeRealAcc.append(toList(F.sigmoid(isReal).mean())[0])

                lossIllu = F.mse_loss(_illum, illum)
                fakeIlluAcc.append(toList(lossIllu)[0])
                loss = lossRealLabel * lambdaAdv + lossIllu
                gloss += loss.data.cpu().numpy().tolist()[0]
                loss.backward()

                avgGLoss.append(gloss)
                torch.nn.utils.clip_grad_norm(self.G.parameters(), 1)
                self.optimG.step()

            logs = logging((avgDLoss, avgGLoss, realRealAcc, fakeRealAcc,
                realIlluAcc, fakeIlluAcc))
            bar.desc = logs

        bar.close()
        return [avgDLoss, avgGLoss, realRealAcc, fakeRealAcc,
                realIlluAcc, fakeIlluAcc]

    def predict(self, illums):
        # FIXME:
        self.G.train()
        self.D.eval()
        bar = tqdm(illums, smoothing=0)
        r = []
        for i, illum in enumerate(bar):
            illum = torch.FloatTensor([illum])

            # Training Generator
            noise = createVariable(torch.randn(1, noiseDim), self.use_cuda, True)
            illum = createVariable(illum, self.use_cuda, True)

            x = self.G(noise, illum)
            #  x = torch.clamp(x, 0, 1)
            isReal, illum = self.D(x)
            r.append((x.data.cpu().numpy()[0], toList(isReal)[0], toList(illum)[0]))

        bar.close()
        return r

    def cuda(self):
        self.use_cuda = True
        self.D = self.D.cuda()
        self.G = self.G.cuda()
        return self

    def cpu(self):
        self.use_cuda = False
        self.D = self.D.cpu()
        self.G = self.G.cpu()
        return self
