#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import torch
from torch import optim
from torch.utils.data import DataLoader
from Dataset import AnimeDataset
from SplitSampler import SplitSampler
import numpy as np
from tqdm import *
import pickle
import itertools as it
import random
import copy
import os
from Model import AnimeModel
from flags import FLAGS
from PIL import Image
from utils import toImage

torch.backends.cudnn.benchmark = True
batch_size=64
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

MODEL = os.getenv('MODEL') or ''

print("[*] Loading dataset")
data = AnimeDataset(FLAGS.dataFolder)
print("[#] Loaded %d data" % len(data))
train = DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn, drop_last=True, shuffle=True)

print("[*] Building model")
model = AnimeModel()
if torch.cuda.is_available():
    model = model.cuda()

print("[*] Start training")
bar = trange(500)
try:
    for epoch in bar:
        pred = model.predict([0.5]*10)
        pred = sorted(pred, key=lambda p: p[1])
        imgs, r, t = zip(*pred)
        for i, x in enumerate(imgs):
            img, org = toImage(x)
            try:
                img.save(os.path.join(MODEL, 'output', 'norm', '%d-%d.jpg' % (epoch, i)))
                org.save(os.path.join(MODEL, 'output', 'orig', '%d-%d.jpg' % (epoch, i)))
            except:
                pass
        logs = model.fit(train)
        logs = ', '.join(map(str, logs))
        tqdm.write("[Epoch {:3d}] {}".format(epoch, logs))
        with open(os.path.join(FLAGS.dataFolder, 'ocomf2', 'Generator-%d.pt' % epoch), 'wb') as f:
            torch.save(model.G, f)
        #  with open(os.path.join(FLAGS.dataFolder, 'training', 'Discriminator-%d.pt' % epoch), 'wb') as f:
            #  torch.save(model.D, f)
except KeyboardInterrupt:
    pass
bar.close()

with open(os.path.join(MODEL, 'output', 'Generator.pt'), 'wb') as f:
    torch.save(model.G, f)
