#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import torch
from torch import optim
from torch.utils.data import DataLoader
from Dataset import ColorDataset
from SplitSampler import SplitSampler
import numpy as np
from tqdm import *
import pickle
import itertools as it
import random
import copy
import os
from Model import ColorModel
from flags import FLAGS
from PIL import Image
from utils import toImage

torch.backends.cudnn.benchmark = True
batch_size=24
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

MODEL = os.getenv('MODEL') or ''

print("[*] Loading dataset")
data = ColorDataset(FLAGS.dataFolder)
print("[#] Loaded %d data" % len(data))
train = DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn, drop_last=True, shuffle=True)

print("[*] Building model")
model = ColorModel()
if torch.cuda.is_available():
    model = model.cuda()

print("[*] Start training")
bar = trange(500)
try:
    for epoch in bar:
        pred = model.predict([[0, 0]]*10, train)
        imgs, = zip(*pred)
        for i, x in enumerate(imgs):
            img, org = toImage(x)
            img.save(os.path.join(MODEL, 'output', 'cnorm', '%d-%d.jpg' % (epoch, i)))
            org.save(os.path.join(MODEL, 'output', 'corig', '%d-%d.jpg' % (epoch, i)))
        logs = model.fit(train)
        logs = ', '.join(map(str, logs))
        tqdm.write("[Epoch {:3d}] {}".format(epoch, logs))
        with open(os.path.join(FLAGS.dataFolder, 'pcom1', 'Colorizer-%d.pt' % epoch), 'wb') as f:
            torch.save(model.G, f)
        #  with open(os.path.join(FLAGS.dataFolder, 'training', 'Discriminator-%d.pt' % epoch), 'wb') as f:
            #  torch.save(model.D, f)
except KeyboardInterrupt:
    pass
bar.close()

with open(os.path.join(MODEL, 'output', 'Colorizer.pt'), 'wb') as f:
    torch.save(model.G, f)
