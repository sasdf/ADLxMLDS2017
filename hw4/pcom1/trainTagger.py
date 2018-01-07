#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import torch
from torch import optim
from torch.utils.data import DataLoader
from Dataset.TaggerDataset import TaggerDataset
from SplitSampler import WeightedSplitSampler
import numpy as np
from tqdm import *
import pickle
import itertools as it
import random
import copy
import os
from Model import TaggerModel
from flags import FLAGS

batch_size=64
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

MODEL = os.getenv('MODEL') or ''

print("[*] Load dataset")
data = TaggerDataset(FLAGS.dataFolder, training=True, tag='hair', filename='merge.pkl.bak')
print("[+] Loaded %d data" % len(data))
train, devel = WeightedSplitSampler(len(data), data.weights, 5000, split=0.03)
train = DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn, sampler=train)
devel = DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn, sampler=devel)

print("[*] Build model")
model = TaggerModel(dropout=0.2)
if torch.cuda.is_available():
    model = model.cuda()

print("[*] Start training")
bar = trange(500)
try:
    best_model = None
    best_criterion = -1e10
    for epoch in bar:
        loss, acc = model.fit(train)
        devel_acc, = model.score(devel)
        if devel_acc.value() > best_criterion:
            best_criterion = devel_acc.value()
            best_model = copy.deepcopy(model)
            tqdm.write("[Epoch {:3d}] {} *"
                    .format(epoch, ', '.join(map(str, [loss, acc, devel_acc]))))
        else:
            tqdm.write("[Epoch {:3d}] {}"
                    .format(epoch, ', '.join(map(str, [loss, acc, devel_acc]))))
except KeyboardInterrupt:
    pass
bar.close()

model = best_model
trainp = model.predict(devel)
trainp = [(x, p, t) for x, p, t in trainp if p != t]
print(len(trainp))
from utils import *
import matplotlib.pyplot as plt
for x, p, t in trainp[:10]:
    print(hairTags[p])
    print(hairTags[t])
    plt.imshow(x.transpose(1, 2, 0))
    plt.show()

with open(os.path.join(MODEL, 'output', 'hairModel.pt'), 'wb') as f:
    torch.save(model, f)

print("[*] Load dataset")
data = TaggerDataset(FLAGS.dataFolder, training=True, tag='eyes', filename='merge.pkl.bak')
print("[+] Loaded %d data" % len(data))
train, devel = WeightedSplitSampler(len(data), data.weights, 5000, split=0.03)
train = DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn, sampler=train)
devel = DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn, sampler=devel)

print("[*] Build model")
model = TaggerModel(dropout=0.2)
if torch.cuda.is_available():
    model = model.cuda()

print("[*] Start training")
bar = trange(500)
try:
    best_model = None
    best_criterion = -1e10
    for epoch in bar:
        loss, acc = model.fit(train)
        devel_acc, = model.score(devel)
        if devel_acc.value() > best_criterion:
            best_criterion = devel_acc.value()
            best_model = copy.deepcopy(model)
            tqdm.write("[Epoch {:3d}] {} *"
                    .format(epoch, ', '.join(map(str, [loss, acc, devel_acc]))))
        else:
            tqdm.write("[Epoch {:3d}] {}"
                    .format(epoch, ', '.join(map(str, [loss, acc, devel_acc]))))
except KeyboardInterrupt:
    pass
bar.close()

model = best_model
trainp = model.predict(devel)
trainp = [(x, p, t) for x, p, t in trainp if p != t]
print(len(trainp))
from utils import *
import matplotlib.pyplot as plt
for x, p, t in trainp[:10]:
    print(eyesTags[p])
    print(eyesTags[t])
    plt.imshow(x.transpose(1, 2, 0))
    plt.show()

with open(os.path.join(MODEL, 'output', 'eyesModel.pt'), 'wb') as f:
    torch.save(model, f)
