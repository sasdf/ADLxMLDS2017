#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import torch
from torch import optim
from torch.utils.data import DataLoader
from Dataset.LmDataset import LmDataset
from SplitSampler import SplitSampler
import numpy as np
from tqdm import *
import pickle
from Net import LmNet
import itertools as it
import random
from Model import LmModel
import copy
import os
from flags import FLAGS

batch_size=32
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

MODEL = os.getenv('MODEL') or ''

print("Load dataset")
vocab = np.load(os.path.join(MODEL, 'output', 'vocab.npy')).tolist()
embedding = np.load(os.path.join(MODEL, 'output', 'embedding.npy'))
data = LmDataset(FLAGS.dataFolder, vocab=vocab)
train, devel = SplitSampler(len(data))
train = DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn, sampler=train)
devel = DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn, sampler=devel)

print("Build model")
model = LmModel(LmNet, embedding, dropout=0.0, optimizer=optim.Adam)
if torch.cuda.is_available():
    model = model.cuda()
best_model = None
best_criterion = -1e10
try:
    for epoch in trange(500):
        loss, train_acc = model.fit(train)
        devel_acc, devel_bleu = model.score(devel)
        if devel_acc > best_criterion:
            best_criterion = devel_acc
            best_model = copy.deepcopy(model)
            tqdm.write("[Epoch {:3d}] Loss: {:5.4f}, TAcc: {:5.2f}, DAcc: {:5.2f} , BLEU: {:5.2f} *"
                    .format(epoch, loss, train_acc * 100, devel_acc * 100, devel_bleu))
        else:
            tqdm.write("[Epoch {:3d}] Loss: {:5.4f}, TAcc: {:5.2f}, DAcc: {:5.2f} , BLEU: {:5.2f}"
                    .format(epoch, loss, train_acc * 100, devel_acc * 100, devel_bleu))
except KeyboardInterrupt:
    pass

model = best_model

with open(os.path.join(MODEL, 'output', 'LmNet.pt'), 'wb') as f:
    torch.save(model, f)

train_acc, train_vacc = model.score(train)
devel_acc, devel_vacc = model.score(devel)

tqdm.write("[Result] TAcc: {:5.2f}, TVcc: {:5.2f}, DAcc: {:5.2f}, DVcc: {:5.2f}"
        .format(train_acc * 100, train_vacc * 100, devel_acc * 100, devel_vacc * 100))