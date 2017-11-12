#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import torch
from torch import optim
from torch.utils.data import DataLoader
from Dataset.MSVDDataset import MSVDDataset
from SplitSampler import SplitSampler
import numpy as np
from tqdm import *
import pickle
from Net import PredNet
import itertools as it
import random
from Model import PredModel
import copy

batch_size=16
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print("Load dataset")
vocab = None
try:
    with open('output/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
except FileNotFoundError:
    pass
data = MSVDDataset('../data', 'training', vocab=vocab)
if not vocab:
    vocab = data.vocab
    with open('output/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)
train = DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn)
data = MSVDDataset('../data', 'testing', vocab=vocab)
devel = DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn)

print("Build model")
model = PredModel(PredNet, len(vocab)+2, dropout=0.1, optimizer=optim.RMSprop)
best_model = None
best_criterion = -1e10
try:
    for epoch in trange(500):
        loss, train_acc = model.fit(train)
        devel_acc, devel_vacc = model.score(devel)
        if devel_acc > best_criterion:
            best_criterion = devel_acc
            best_model = copy.deepcopy(model)
            tqdm.write("[Epoch {:3d}] Loss: {:5.4f}, TAcc: {:5.2f}, DAcc: {:5.2f} , DVcc: {:5.2f} *"
                    .format(epoch, loss, train_acc * 100, devel_acc * 100, devel_vacc * 100))
        else:
            tqdm.write("[Epoch {:3d}] Loss: {:5.4f}, TAcc: {:5.2f}, DAcc: {:5.2f} , DVcc: {:5.2f}"
                    .format(epoch, loss, train_acc * 100, devel_acc * 100, devel_vacc * 100))
except KeyboardInterrupt:
    pass

model = best_model

with open('output/PredNet.pt', 'wb') as f:
    torch.save(model, f)

train_acc, train_vacc = model.score(train)
devel_acc, devel_vacc = model.score(devel)

tqdm.write("[Result] TAcc: {:5.2f}, TVcc: {:5.2f}, DAcc: {:5.2f}, DVcc: {:5.2f}"
        .format(train_acc * 100, train_vacc * 100, devel_acc * 100, devel_vacc * 100))
