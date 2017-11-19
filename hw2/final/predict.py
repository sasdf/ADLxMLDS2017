#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import torch
from torch.utils.data import DataLoader
from Dataset.MSVDDataset import MSVDDataset
import numpy as np
from tqdm import *
import pickle
from Net import PredNet
import itertools as it
import random
from Model import PredModel
from utils import toSent
import sys
import os
from flags import FLAGS
import re
from Model.bleu_eval_new import BLEU

batch_size=32
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

MODEL = os.getenv('MODEL') or ''

print("Load dataset")
vocab = np.load(os.path.join(MODEL, 'output', 'vocab.npy')).tolist()
vocab_dict = {i: w for w, i in vocab.items()}
data = MSVDDataset(FLAGS.dataFolder, 'testing', vocab=vocab, cache=False)
devel = DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn, shuffle=False)

with open(os.path.join(MODEL, 'output', 'PredNet.pt'), 'rb') as f:
    predmodel = torch.load(f)

if torch.cuda.is_available():
    predmodel.cuda()
else:
    predmodel.cpu()

develp = predmodel.predict(devel)

with open(FLAGS.outputFile, 'w') as f:
    for id, (ci, xi, ls) in zip(data.id, develp):
        xi = toSent(vocab_dict, xi)
        f.write('%s,%s\n' % (id, xi))

data = MSVDDataset(FLAGS.dataFolder, 'peer_review', vocab=vocab, cache=False)
devel = DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn, shuffle=False)

develp = predmodel.predict(devel)

with open(FLAGS.outputFilePeer, 'w') as f:
    for id, (ci, xi, ls) in zip(data.id, develp):
        xi = toSent(vocab_dict, xi)
        f.write('%s,%s\n' % (id, xi))
