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
from utils import trim_phone
import sys

batch_size=16
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

print("Load dataset")
with open('special/output/vocab.pkl', 'rb') as f:
    vocab = pickle.load(f)
vocab_dict = {i: w for w, i in vocab.items()}
#  data = MSVDDataset('../data', 'training', vocab=vocab)
#  train = DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn)
data = MSVDDataset(sys.argv[1], 'testing', vocab=vocab, 
        id=['klteYv1Uv9A_27_33.avi', '5YJaS2Eswg0_22_26.avi',
            'UbmZAe5u5FI_132_141.avi', 'JntMAcTlOF0_50_70.avi',
            'tJHUH9tpqPg_113_118.avi'])
devel = DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn, shuffle=False)

with open('special/output/PredNet.pt', 'rb') as f:
    predmodel = torch.load(f)

if torch.cuda.is_available():
    predmodel.cuda()
else:
    predmodel.cpu()

develp = predmodel.predict(devel)

with open(sys.argv[2], 'w') as f:
    for id, (ci, xi) in zip(data.id, develp):
        p = ' '.join(vocab_dict[i] for i in xi)
        f.write('%s,%s\n' % (id, p))
