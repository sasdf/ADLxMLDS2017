#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import torch
from torch.utils.data import DataLoader

import random
import numpy as np
from tqdm import *
import itertools as it
import pickle
import sys

# Sub-modules
from Dataset.TIMITDataset import TIMITDataset
from Dataset.FixDataset import FixDataset
from PredNet import PredNet
from FixNet import FixNet
from model import Model
from utils import trim_phone
from preprocess import preprocess


batch_size=64
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

print("Load dataset")
testd = TIMITDataset(preprocess(sys.argv[1]))
test = DataLoader(testd, batch_size=batch_size, collate_fn=testd.collate_fn, shuffle=False)

with open('rcnn/output/PredNet.pt', 'rb') as f:
    predmodel = torch.load(f)

testp = predmodel.predict(test)
testdataset = FixDataset(testp)
test = DataLoader(testdataset, batch_size=batch_size, collate_fn=testdataset.collate_fn)

with open('omap.pkl', 'rb') as f:
    omap = pickle.load(f)
    omap[0] = ''

with open(sys.argv[2], 'w') as f:
    f.write('id,phone_sequence\n')
    for p, l, _, n in test:
        for pi, li, ni in zip(p, l, n):
            pi = trim_phone(pi.numpy()[:li])[1:-1]
            f.write(ni+','+''.join(omap[c] for c in pi)+'\n')
print("")

