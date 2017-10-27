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

with open('777/output/PredNet.pt', 'rb') as f:
    predmodel = torch.load(f)

testp = predmodel.predict(test)
testdataset = FixDataset(testp)
test = DataLoader(testdataset, batch_size=batch_size, collate_fn=testdataset.collate_fn)

def fix(model, dataset):
    data = model.predict(dataset)
    dataset = FixDataset(data)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)

with open('omap.pkl', 'rb') as f:
    omap = pickle.load(f)
    omap[0] = ''

print("Build fixing model")
try:
    rbar = tqdm(it.count())
    for round in rbar:
        try:
            with open('777/output/FixNet.%d.pt' % round, 'rb') as f:
                fixmodel = torch.load(f)
                test = fix(fixmodel, test)
        except FileNotFoundError:
            break
    rbar.close()

    with open(sys.argv[2], 'w') as f:
        f.write('id,phone_sequence\n')
        for c, p, l, _, n in test:
            for ci, pi, li, ni in zip(c, p, l, n):
                pi = trim_phone(pi.numpy()[:li], ci.tolist()[:li])[1:-1]
                f.write(ni+','+''.join(omap[o] for o in pi)+'\n')

except KeyboardInterrupt:
    pass
print("")
