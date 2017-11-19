#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import pickle
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch import LongTensor, FloatTensor
from torch.autograd import Variable
import json
import itertools as it
import random
from tqdm import tqdm

class LmDataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, base_path, vocab):
        path = os.path.join(base_path, 'extra', '1blm', 'lmcorpus.txt')
        with open(path) as f:
            data = [[vocab[w] for w in l.split()[:-1]] for l in f]
        self.data = data

    def _pad(self, arr):
        origlen = list(map(len, arr))
        maxlen = max(origlen)
        pad = [0]
        result = LongTensor([e + pad * (maxlen - len(e)) for e in arr])
        return result, LongTensor(origlen)

    def collate_fn(self, batch):
        label, labelLen = self._pad(batch)
        x = torch.ones(label.size(0), 1).long()
        x = torch.cat([x, label[:,1:]], 1)
        return x, label, labelLen,

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
