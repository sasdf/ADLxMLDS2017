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

class FixDataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, data):
        if isinstance(data, str):
            with open(data, 'rb') as f:
                self.data = pickle.load(f)
                #  self.data = [d for d in self.data if d[1] == 0]
        else:
            self.data = data

    def _pad(self, arr, dtype, pv):
        origlen = list(map(len, arr))
        maxlen = max(origlen)
        pad = [pv]
        result = dtype([e + pad * (maxlen - len(e)) for e in arr])
        return result, LongTensor(origlen)

    def collate_fn(self, batch):
        conf, pred, y, n = zip(*batch)
        x, l = self._pad(pred, LongTensor, 0)
        c, _ = self._pad(conf, FloatTensor, 0.0)
        y, _ = self._pad(y, LongTensor, 0)
        return c, x, l, y, n

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
