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

class TIMITDataset(Dataset):
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
                data = pickle.load(f)
        self.data = data

    def _pad(self, arr, dtype, pv):
        origlen = list(map(len, arr))
        maxlen = max(origlen)
        pad = [[pv] * arr[0].shape[1]]
        result = dtype([e.tolist() + pad * (maxlen - len(e)) for e in arr])
        return result, LongTensor(origlen)

    def collate_fn(self, batch):
        n, g, mfcc, bank, label = zip(*batch)
        mfcc, mfccl = self._pad(mfcc, FloatTensor, 0.0)
        #  bank, bankl = self._pad(bank)
        #  label = [l + 1 for l in label]
        label, labell = self._pad(label, LongTensor, 0)
        return mfcc, mfccl, label, n

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
