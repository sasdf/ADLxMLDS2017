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

class MSVDDataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, base_path, name, vocab=None, id=None):
        label_path = os.path.join(base_path, name + '_label.json')
        id_path    = os.path.join(base_path, name + '_id.txt')
        feat_path  = os.path.join(base_path, name, 'feat')
        if not os.path.exists(feat_path):
            feat_path = os.path.join(base_path, name + '_data', 'feat')

        try:
            with open(label_path) as f:
                label = json.load(f)
                label = {l['id']: l['caption'] for l in label}
        except FileNotFoundError:
            label = None

        if not vocab:
            corpus = set(' '.join(it.chain(*list(label.values()))).split(' '))
            vocab = {w: i+2 for i, w in enumerate(corpus)}

        if not id:
            try:
                with open(id_path) as f:
                    id = [l.strip() for l in f]
            except FileNotFoundError:
                id = list(label.keys())
        #  id = id[:10]
        
        if label:
            label = [[[vocab[w] for w in l.split(' ') if w in vocab] for l in label[i]] for i in id]
        
        bar = tqdm(id, smoothing=0)
        feat = [FloatTensor(np.load(os.path.join(feat_path, i+'.npy'))) for i in bar]

        self.vocab = vocab
        self.label = label
        self.id    = id
        self.feat  = feat

    def _pad(self, arr):
        origlen = list(map(len, arr))
        maxlen = max(origlen)
        pad = [0]
        result = LongTensor([e + pad * (maxlen - len(e)) for e in arr])
        return result, LongTensor(origlen)

    def collate_fn(self, batch):
        feat, label = zip(*batch)
        feat = torch.stack(feat).contiguous()
        label, labelLen = self._pad(label)
        return feat, label, labelLen

    def __getitem__(self, index):
        label = random.choice(self.label[index])
        return self.feat[index], label

    def __len__(self):
        return len(self.id)
