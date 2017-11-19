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
from nltk import word_tokenize

def tokenize(s):
    s = [str(w) for w in word_tokenize(s)]
    if s[-1] == '.':
        s.pop()
    return s

class MSVDDataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, base_path, name, vocab=None, id=None, cache=True):
        cache_path = os.path.join(base_path, name + '.pt')
        try:
            cacmtime = os.stat(cache_path).st_mtime
            srcmtime = os.stat(__file__).st_mtime
            if srcmtime > cacmtime:
                raise FileNotFoundError
            with open(cache_path, 'rb') as f:
                cache_data = torch.load(f)
            self.vocab = cache_data['vocab']
            self.origLabel = cache_data['origLabel']
            self.label = cache_data['label']
            self.id    = cache_data['id']
            self.feat  = cache_data['feat']
        except FileNotFoundError:
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
                origLabel = None

            if not vocab:
                corpus = set(tokenize(' '.join(it.chain(*list(label.values())))))
                vocab = {w: i+2 for i, w in enumerate(corpus)}

            if not id:
                try:
                    with open(id_path) as f:
                        id = [l.strip() for l in f]
                except FileNotFoundError:
                    id = list(label.keys())
            #  id = id[:10]
            
            if label:
                origLabel = [[x.rstrip('.') for x in label[i]] for i in id]
                label = [[[vocab[w] if w in vocab else -1 for w in tokenize(l)] for l in label[i]] for i in tqdm(id)]
            
            bar = tqdm(id, smoothing=0)
            feat = [FloatTensor(np.load(os.path.join(feat_path, i+'.npy'))) for i in bar]
            print(feat[0].shape)

            self.vocab = vocab
            self.origLabel = origLabel
            self.label = label
            self.id    = id
            self.feat  = feat
            if cache:
                cache_data = {
                    'vocab': vocab,
                    'origLabel': origLabel,
                    'label': label,
                    'id': id,
                    'feat': feat
                }
                with open(cache_path, 'wb') as f:
                    torch.save(cache_data, f)
        if self.label:
            self.selected = [[w for w in random.choice(l) if w >= 0] for l in self.label]
        else:
            self.selected = self.id[:]
            self.origLabel = self.id[:]
        self.count = [0 for i in self.id]

    def _pad(self, arr):
        origlen = list(map(len, arr))
        maxlen = max(origlen)
        pad = [0]
        result = LongTensor([e + pad * (maxlen - len(e)) for e in arr])
        return result, LongTensor(origlen)

    def collate_fn(self, batch):
        feat, label, labels = zip(*batch)
        feat = torch.stack(feat).contiguous()
        if self.label:
            label, labelLen = self._pad(label)
        else:
            label = None
            labelLen = None
        return feat, label, labelLen, labels

    def __getitem__(self, index):
        if self.label:
            self.count[index] += 1
            if self.count[index] > 6:
                self.count[index] = 0
                self.selected[index] = [w for w in random.choice(self.label[index]) if w >= 0]
        return self.feat[index], self.selected[index], self.origLabel[index]

    def __len__(self):
        return len(self.id)
