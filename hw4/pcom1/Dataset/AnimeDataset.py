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
from collections import Counter

class AnimeDataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, base_path):
        imagesPath = os.path.join(base_path, 'images.pixiv2.tagged.pkl')
        print('[*] Loading Images')
        with open(imagesPath, 'rb') as f:
            data = pickle.load(f)
            #  data = data[:1000]
        images, tags = zip(*data)
        data = []
        for x in images:
            r, g, b = x
            x = r * 0.299 + g * 0.587 + b * 0.114
            t = x[8:24, 16:-16]
            l = t >= np.percentile(t, 25)
            u = t <= np.percentile(t, 75)
            t = float(t[np.logical_and(l, u)].mean())
            data.append((x, t))
        data = [(torch.from_numpy(img).float().unsqueeze(0), FloatTensor([tags])) for img, tags in data]
        self.data = data

    def collate_fn(self, batch):
        imgs, tags = zip(*batch)
        imgs = torch.stack(imgs, 0)
        tags = torch.stack(tags, 0).squeeze(1)
        return imgs, tags

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
