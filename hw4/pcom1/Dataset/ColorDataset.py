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

class ColorDataset(Dataset):
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
        tags = [[h, e] if e != 0 else [h, 1] for h, e in tags]
        tags = list(map(tuple, tags))
        counter = Counter(tags).most_common()
        filt = {t: 1 for t, n in counter if n > 50}
        data = [(i, t) for i, t in zip(images, tags) if t in filt]
        data = [(torch.from_numpy(img).float(), LongTensor(tags)) for img, tags in data]
        self.data = data

    def collate_fn(self, batch):
        imgs, tags = zip(*batch)
        imgs = torch.stack(imgs, 0)
        tags = torch.stack(tags, 0)
        return imgs, tags

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
