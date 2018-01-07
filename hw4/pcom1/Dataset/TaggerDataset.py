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
from utils import toHairIdx, toEyesIdx
from collections import Counter

class TaggerDataset(Dataset):
    """Dataset wrapping data and target tensors.

    Each sample will be retrieved by indexing both tensors along the first
    dimension.

    Arguments:
        data_tensor (Tensor): contains sample data.
        target_tensor (Tensor): contains sample targets (labels).
    """

    def __init__(self, base_path, tag, training=False, filename='merge.pkl'):
        imagesPath = os.path.join(base_path, filename)
        tagsPath = os.path.join(base_path, 'tags_clean.json')
        print('[*] Loading Images')
        with open(imagesPath, 'rb') as f:
            images, fns = pickle.load(f)
            #  images = images[:1000]
            #  fns = fns[:1000]
            images = [torch.from_numpy(img).float() / 255.0 for img in images]
            ids = [fn[:-4] for fn in fns]

        if training:
            print('[*] Loading Tags')
            with open(tagsPath, 'r') as f:
                _tags = {i: t for i, t in json.load(f)}
                tags = []
                for i in ids:
                    t = _tags[i]
                    hair = [z for z in t if z.endswith('hair')]
                    eyes = [z for z in t if z.endswith('eyes')]
                    if len(hair) != 1 or len(eyes) != 1:
                        t = None
                    else:
                        t = LongTensor([toHairIdx[hair[0]], toEyesIdx[eyes[0]]])
                    tags.append(t)
            images, tags = zip(*[z for z in zip(images, tags) if z[1] is not None])
        else:
            tags = [LongTensor([0, 0])] * len(images)

        if tag == 'hair':
            tags = [LongTensor([t[0]]) for t in tags]
        elif tag == 'eyes':
            tags = [LongTensor([t[1]]) for t in tags]
        else:
            raise AttributeError('Unknown Tag')

        c = Counter(tags)
        w = {k: 1/v for k, v in c.items()}
        self.weights = [w[t] for t in tags]

        self.images = images
        self.tags = tags

    def collate_fn(self, batch):
        imgs, tags = zip(*batch)
        imgs = torch.stack(imgs, 0)
        tags = torch.cat(tags, 0)
        return imgs, tags

    def __getitem__(self, index):
        return self.images[index], self.tags[index]

    def __len__(self):
        return len(self.images)
