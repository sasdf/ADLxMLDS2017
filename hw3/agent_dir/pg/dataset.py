#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import numpy as np
import torch
from torch.utils.data import Dataset
from torch import LongTensor, FloatTensor
from torch.autograd import Variable
import json
import itertools as it
import random
from tqdm import tqdm

class Experience(Dataset):

    def __init__(self, memory, size):
        super().__init__()
        self.size = size
        self.update(memory)

    def update(self, memory):
        self.memory = memory

    def collate_fn(self, batch):
        (state, next_state, action, reward, importance, index) = zip(*batch)
        state = torch.stack(state, 0)
        next_state = torch.stack(next_state, 0)
        action = torch.stack(action, 0)
        reward = torch.stack(reward, 0)
        importance = FloatTensor(importance)
        return state, next_state, action, reward, importance, index

    def __getitem__(self, _):
        return self.memory.sample()

    def __len__(self):
        return self.size

