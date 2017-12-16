import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch import LongTensor
import math, copy


GAMMA = 0.99
EPS_START = 0.99
EPS_END = 0.05
EPS_DECAY = 100000
hiddenSZ = 128
outputSZ = 6

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class PG(nn.Module):
    def __init__(self):
        super(PG, self).__init__()
        
        self.i = nn.Sequential(
                nn.Conv2d( 1, 16, 8, 4),
                nn.ReLU(),
                nn.Conv2d(16, 32, 4, 2),
                nn.ReLU(),
                Flatten(),
                nn.Linear(2048, hiddenSZ),
                nn.ReLU(),
                nn.Linear(hiddenSZ, outputSZ),
                nn.Softmax(),
                )

    def forward(self, s):
        return self.i(s)
