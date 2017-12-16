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
hiddenSZ = 512
outputSZ = 4

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        
        self.i = nn.Sequential(
                nn.Conv2d( 4, 32, 8, 4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, 2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1),
                nn.ReLU(),
                Flatten(),
                nn.Linear(3136, hiddenSZ),
                nn.ReLU(),
                )

        #  self.v = nn.Sequential(
                #  nn.Linear(hiddenSZ, 1),
                #  nn.SELU()
                #  )

        self.a = nn.Sequential(
                nn.Linear(hiddenSZ, outputSZ),
                nn.ReLU()
                )

    def A(self, state):
        state = self.i(state)
        a = self.a(state)
        a = a.max(1)[1]
        return a.view(-1)

    def Q(self, state, action=None):
        state = self.i(state)
        #  v = self.v(state)
        a = self.a(state)
        #  q = v - a.mean(1, keepdim=True) + a
        #  q = v.expand_as(a) + a - a.mean(1, keepdim=True).expand_as(a)
        q = a
        if action is not None:
            q = q.gather(1, action)
        return q

class DDQN(nn.Module):
    def __init__(self):
        super(DDQN, self).__init__()
        self.steps_done = 0

        self.targetNet = QNetwork()
        self.evalNet = QNetwork()
        self.sync()

    def sync(self):
        self.targetNet = copy.deepcopy(self.evalNet)
        for p in self.targetNet.parameters():
            p.data.require_grad = False

    def make_action(self, state):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            return self.evalNet.A(state)
        else:
            return Variable(LongTensor([random.randrange(outputSZ) for s in state]))

    def loss(self, state, next_state, action, reward):
        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        Q = self.evalNet.Q(state, action)

        # Compute argmax_a Q(s_{t+1}, a) for all next states.
        #  An = self.evalNet.A(next_state).view(-1, 1)

        #  Qn = self.targetNet.Q(next_state, An).detach()

        Qn = self.targetNet.Q(next_state).max(1, keepdim=True)[0].detach()

        # Compute the expected Q values
        expectedQ = (Qn * GAMMA) + reward

        # Compute Huber loss
        #  loss = torch.abs(Q - expectedQ)
        #  loss = (Q - expectedQ) ** 2
        loss = F.smooth_l1_loss(Q, expectedQ)
        #  loss = torch.abs(Q - expectedQ).sum(0).view(-1)

        return loss
        #  return loss.squeeze(1)



