import torch
from torch.utils.data import DataLoader
from Dataset.TIMITDataset import TIMITDataset
from Dataset.FixDataset import FixDataset
from SplitSampler import SplitSampler
import numpy as np
from tqdm import *
import pickle
from PredNet import PredNet
from FixNet import FixNet
import itertools as it
import random
from model import Model
from modelPred import ModelPred
from utils import trim_phone

batch_size=32
random.seed(242)
np.random.seed(242)
torch.manual_seed(242)
torch.cuda.manual_seed(242)

print("Load dataset")
timit = TIMITDataset('../data/train.pkl')
train, devel = SplitSampler(len(timit))
train = DataLoader(timit, batch_size=batch_size, collate_fn=timit.collate_fn, sampler=train)
devel = DataLoader(timit, batch_size=batch_size, collate_fn=timit.collate_fn, sampler=devel)

with open('output/PredNet.pt', 'rb') as f:
    predmodel = torch.load(f)

train_acc, train_dis = predmodel.score(train)
devel_acc, devel_dis = predmodel.score(devel)

tqdm.write("[Result] TAcc: {:5.2f}, TDis: {:6.3f}, DAcc: {:5.2f}, DDis: {:6.3f}"
        .format(train_acc * 100, train_dis, devel_acc * 100, devel_dis))

trainp = predmodel.predict(train)
develp = predmodel.predict(devel)

with open('output/trainp.pkl', 'wb') as f:
    pickle.dump(trainp, f)
with open('output/develp.pkl', 'wb') as f:
    pickle.dump(develp, f)
