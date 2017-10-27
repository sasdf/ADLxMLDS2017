memory_required=1024
import subprocess
import re
import time
import sys
print ('Checking Memory')
while True:
    m = subprocess.check_output(['nvidia-smi', '-q', '-d', 'MEMORY'])
    m = str(m).split('\n')
    free = [ re.search(r'Free\s*:\s*([0-9]+)\s*MiB', x) for x in m ]
    total = [ re.search(r'Total\s*:\s*([0-9]+)\s*MiB', x) for x in m ]
    free = [int(x.group(1)) for x in free if x][0]
    total = [int(x.group(1)) for x in total if x][0]
    sys.stdout.write('\33[2K\rTotal: %5d, Free: %5d, Need: %5d ' % (total, free, memory_required))
    sys.stdout.flush()
    if free >= memory_required:
        #  time.sleep(300)
        #  if free >= memory_required:
            print ('Enough Free memory available')
            break
    for i in range(3):
        time.sleep(.5)
        sys.stdout.write('.')
        sys.stdout.flush()

import torch
from torch import optim
from torch.utils.data import DataLoader
from Dataset.TIMITDataset import TIMITDataset
from Dataset.FixDataset import FixDataset
from SplitSampler import SplitSampler
import numpy as np
from tqdm import *
import pickle
from PredNet import PredNet
import itertools as it
import random
from model import Model
import copy

batch_size=32
random.seed(242)
np.random.seed(242)
torch.manual_seed(242)
torch.cuda.manual_seed(242)

print("Load dataset")
timit = TIMITDataset('../../data/train.pkl')
train, devel = SplitSampler(len(timit))
train = DataLoader(timit, batch_size=batch_size, collate_fn=timit.collate_fn, sampler=train)
devel = DataLoader(timit, batch_size=batch_size, collate_fn=timit.collate_fn, sampler=devel)

print("Build model")
predmodel = Model(PredNet, dropout=0.2, optimizer=optim.RMSprop)
best_model = None
best_dis = 1e10
try:
    for epoch in trange(180):
        loss, train_acc = predmodel.fit(train)
        devel_acc, devel_dis = predmodel.score(devel)
        if devel_dis < best_dis:
            best_dis = devel_dis
            best_model = copy.deepcopy(predmodel)
            tqdm.write("[Epoch {:3d}] Loss: {:5.4f}, TAcc: {:5.2f}, DAcc: {:5.2f}, DDis: {:6.3f} *"
                    .format(epoch, loss, train_acc * 100, devel_acc * 100, devel_dis))
        else:
            tqdm.write("[Epoch {:3d}] Loss: {:5.4f}, TAcc: {:5.2f}, DAcc: {:5.2f}, DDis: {:6.3f}"
                    .format(epoch, loss, train_acc * 100, devel_acc * 100, devel_dis))
except KeyboardInterrupt:
    pass

predmodel = best_model

train_acc, train_dis = predmodel.score(train)
devel_acc, devel_dis = predmodel.score(devel)

tqdm.write("[Result] TAcc: {:5.2f}, TDis: {:6.3f}, DAcc: {:5.2f}, DDis: {:6.3f}"
        .format(train_acc * 100, train_dis, devel_acc * 100, devel_dis))

with open('output/PredNet.pt', 'wb') as f:
    torch.save(predmodel, f)

trainp = predmodel.predict(train)
develp = predmodel.predict(devel)

with open('output/trainp.pkl', 'wb') as f:
    pickle.dump(trainp, f)
with open('output/develp.pkl', 'wb') as f:
    pickle.dump(develp, f)
