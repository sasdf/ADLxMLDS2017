import torch
from torch.utils.data import DataLoader
from Dataset.TIMITDataset import TIMITDataset
from Dataset.FixDataset import FixDataset
from SplitSampler import SplitSampler
import numpy as np
from tqdm import *
import pickle
from FixNet import FixNet
import itertools as it
import random
from model import Model
import copy

batch_size=64
random.seed(242)
np.random.seed(242)
torch.manual_seed(242)
torch.cuda.manual_seed(242)

print("Load dataset")
traindataset = FixDataset('output/trainp.pkl')
develdataset = FixDataset('output/develp.pkl')
train = DataLoader(traindataset, batch_size=batch_size, collate_fn=traindataset.collate_fn)
devel = DataLoader(develdataset, batch_size=batch_size, collate_fn=develdataset.collate_fn)

def fix(model, dataset):
    data = model.predict(dataset)
    dataset = FixDataset(data)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=dataset.collate_fn)

print("Build fixing model")
fixmodel = Model(FixNet, dropout=0.2)
try:
    for round in it.count():
        best_model = None
        best_dis = 1e10
        try:
            for epoch in trange(200):
                loss, train_acc = fixmodel.fit(train)
                devel_acc, devel_dis = fixmodel.score(devel)
                if devel_dis < best_dis:
                    best_dis = devel_dis
                    best_model = copy.deepcopy(fixmodel)
                    tqdm.write("[Epoch {:3d}] Loss: {:5.4f}, TAcc: {:5.2f}, DAcc: {:5.2f}, DDis: {:6.3f} *"
                            .format(epoch, loss, train_acc * 100, devel_acc * 100, devel_dis))
                else:
                    tqdm.write("[Epoch {:3d}] Loss: {:5.4f}, TAcc: {:5.2f}, DAcc: {:5.2f}, DDis: {:6.3f}"
                            .format(epoch, loss, train_acc * 100, devel_acc * 100, devel_dis))
        except KeyboardInterrupt:
            pass
        if best_model:
            fixmodel = best_model
        inp = input("\n\nNext?[Y/n]")
        if inp in ['n', 'N', 'no', 'No', 'NO']:
            break
        with open('output/FixNet.%d.pt' % round, 'wb') as f:
            torch.save(fixmodel, f)
        train = fix(fixmodel, train)
        devel = fix(fixmodel, devel)

    with open('output/FixNet.%d.pt' % round, 'wb') as f:
        torch.save(fixmodel, f)

    train_acc, train_dis = fixmodel.score(train)
    devel_acc, devel_dis = fixmodel.score(devel)

    tqdm.write("[Result] TAcc: {:5.2f}, TDis: {:6.3f}, DAcc: {:5.2f}, DDis: {:6.3f}"
            .format(train_acc * 100, train_dis, devel_acc * 100, devel_dis))
except KeyboardInterrupt:
    pass
