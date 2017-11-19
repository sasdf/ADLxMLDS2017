from torch.utils.data.sampler import SubsetRandomSampler
import random

def SplitSampler(datalen, split=0.1, split_shuffle=True):
    idx = list(range(datalen))
    if split_shuffle:
        random.shuffle(idx)
    split = int(datalen * split)
    train = SubsetRandomSampler(idx[split:])
    devel = SubsetRandomSampler(idx[:split])
    return train, devel
