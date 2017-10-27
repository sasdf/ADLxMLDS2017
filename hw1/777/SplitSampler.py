from torch.utils.data.sampler import SubsetRandomSampler
import random

def SplitSampler(datalen, split=0.1):
    idx = list(range(datalen))
    random.shuffle(idx)
    split = int(datalen * split)
    train = SubsetRandomSampler(idx[split:])
    devel = SubsetRandomSampler(idx[:split])
    return train, devel
