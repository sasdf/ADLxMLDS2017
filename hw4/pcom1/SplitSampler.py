import torch
from torch.utils.data.sampler import SubsetRandomSampler, Sampler
import random

def SplitSampler(datalen, split=0.1, split_shuffle=True):
    idx = list(range(datalen))
    if split_shuffle:
        random.shuffle(idx)
    sz = int(datalen * split)
    train = SubsetRandomSampler(idx[sz:])
    devel = SubsetRandomSampler(idx[:sz])
    return train, devel

def WeightedSplitSampler(datalen, weights, num_samples, replacement=True, split=0.1, split_shuffle=True):
    idx = list(range(datalen))
    if split_shuffle:
        random.shuffle(idx)
    weights = [weights[i] for i in idx]
    sz = int(datalen * split)
    train = WeightedSubsetSampler(idx[sz:], weights[sz:], int(num_samples * (1 - split)), replacement)
    devel = WeightedSubsetSampler(idx[:sz], weights[:sz], int(num_samples * split), replacement)
    return train, devel


class WeightedSubsetSampler(Sampler):
    """Samples elements from [0,..,len(weights)-1] with given probabilities (weights).

    Arguments:
        weights (list)   : a list of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.
    """

    def __init__(self, indices, weights, num_samples, replacement=True):
        self.indices = indices
        self.weights = torch.DoubleTensor(weights)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(self.weights, self.num_samples, self.replacement))

    def __len__(self):
        return self.num_samples
