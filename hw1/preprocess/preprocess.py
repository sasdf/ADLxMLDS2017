import sys
import os
import numpy as np
import itertools as it
import functools as fn
from collections import namedtuple
import pickle

Name = namedtuple('Name', ['gender', 'speaker', 'sent', 'frame', 'raw'])
Name.__hash__ = lambda x: hash(x.raw)

def parseName(name):
    s = name.split('_')
    return Name(
            gender=s[0][0],
            speaker=s[0],
            sent=s[1],
            frame=int(s[2]),
            raw=name
            )

def processGroup(g):
    feat = sorted(list(g), key=lambda x: x[0].frame)
    name = feat[0][0]
    feat = np.vstack([f[1] for f in feat])
    return [name, feat]

def processArk(fname):
    print("Process %s" % fname)
    with open(fname) as f:
        data = f.readlines()
    data = [l.strip().split(' ', 1) for l in data]
    data = [(parseName(name), np.fromstring(feat, sep=' ', dtype=np.float32)) for name, feat in data]
    data = it.groupby(data, lambda x: x[0].sent)
    data = [processGroup(feat) for sent, feat in data]
    return data

def processLabel(fname, cmap, smap):
    print("Process Label")
    with open(smap) as f:
        smap = dict(l.strip().split('\t') for l in f)
    with open(cmap) as f:
        m = [l.strip().split('\t') for l in f]
        cmap = list(set(smap[s] for s, i, c in m))
        cmap = {v: i + 1 for i, v in enumerate(cmap)}
        omap = {cmap[s]: c for s, i, c in m if s in cmap}
    
    with open(fname) as f:
        data = f.readlines()
    data = [l.strip().split(',') for l in data]
    data = [[parseName(n), cmap[smap[l]]] for n, l in data]
    data = it.groupby(data, lambda x: x[0].sent)
    data = [processGroup(feat) for sent, feat in data]
    return data, omap
    
def merge(mfcc, bank, labl):
    print("Merging")
    data = {}
    for n, v in mfcc:
        data[n] = []
        data[n].append(n.speaker+'_'+n.sent)
        data[n].append((n.gender == 'f') + 0)
        data[n].append(v)
    for n, v in bank:
        data[n].append(v)
    if labl:
        for n, v in labl:
            data[n].append(v)
    else:
        for n, a in data.items():
            a.append(np.zeros((a[-1].shape[0], 1), dtype=np.int32))
    return list(data.values())

def preprocess(data):
    emfcc = processArk(os.path.join(data, 'mfcc', 'test.ark'))
    ebank = processArk(os.path.join(data, 'fbank', 'test.ark'))

    test = merge(emfcc, ebank, None)
    return test
