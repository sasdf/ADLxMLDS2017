import sys
import json
import pickle
import numpy as np
from tqdm import *
from nltk import word_tokenize


print("Tokenizing")
corpus = []
with open(sys.argv[3]) as f:
    for l in tqdm(json.load(f)):
        for li in l['caption']:
            li = [w for w in word_tokenize(li)]
            if li[-1] == '.':
                li.pop()
            corpus.append(li)
vocab = list(set(w for l in corpus for w in l))
print(len(vocab))

print("Loading GloVe")
vector = np.load(sys.argv[1])
with open(sys.argv[2]) as f:
    words = [l[:-1] for l in f]
    words = {w: i for i, w in enumerate(words)}
print(len(words))

oov = [w for w in vocab if w not in words]
vocab = ['</s>', '<s>'] + [w for w in vocab if w in words]
print(oov)
print(len(vocab))
emb = np.vstack([vector[words[w]] for w in vocab])
vocab = {w: i for i, w in enumerate(vocab)}
np.save('output/embedding.npy', emb)
np.save('output/vocab.npy', vocab)
corpus = [[vocab[w] for w in l if w in vocab] for l in corpus]
np.save('output/corpus.npy', corpus)
