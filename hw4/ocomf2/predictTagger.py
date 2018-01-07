#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import torch
from torch import optim
from torch.utils.data import DataLoader
from Dataset.TaggerDataset import TaggerDataset
from SplitSampler import SplitSampler
import numpy as np
from tqdm import *
import pickle
import itertools as it
import random
import copy
import os
from Model import TaggerModel
from flags import FLAGS

batch_size=64
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

MODEL = os.getenv('MODEL') or ''

print("[*] Load dataset")
data = TaggerDataset(FLAGS.dataFolder, training=False, tag='hair', filename='merge.pixiv2.pkl')
print("[+] Loaded %d data" % len(data))
test = DataLoader(data, batch_size=batch_size, collate_fn=data.collate_fn, shuffle=False)

print("[*] Loading model")
with open(os.path.join(MODEL, 'output', 'hairModel.pt'), 'rb') as f:
    hairModel = torch.load(f)
with open(os.path.join(MODEL, 'output', 'eyesModel.pt'), 'rb') as f:
    eyesModel = torch.load(f)

print("[*] Start predicting")
testHair = hairModel.predict(test)
testEyes = eyesModel.predict(test)

testp = [(x, (h, e)) for (x, h, t, c), (_, e, t, d) in zip(testHair, testEyes) if c > 0.4 and d > 0.4]
print(len(testp))

#  print("[*] Visualize result")
#  from utils import *
#  import cv2
#  cv2.namedWindow('viewer', cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
#  cv2.resizeWindow('viewer', 256, 256)
#  cv2.imshow('viewer', 0)
#  def hasWindow():
    #  return cv2.getWindowProperty('viewer', cv2.WND_PROP_VISIBLE) > 0
#  g = 0
#  t = 0
#  for x, p in testp[:100]:
    #  print(hairTags[p[0]], eyesTags[p[1]])
    #  img = x.transpose(1, 2, 0)
    #  img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #  img = cv2.resize(img, (256, 256), cv2.INTER_LANCZOS4)
    #  cv2.imshow('viewer', img)
    #  while hasWindow():
        #  k = cv2.waitKey(1000)
        #  if k == ord('q'):
            #  exit(0)
        #  if k == ord('a'):
            #  g += 1
            #  t += 1
            #  break
        #  if k == ord('s'):
            #  t += 1
            #  break
    #  print('Good: %d, Total: %d, Ratio: %.3f' % (g, t, g/t))

with open(os.path.join(FLAGS.dataFolder, 'images.pixiv2.tagged.pkl'), 'wb') as f:
    pickle.dump(testp, f)
