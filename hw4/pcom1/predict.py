#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import torch
import numpy as np
from tqdm import *
import sys, os
import random
import time
from utils import *
from skimage import io, transform, filters

torch.backends.cudnn.benchmark = True
batch_size=64
seed = 6
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
MODEL = os.getenv('MODEL') or ''

print("[*] Loading model")
noiseDim = 128
generator = torch.load(os.path.join(MODEL, 'output', 'Generator-80.pt'))
colorizer = torch.load(os.path.join(MODEL, 'output', 'Colorizer-8.pt'))
generator = generator.train()
colorizer = colorizer.eval()

hairIllum = [0.4, 0.9, 0.6, 0.5, 0.5, 0.3, 0.5, 0.6, 0.3, 0.15, 0.4, 0.7]

with open(sys.argv[1]) as f:
    #  r = []
    for l in f:
        l = l.strip()
        id, tagtext = l.split(',', 1)
        for hair in hairTags:
            if hair in tagtext:
                break
        for eyes in eyesTags:
            if eyes in tagtext:
                break
        hair = toHairIdx[hair]
        eyes = toEyesIdx[eyes]
        if eyes == 0 or eyes == 11:
            eyes = 1
        print(hair, eyes)
        noise = torch.randn(5, noiseDim) * 0.2
        #  noise = torch.FloatTensor(10, noiseDim).fill_(0.1).bernoulli_()
        #  noise += mean.unsqueeze(0)
        illum = torch.FloatTensor([hairIllum[hair] for _ in range(5)])

        tag = torch.LongTensor([[hair, eyes] for _ in range(5)])

        # Generate
        noise = createVariable(noise, True, True)
        illum = createVariable(illum, True, True)
        hair = createVariable(tag[:, 0], True, True)
        eyes = createVariable(tag[:, 1], True, True)

        x = generator(noise, illum)
        x = colorizer(hair, eyes, x)
        x = x.data.cpu().numpy()
        x = x.transpose(0, 2, 3, 1)
        #  rr = []
        for i, img in enumerate(x):
            img = np.clip(img, 0, 1)
            #  img = transform.resize(img, (80, 80))
            img = filters.gaussian(img, sigma=0.5, multichannel=True)
            img = transform.resize(img, (64, 64), mode='reflect')
            img = np.clip(img * 255 + 10, 0, 255).astype(np.uint8)
            #  rr.append(img)
            io.imsave('samples/extra_sample_%s_%d.jpg' % (id, i+1), img)
        #  r.append(np.concatenate(rr, 0))
    #  img = np.concatenate(r, 1)

    #  io.imshow(img)
    #  io.show()
