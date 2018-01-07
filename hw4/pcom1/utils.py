#!/usr/bin/env python3
# coding=utf8
# version: 1.0.0

import re

hairTags =['orange hair',  'white hair', 'aqua hair',   'gray hair',
           'green hair',   'red hair',   'purple hair', 'pink hair',
           'blue hair',    'black hair', 'brown hair',  'blonde hair']

eyesTags = ['gray eyes',   'black eyes', 'orange eyes', 'pink eyes',
            'yellow eyes', 'aqua eyes',  'purple eyes', 'green eyes',
            'brown eyes',  'red eyes',   'blue eyes',   'padding']

toHairIdx = {v: i for i, v in enumerate(hairTags)}

toEyesIdx = {v: i for i, v in enumerate(eyesTags)}

import torch
from torch.autograd import Variable
from torch import Tensor

def createVariable(tensor, use_cuda, volatile=False, **kwargs):
    var = Variable(tensor, volatile=volatile, **kwargs)
    return var.cuda() if use_cuda else var

def toList(x):
    if isinstance(x, Variable):
        return x.data.cpu().numpy().tolist()
    if isinstance(x, Tensor):
        return x.cpu().numpy().tolist()

import numpy as np
from PIL import Image
def illum(x):
    r, g, b = x[:,:,0], x[:,:,1], x[:,:,2]
    l = r * 0.299 + g * 0.587 + b * 0.114
    return l

def toImage(x):
    x = x.transpose(1, 2, 0)
    if x.shape[2] == 3:
        I = illum(x)
    else:
        x = x.reshape(x.shape[:2])
        I = x
    img = np.clip(x, -1, 2)
    img = np.clip((img - I.mean()) / (I.std() + 1e-20) / 3 + 0.667, 0, 1)
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)

    org = np.clip(x, 0, 1)
    org = (org * 255).astype(np.uint8)
    org = Image.fromarray(org)

    return img, org
