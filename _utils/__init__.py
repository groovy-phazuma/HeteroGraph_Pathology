# -*- coding: utf-8 -*-
"""
Created on 2023-06-01 (Thu) 22:49:58

@author: I.Azuma
"""

import os
import random
import warnings

import dgl
import numpy as np
import torch

def set_seed(rndseed, cuda: bool = True, extreme_mode: bool = False):
    os.environ["PYTHONHASHSEED"] = str(rndseed)
    random.seed(rndseed)
    np.random.seed(rndseed)
    torch.manual_seed(rndseed)
    if cuda:
        torch.cuda.manual_seed(rndseed)
        torch.cuda.manual_seed_all(rndseed)
    if extreme_mode:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    dgl.seed(rndseed)
    dgl.random.seed(rndseed)
