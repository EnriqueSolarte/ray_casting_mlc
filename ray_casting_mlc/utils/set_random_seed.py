""" 
This code was taken form LGTNet 
https://github.com/zhigangjiang/LGT-Net/blob/main/models/other/init_env.py
"""
import random
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import os


def set_random_seed(seed=19880724):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHON_SEED'] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
