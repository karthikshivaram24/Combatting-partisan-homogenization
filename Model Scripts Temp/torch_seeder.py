"""
Seeds all torch and numpy functions that use some sort of randomization to generate values.
"""


import torch
import numpy as np
import random

np.random.seed(24101990)
RANDOM_SEED = np.random.randint(low=1, high=100000, size=1)[0]

def seed_all(seed):
    if not seed:
        seed = 10

    print("[ Using Seed : ", seed, " ]")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
