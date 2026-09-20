import os
from datetime import timedelta

import numpy as np

import torch
from torch.distributed import is_initialized, init_process_group, destroy_process_group


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_ddp(rank: int, world_size: int):
    if world_size <= 1:
        return
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    init_process_group('nccl', rank=rank, world_size=world_size, timeout=timedelta(hours=1))


def cleanup_ddp():
    if is_initialized():
        destroy_process_group()