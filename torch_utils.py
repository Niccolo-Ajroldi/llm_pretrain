
import os
import random
import numpy as np
import torch

from torch.distributed import init_process_group, destroy_process_group


def pytorch_setup(cfg):
  """Returns device, rank, seed, etc and initialize DDP"""
  ddp = int(os.environ.get('RANK', -1)) != -1  # check if DDP is enabled

  if ddp:
    init_process_group(backend='nccl')
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{local_rank}'
    torch.cuda.device(device)
    master_process = (rank == 0)
    seed_offset = rank
  else:
    master_process = True
    seed_offset = 0
    local_rank = None
    world_size = 1
    device = 'cpu'
    if torch.cuda.is_available():
      device = 'cuda'
    elif torch.backends.mps.is_available():
      device = 'mps'  # NOTE: macOS metal support to be tested

  random.seed(cfg.seed + seed_offset)
  np.random.seed(cfg.seed + seed_offset)
  torch.manual_seed(cfg.seed + seed_offset)
  torch.cuda.manual_seed(cfg.seed + seed_offset)
  
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True

  if cfg.determinisitc:
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    torch.backends.cudnn.benchmark = False
    
  return local_rank, world_size, device, master_process


def destroy_ddp():
  if torch.distributed.is_initialized():
    torch.distributed.barrier()
    destroy_process_group()

