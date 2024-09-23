
import os
import torch

from torch import distributed as dist
from flax.training.checkpoints import latest_checkpoint

# TODO: load_checkpoint now loads a module! test with compile


def save_checkpoint(micro_step, engine, local_rank, cfg, job_idx=None):

  if dist.is_initialized() and cfg.fsdp: # FSDP enabled
    dist.barrier()  # make sure training is done on all ranks
  elif local_rank!=0:  # FSDP disabled, only rank 0 saves a checkpoint
    return

  # orig_model can be wrapped in DDP/FSDP, but is not torch-compiled
  model = engine.orig_model if cfg.torch_compile else engine.model

  optimizer = engine.optimizer
  scheduler = engine.scheduler
  scaler = engine.scaler

  state = {
    "micro_step": micro_step,
    "state_dict": model.state_dict(),
    "optimizer": optimizer.state_dict(),
    "scheduler": scheduler.state_dict() if scheduler is not None else {},
    "scaler": scaler.state_dict(),
  }

  if local_rank == 0:

    exp_dir = os.path.join(cfg.out_dir, cfg.exp_name)
    if job_idx is not None:  # subfolder for each job in the sweep
      exp_dir = os.path.join(exp_dir, f"job_idx_{job_idx}")

    save_path = os.path.join(exp_dir, f'ckpt_micro_step_{micro_step}.pth')
    torch.save(state, save_path)
    print(f"Checkpoint saved to {save_path}")


def maybe_load_checkpoint(cfg, device):
  
  ckpt = None
  micro_step_start = 0
  
  if cfg.resume:
    
    # resume from a specified exp or from the same exp
    exp_name = cfg.resume_exp_name if cfg.resume_exp_name is not None else cfg.exp_name
    ckpt_dir = os.path.join(cfg.out_dir, exp_name)
    print(f"Resuming from {ckpt_dir}")
    
    # resume from a specified checkpoint or from the latest
    if cfg.resume_micro_step is not None:
      ckpt_path = os.path.join(ckpt_dir, f'ckpt_micro_step_{cfg.resume_micro_step}.pth')
    else:
      ckpt_path = latest_checkpoint(ckpt_dir, prefix='ckpt_')
    
    # load checkpoint
    print(f"Loading checkpoint from {ckpt_path}")
    
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    micro_step_start = ckpt['micro_step']
  
  return ckpt, micro_step_start
