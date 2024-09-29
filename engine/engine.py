"""
TorchEngine defines a training wrapper.
It takes care of DDP and FSDP routines, torch compile, grad scaler,
autocasting, clipping and gradient accumulation.
Ultimately, it defines a training step and an evaluation function.

NOTE: FSDP currently does not support gradient accumulation 
      outside no_sync() when using CPU offloading

TODO:
- use_orig_params=cfg.torch_compile
"""

from functools import partial
from contextlib import nullcontext

import torch

from torch import distributed as dist
from torch.nn import CrossEntropyLoss
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from models import get_param_groups
from optim import intialize_optimizer, initalize_scheduler

def _move_to_device(batch, seq_len, device):
  """Slice batch to get inputs and targets, and move them to device."""

  inputs = batch['input_ids'][:,:seq_len]
  targets = batch['input_ids'][:,1:(seq_len+1)]

  if 'cuda' in device:
    # pin arrays allows to move them to GPU asynchronously (non_blocking=True)
    inputs = inputs.pin_memory().to(device, non_blocking=True)
    targets = targets.pin_memory().to(device, non_blocking=True)
  else:
    inputs, targets = inputs.to(device), targets.to(device)

  return inputs, targets


class TorchEngine(torch.nn.Module):
  """
  A module containing model, optimizer, scheduler, grad scaler.
  Wraps together a training step. Takes care of grad accumulation.
  """
  def __init__(
      self,
      model,
      cfg,
      device,
      local_rank,
      ckpt,
      ):
    super().__init__()

    self.micro_steps = 0
    self.accumulated_samples = 0

    self.seq_len = cfg.seq_len
    self.accumulation_steps = cfg.grad_accumulation_steps
    self.grad_clip = cfg.grad_clip
    self.fsdp = cfg.fsdp
    self.dtype = cfg.dtype

    self.device = device
  
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]

    # Load model state dict
    if cfg.resume:
      model.load_state_dict(ckpt['state_dict'])
      self.micro_steps = ckpt['micro_step']

    # Move model to device and to DDP
    self.model = model.to(device)
    if torch.distributed.is_initialized():
      if self.fsdp:
        print(f"Wrapping model with FSDP.")
        my_auto_wrap_policy = partial(
          size_based_auto_wrap_policy, min_num_params=100
        )
        model = FSDP(
          model, 
          auto_wrap_policy=my_auto_wrap_policy,
          mixed_precision=MixedPrecision(ptdtype),
          use_orig_params=cfg.torch_compile,  # TODO
        )        
      else:
        print(f"FSDP disabled, defaulting to DDP.")
        self.model = DDP(self.model, device_ids=[local_rank])

    # Compile
    if cfg.torch_compile:
      print(f"Compiling model.")
      self.orig_model = self.model
      self.model = torch.compile(self.model)

    # AMP
    self.ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    # Grad scaler if training in fp16, if enabled=False, scaler is a no-op
    self.scaler = torch.amp.GradScaler(enabled=(self.dtype == 'float16'))

    # Loss
    self.criterion = CrossEntropyLoss()

    # Optimizer
    param_groups = get_param_groups(model, cfg.weight_decay)
    self.optimizer = intialize_optimizer(param_groups, cfg)
    self.scheduler = initalize_scheduler(self.optimizer, cfg)

    if cfg.resume:
      self.optimizer.load_state_dict(ckpt['optimizer'])
      self.scheduler.load_state_dict(ckpt['scheduler'])
      self.scaler.load_state_dict(ckpt['scaler'])


  def step(self, batch):
    """Wraps a fwd pass, backwd pass, and optimization step."""

    self.model.train()

    self.micro_steps += 1
    self.accumulated_samples += 1

    inputs, targets = _move_to_device(batch, self.seq_len, self.device)

    # sync gradients at the last accumulation step
    if torch.distributed.is_initialized():
      self.model.require_backward_grad_sync = \
        (self.accumulated_samples == self.accumulation_steps)

    # forward pass with autocasting
    with self.ctx:
      output = self.model(inputs)
      logits = getattr(output, 'logits', output)
      loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
      loss = loss / self.accumulation_steps

    # detach for logging (scale up to undo the division above)
    loss_val = loss.detach() * self.accumulation_steps
    if torch.isnan(loss_val):
      raise ValueError("Train loss is nan")

    # backward pass, with gradient scaling if training in fp16
    self.scaler.scale(loss).backward()

    if self.grad_clip:
      self.scaler.unscale_(self.optimizer)
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

    # step after accumulation
    if self.accumulated_samples == self.accumulation_steps:
      self.accumulated_samples = 0

      # step the optimizer, step the scaler if training in fp16
      self.scaler.step(self.optimizer)
      self.scaler.update()

      # flush the gradients
      self.optimizer.zero_grad(set_to_none=True) 

      # step the scheduler
      if self.scheduler:
        self.scheduler.step()

    return loss_val


  @torch.no_grad()
  def eval(self, validloader):
    """Evaluate model on a dataloader."""

    self.model.eval()

    # Compute loss on validloader
    total_loss = 0.0
    num_batches = 0
    for batch in validloader:
      inputs, targets = _move_to_device(batch, self.seq_len, self.device)

      with self.ctx:
        output = self.model(inputs)
        logits = getattr(output, 'logits', output)
        loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        if torch.isnan(loss) or loss is None:
          raise ValueError("Validation loss is nan")

        total_loss += loss.item()
        num_batches += 1

    # reduce loss across processes
    if dist.is_initialized():
      loss_num_batches = torch.tensor([total_loss, num_batches], device=self.device)
      dist.all_reduce(loss_num_batches)
      total_loss = loss_num_batches[0].item() / dist.get_world_size()
      num_batches = int(loss_num_batches[1].item()) // dist.get_world_size()  # superflous redundant if dataloader has drop_last=True

    # Calculate average loss
    avg_loss = total_loss / num_batches

    return avg_loss
