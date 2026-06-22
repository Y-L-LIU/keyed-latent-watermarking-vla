# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import torch
import torch.distributed as dist


def _configure_model(model, shard_fn, param_dtype, device, eval_mode=True):
    """
    TODO
    """
    if eval_mode:
        model.eval().requires_grad_(False)
    if dist.is_initialized():
        dist.barrier()

    if dist.is_initialized() and dist.get_world_size() > 1:
        model.to(param_dtype)
        model = shard_fn(model)
    elif hasattr(model, 'hf_device_map'):
        pass
    else:
        model.to(param_dtype)
        model.to(device)

    return model


def init_distributed(world_size, local_rank, rank):
    # if world_size > 1:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl",
                            init_method="env://",
                            rank=rank,
                            world_size=world_size)

def dist_mean(local_tensor):
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.AVG)
    return local_tensor

def dist_max(local_tensor):
    if dist.is_initialized():
        dist.all_reduce(local_tensor, op=dist.ReduceOp.MAX)
    return local_tensor
