import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
import functools

def setup_dist(rank=None, world_size=None):
    """Initializes the distributed backend."""
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group("nccl")

def cleanup_dist():
    """Destroys the distributed process group."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def get_rank():
    """Returns the rank of the current process."""
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def get_world_size():
    """Returns the world size."""
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()

def is_main_process():
    """Returns True if the current process is the main process (rank 0)."""
    return get_rank() == 0

def reduce_tensor(tensor, op="mean"):
    """
    Reduces the tensor data across all machines.
    """
    if not dist.is_available() or not dist.is_initialized() or get_world_size() == 1:
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if op == "mean":
        rt /= get_world_size()
    return rt


def setup_fsdp(model, device, cfg):
    """
    Wraps the model in FSDP according to configuration.
    """
    min_num_params = cfg.get("training", {}).get("fsdp_min_num_params", 1e6)
    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=min_num_params)
    
    mixed_precision_dtype = cfg.get("training", {}).get("fsdp_mixed_precision_dtype", "float16")
    mp_dtype = getattr(torch, mixed_precision_dtype, torch.float32)
    mp_policy = MixedPrecision(
        param_dtype=mp_dtype,
        reduce_dtype=mp_dtype,
        buffer_dtype=mp_dtype,
    )

    sharding_type = cfg.get("training", {}).get("fsdp_sharding_strategy", "FULL_SHARD")
    sharding_strategy = getattr(ShardingStrategy, sharding_type, ShardingStrategy.FULL_SHARD)

    cpu_offload_enabled = cfg.get("training", {}).get("fsdp_cpu_offload", False)
    cpu_offload = CPUOffload(offload_params=cpu_offload_enabled)

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mp_policy,
        sharding_strategy=sharding_strategy,
        device_id=device,
        cpu_offload=cpu_offload,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=True
    )
    
    return fsdp_model
