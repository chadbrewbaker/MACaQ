"""Tensor-parallel views of the saved full-expert checkpoint layout."""


def partition_expert_tensor(name, tensor, shape, rank, world):
    if tuple(tensor.shape) == tuple(shape):
        return tensor
    if world < 2 or not 0 <= rank < world:
        raise ValueError("expert tensor shape differs outside tensor parallelism")
    if name in ("w13_weight", "w13_weight_scale"):
        import torch
        full = tensor.shape[1] // 2
        if full % world:
            raise ValueError("expert output width is not divisible by tensor parallel size")
        part = full // world
        first = tensor[:, rank * part:(rank + 1) * part]
        second = tensor[:, full + rank * part:full + (rank + 1) * part]
        result = torch.cat((first, second), dim=1)
    elif name in ("w2_weight", "w2_weight_scale", "w2_pre_scale"):
        if tensor.shape[-1] % world:
            raise ValueError("expert input width is not divisible by tensor parallel size")
        result = tensor.chunk(world, dim=-1)[rank]
    else:
        raise ValueError(f"unsupported partition for expert tensor {name}")
    if tuple(result.shape) != tuple(shape):
        raise ValueError(f"partitioned expert shape differs: {name}")
    return result.contiguous()
