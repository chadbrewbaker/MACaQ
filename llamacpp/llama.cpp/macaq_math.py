"""MACAQ weight-only transform shared by HF and direct GGUF conversion."""
import math
import torch


def macaq_hadamard(b):
    H = torch.tensor([[(-1) ** bin(i & j).count('1') for j in range(b)]
                      for i in range(b)], dtype=torch.float32)
    return H / math.sqrt(b)


def macaq_transform_weight(w, pre_scale, had_block):
    out, n_in = w.shape
    if pre_scale.shape[0] != n_in:
        raise ValueError(f'pre_scale has {pre_scale.shape[0]} entries, expected {n_in}')
    if n_in % had_block:
        raise ValueError(f'input width {n_in} is not a multiple of {had_block}')
    scaled = w.float() / pre_scale.float()
    H = macaq_hadamard(had_block).to(scaled.device)
    return (scaled.reshape(out, n_in // had_block, had_block) @ H).reshape(out, n_in)
