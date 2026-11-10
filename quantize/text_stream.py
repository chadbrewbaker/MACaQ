"""Shared layer streaming for calibrated dense text quantization."""

import os
from pathlib import Path

import torch
from torch import nn

class Streamer:
    """Streams decoder layers from the safetensors checkpoint one at a time (no full model in RAM),
    with the layer-0 inputs (hidden states + the layer's kwargs) captured from the real embedding
    path, so hybrid layer signatures need no hand plumbing."""

    def __init__(self, hf_path: str, dev, n_calib_tokens: int, chunk_tokens: int, calib_spec: str):
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoTokenizer
        from mxfp4_helper import (load_weight_index, ShardCache, set_nested_attr,
                                  load_block_from_safetensors, capture_layer_input)
        self._load_block = load_block_from_safetensors
        self.dev = dev
        self.snapshot = Path(hf_path)
        self.weight_map = load_weight_index(self.snapshot)
        self.shards = ShardCache(self.snapshot)
        cfg = AutoConfig.from_pretrained(self.snapshot)
        self.text_config = getattr(cfg, "text_config", cfg)
        mt = self.text_config.model_type
        if mt in ("qwen3_5", "qwen3_5_text"):
            from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM as CLM, Qwen3_5DecoderLayer as Layer
            self.prefix = "model.language_model.layers."
            top = "model.language_model."
        elif mt == "qwen3":
            from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM as CLM, Qwen3DecoderLayer as Layer
            self.prefix = "model.layers."
            top = "model."
        else:
            raise SystemExit(f"no streaming adapter for model_type {mt!r}")
        if top + "embed_tokens.weight" not in self.weight_map:          # e.g. an older layout
            top = "model."
            self.prefix = "model.layers."
        self.Layer = Layer
        self.n_layers = self.text_config.num_hidden_layers
        self.chunk_tokens = chunk_tokens

        with init_empty_weights():
            self.model = CLM(self.text_config)
        self.model.eval()
        self.model.model.rotary_emb = type(self.model.model.rotary_emb)(config=self.text_config).to(dev)
        for k, attr in ((top + "embed_tokens.weight", "model.embed_tokens.weight"),
                        (top + "norm.weight", "model.norm.weight")):
            set_nested_attr(self.model, attr, self.shards.get(self.weight_map[k]).get_tensor(k).to(dev, torch.bfloat16))
        self.Hdim = self.text_config.hidden_size

        tok = AutoTokenizer.from_pretrained(self.snapshot)
        if os.path.isfile(calib_spec):
            text = open(calib_spec, encoding="utf-8").read()
        else:
            from datasets import load_dataset
            dpath, dcfg, dsplit = calib_spec.split(":")
            ds = load_dataset(dpath, dcfg, split=dsplit)
            text = "\n\n".join(t for t in ds["text"] if t.strip())
        ids = tok(text, return_tensors="pt", truncation=True, max_length=n_calib_tokens).input_ids.to(dev)
        self.T = ids.shape[1]
        chunks = [ids[:, i:i + chunk_tokens] for i in range(0, self.T, chunk_tokens)]

        sentinel = self.load_block(0)
        self.model.model.layers[0] = sentinel
        self.cached = [list(capture_layer_input(self.model.model, c, layer_idx=0)) for c in chunks]
        self.model.model.layers[0] = None
        for c in self.cached:
            c[0] = (self.host_tensor(c[0][0]),) + tuple(c[0][1:])
        self.model.model.embed_tokens.weight = nn.Parameter(torch.empty(0), requires_grad=False)
        del sentinel, ids, chunks
        torch.cuda.empty_cache()

    def host_tensor(self, tensor):
        tensor = tensor.cpu()
        return tensor.pin_memory() if self.dev.type == "cuda" else tensor

    def load_block(self, li: int) -> nn.Module:
        block = self.Layer(self.text_config, layer_idx=li).to(torch.bfloat16)
        self._load_block(block, li, self.weight_map, self.shards, self.prefix)
        return block.to(self.dev).eval()

    def _args(self, a):
        return (a[0].to(self.dev, non_blocking=True),) + tuple(a[1:])

    @torch.no_grad()
    def forward_all(self, block, keep: bool):
        """Run every chunk through `block`; with keep=True the outputs become the next block's inputs."""
        for c in self.cached:
            a, kw = c
            out = block(*self._args(a), **kw)
            if keep:
                h = out[0] if isinstance(out, tuple) else out
                c[0] = (self.host_tensor(h.detach()),) + tuple(a[1:])
            del out

    @torch.no_grad()
    def final_norm_hessian(self):
        norm = self.model.model.norm
        Hh = torch.zeros(self.Hdim, self.Hdim, dtype=torch.float32, device=self.dev)
        for a, _ in self.cached:
            x = norm(a[0].to(self.dev)).reshape(-1, self.Hdim).float()
            Hh.addmm_(x.T, x)
        return Hh

    def ssm_out_col_perm(self):
        """llama.cpp's converter reorders the V heads of the GDN tensors; for out_proj that permutes
        the INPUT axis, so the Hessian must be permuted the same way."""
        tc = self.text_config
        nk, nv = tc.linear_num_key_heads, tc.linear_num_value_heads
        if nk == nv:
            return None
        hd, per = tc.linear_value_head_dim, nv // nk
        idx = torch.arange(nv * hd).reshape(nk, per, hd).permute(1, 0, 2).reshape(-1)
        return idx.to(self.dev)
