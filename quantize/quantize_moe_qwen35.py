"""
quantize_moe_qwen35.py , real MACAQ MXFP4 W4A4 for Qwen3.5-MoE, emitting a
vLLM-loadable checkpoint (stock `mxfp4` quant_method + this repo's plugin).

This runs the SAME per-linear MACAQ method as the dense path (mxfp4_realquant_h:
block-Hadamard → exponent screen → flip-refinement), per expert, with two
MoE-specific choices:

  • Calibration (streaming, block-by-block so the full model never resides):
    each block is calibrated on the already-quantized upstream output; a forward
    hook captures the FULL set of tokens entering the experts and the router's
    softmax probs. `--calib_mode router` (default, best) weights each expert's
    Hessian by its router probability over all tokens,  H_e = Σ_t g_{t,e} xₜxₜᵀ,
    so no expert is starved. ('uniform' = one shared Hessian; 'router-hard' =
    only the tokens hard-routed to each expert.)

  • Per-expert D1, fully localized. Each expert linear is quantized EXACTLY like a
    dense linear: its own activation-aware diagonal p = w_std^α/act_rms^(1−α)
    (pre_mode="balance", derived from that expert's weight + router-weighted
    Hessian) folded into (x·p) and (W/p), plus a 32-wide block-Hadamard. No
    cross-expert sharing and no model-specific assumption , gate_up AND down both
    get the full (balance D1 + Hadamard + screen + flip) treatment. The vLLM
    grouped W4A4 kernel applies each expert's (p, Hadamard) to its routed
    activation inside the GEMM (gathered by expert id), so there is no separate
    online transform / extra kernel launch.

Forward (per expert e), with that expert's p_e and B = blkdiag(H_32):
    M_W(gate_up) = ((W/p_e)·B)   activation = ((x·p_e)·B)        → cancels
    M_W(down)    = ((W/q_e)·B)   activation = ((inter·q_e)·B)    → cancels

Saved tensors per block prefix `model.layers.{L}.mlp.experts.`:
    w13_weight [E,2I,H//2] u8, w13_weight_scale [E,2I,H//32] u8,
    w2_weight  [E,H,I//2]  u8, w2_weight_scale  [E,H,I//32]  u8,
    w13_pre_scale [E,H] bf16   (per-expert gate_up D1),
    w2_pre_scale  [E,I] bf16   (per-expert down D1)
plus all non-expert tensors (attn, norms, GDN, router, embeddings, lm_head) bf16.

config.json: architectures=["Qwen3_5MoeForCausalLM"], quantization_config marks
quant_method=mxfp4 + macaq_moe=True + rotation_block_size=32.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gc
import json
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm
from safetensors.torch import save_file
from checkpoint_config import portable_config

from mxfp4_helper import (
    CACHE_DIR, unfuse_moe_experts, QuantizedFusedExperts,
    find_snapshot_dir, load_weight_index, ShardCache,
    set_nested_attr, load_block_from_safetensors, capture_layer_input,
)
from macaq_mxfp4 import mxfp4_realquant_h, OCPDenseW4A4Linear, MXFP4_BLOCK

ST_BLOCKS_PREFIX = "model.language_model.layers."
ST_TEXT_TOPLEVEL = {
    "model.language_model.embed_tokens.weight": "model.embed_tokens.weight",
    "model.language_model.norm.weight":          "model.norm.weight",
    "lm_head.weight":                             "lm_head.weight",
}

HAD_BLOCK = 32          # block-Hadamard for both gate_up and down (= MXFP4 group)
SHARD_BYTES = 5_000_000_000


# ── per-expert MACAQ quant (CPU-importable; no transformers needed) ───────────
@torch.no_grad()
def _macaq_expert(W, H, n_tokens, had_block, cfg):
    """Run MACAQ on one expert Linear weight W [out,in] with Hessian H [in,in];
    return an OCPDenseW4A4Linear (which exposes unpadded W_packed/W_scales + the
    per-expert pre_scale, and applies the transform on its forward for the
    error-propagation re-forwards).

    Identical to the dense per-Linear method: pre_mode="balance" derives this
    expert's own activation-aware diagonal p = w_std^α/act_rms^(1−α) from W and H,
    folds it into (W/p) and (H·p·pᵀ), then rotates/screens/flips in that frame."""
    art = mxfp4_realquant_h(
        W.float(), H, n_tokens=n_tokens, had_block=had_block,
        n_flip_iters=cfg["n_flip_iters"], ridge=cfg["ridge"], do_flip=cfg["flip"],
        pre_mode="balance", pre_alpha=cfg["pre_alpha"],
        exp_screen=cfg["exp_screen"], exp_offsets=cfg["exp_offsets"],
        fast_flip=cfg["fast_flip"], compile_flip=False)
    out_f, in_f = W.shape
    return OCPDenseW4A4Linear(
        art["W_packed"], art["W_scales"], art["BH"], None, MXFP4_BLOCK,
        out_f, in_f, pre_scale=art["pre_scale"]).to(W.device)


@torch.no_grad()
def quantize_experts_ocp(emod, X, probs, cfg, device):
    """Quantize one experts module (in place: each per-expert Linear → an
    OCPDenseW4A4Linear, so the block can be re-forwarded quantized) and return the
    stacked vLLM tensors + the per-expert pre_scales.

      X      [T,H]  full set of tokens entering the experts (gate_up input)
      probs  [T,E] | None  router weighting for each expert's Hessian

    Each expert's gate_up and down are quantized like independent dense linears ,
    their own balance D1 (from that expert's router-weighted Hessian) + Hadamard.
    The per-expert diagonals are emitted (w13_pre_scale [E,H], w2_pre_scale [E,I])
    and applied per-expert inside the serving kernel; nothing is shared."""
    Xf = X.float()
    E = emod.num_experts
    w13_w, w13_s, w2_w, w2_s, w13_p, w2_p = [], [], [], [], [], []

    for e in range(E):
        gu, dp = emod.gate_up_proj[e], emod.down_proj[e]
        w = probs[:, e] if probs is not None else None

        # gate_up Hessian (router-weighted, or uniform if probs is None)
        if probs is not None:
            H_gu = (Xf * w.unsqueeze(1)).T @ Xf
            n_gu = float(w.sum().clamp_min(1.0))
        else:
            H_gu = Xf.T @ Xf
            n_gu = float(Xf.shape[0])

        # down input: push the FULL token set through this expert's fp gate_up
        # (BEFORE gate_up is replaced by its quantized version).
        gate_up_out = gu(X)
        g_act, u_act = gate_up_out.chunk(2, dim=-1)
        inter = (emod.act_fn(g_act) * u_act).float()
        del gate_up_out, g_act, u_act
        if probs is not None:
            H_dp = (inter * w.unsqueeze(1)).T @ inter
            n_dp = n_gu
        else:
            H_dp = inter.T @ inter
            n_dp = float(inter.shape[0])
        del inter

        wq_gu = _macaq_expert(gu.weight.data, H_gu, n_gu, HAD_BLOCK, cfg)
        wq_dp = _macaq_expert(dp.weight.data, H_dp, n_dp, HAD_BLOCK, cfg)
        emod.gate_up_proj[e] = wq_gu
        emod.down_proj[e] = wq_dp
        w13_w.append(wq_gu.W_packed.cpu()); w13_s.append(wq_gu.W_scales.cpu())
        w2_w.append(wq_dp.W_packed.cpu());  w2_s.append(wq_dp.W_scales.cpu())
        w13_p.append(wq_gu.pre_scale.cpu()); w2_p.append(wq_dp.pre_scale.cpu())
        del H_gu, H_dp

    return {
        "w13_weight":       torch.stack(w13_w, 0),
        "w13_weight_scale": torch.stack(w13_s, 0),
        "w2_weight":        torch.stack(w2_w, 0),
        "w2_weight_scale":  torch.stack(w2_s, 0),
        "w13_pre_scale":    torch.stack(w13_p, 0).contiguous(),   # [E,H] bf16
        "w2_pre_scale":     torch.stack(w2_p, 0).contiguous(),    # [E,I] bf16
    }


# ── MoE group discovery (experts module + sibling router) ─────────────────────
def collect_moe_groups(block):
    """[(parent, experts_name, experts_mod, router_mod)] for each unfused
    QuantizedFusedExperts paired with its sibling 'gate' router module."""
    experts, routers = {}, {}
    for name, mod in block.named_modules():
        if isinstance(mod, QuantizedFusedExperts):
            experts[name.rsplit('.', 1)[0] if '.' in name else ''] = (name, mod)
        elif name.rsplit('.', 1)[-1] == 'gate':
            routers[name.rsplit('.', 1)[0] if '.' in name else ''] = mod
    return [(p, en, em, routers.get(p)) for p, (en, em) in experts.items()]


@torch.no_grad()
def quantize_block_moe(block, block_idx, cached_inputs, cfg, device):
    """Capture each experts module's full input + router probs over the calib
    chunks, quantize every expert (in place), and return {moe_name: tensors}."""
    groups = collect_moe_groups(block)
    if not groups:
        return {}
    moe_in = {p: [] for p, *_ in groups}
    router_probs = {p: [] for p, *_ in groups}
    need_router = cfg["calib_mode"] in ("router", "router-hard")
    handles = []
    for parent, ename, emod, rmod in groups:
        def _ehook(_m, inp, _o, p=parent):
            x = inp[0]
            if x.numel():
                moe_in[p].append(x.detach().reshape(-1, x.shape[-1]).bfloat16().cpu())
        handles.append(emod.register_forward_hook(_ehook))
        if need_router and rmod is not None:
            def _rhook(_m, _i, out, p=parent):
                # Qwen3.5's `gate` is a plain Linear → raw LOGITS (not softmax).
                # Capture them; softmax-over-experts is applied once below.
                o = out[0] if isinstance(out, tuple) else out
                router_probs[p].append(o.detach().reshape(-1, o.shape[-1]).float().cpu())
            handles.append(rmod.register_forward_hook(_rhook))

    with torch.no_grad():
        for a, kw in cached_inputs:
            block(*a, **kw)
    for h in handles:
        h.remove()

    out = {}
    for parent, ename, emod, rmod in groups:
        if not moe_in[parent]:
            continue
        X = torch.cat(moe_in[parent], 0).to(device)
        probs = None
        if need_router and router_probs[parent]:
            logits = torch.cat(router_probs[parent], 0).to(device)    # [T,E] gate logits
            # Convert to the actual routing probabilities. Weighting the Hessian
            # by raw logits (here all-negative) makes it non-PSD → the flip /
            # exponent-screen optimize a garbage metric and destroy the weights.
            rp = torch.softmax(logits.float(), dim=-1)                # [T,E] probs
            del logits
            if cfg["calib_mode"] == "router":
                probs = rp
            else:                                                     # router-hard
                top_k = getattr(rmod, "top_k", rp.shape[-1])
                topi = rp.topk(top_k, dim=-1).indices
                probs = torch.zeros_like(rp).scatter_(1, topi, 1.0)
                del rp
        out[ename] = quantize_experts_ocp(emod, X, probs, cfg, device)
        del X, probs
        moe_in[parent].clear(); router_probs[parent].clear()
        gc.collect(); torch.cuda.empty_cache()
    return out


# ── collect non-expert (bf16) tensors of a quantized block ────────────────────
def collect_bf16_block_tensors(block, block_idx):
    out = {}
    prefix = f"model.layers.{block_idx}"
    skip = set()
    for name, mod in block.named_modules():
        if mod.__class__.__name__ == "QuantizedFusedExperts":
            skip.add(name + ".")
    for name, t in list(block.named_parameters(recurse=True)) + \
                    list(block.named_buffers(recurse=True)):
        if t is None or any(name.startswith(s) for s in skip):
            continue
        out[f"{prefix}.{name}"] = t.detach().to(torch.bfloat16).contiguous().cpu()
    return out


# ── sharded safetensors writer ────────────────────────────────────────────────
class ShardedSafetensorsWriter:
    def __init__(self, out_dir: Path, shard_bytes: int):
        self.out_dir = out_dir; out_dir.mkdir(parents=True, exist_ok=True)
        self.shard_bytes = shard_bytes
        self.buf = {}; self.cur = 0; self.idx = 0
        self.weight_map = {}; self.total = 0

    def _flush(self):
        if not self.buf:
            return
        tmp = f".pending-{self.idx:05d}.safetensors"
        save_file(self.buf, str(self.out_dir / tmp))
        for k in self.buf:
            self.weight_map[k] = tmp
        self.idx += 1; self.buf = {}; self.cur = 0

    def add(self, name, t):
        t = t.detach().cpu().contiguous()
        nb = t.numel() * t.element_size()
        if self.cur + nb > self.shard_bytes and self.buf:
            self._flush()
        self.buf[name] = t; self.cur += nb; self.total += nb

    def finalize(self):
        self._flush()
        rename = {}
        for old in list(self.weight_map.values()):
            if old in rename:
                continue
            i = int(old.split("-")[1].split(".")[0])
            rename[old] = f"model-{i+1:05d}-of-{self.idx:05d}.safetensors"
            (self.out_dir / old).rename(self.out_dir / rename[old])
        self.weight_map = {k: rename[v] for k, v in self.weight_map.items()}
        with open(self.out_dir / "model.safetensors.index.json", "w") as f:
            json.dump({"metadata": {"total_size": self.total},
                       "weight_map": self.weight_map}, f, indent=2)


def main():
    import argparse
    from accelerate import init_empty_weights
    from datasets import load_dataset
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeForCausalLM, Qwen3_5MoeDecoderLayer, Qwen3_5MoeTextRotaryEmbedding,
    )

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--model-id", default="Qwen/Qwen3.5-35B-A3B")
    p.add_argument("--out-dir", default="./qwen3.5-35b-a3b-macaq-mxfp4-w4a4")
    p.add_argument("--n-calib-tokens", type=int, default=256_000)
    p.add_argument("--chunk-tokens", type=int, default=2048)
    p.add_argument("--calib-mode", choices=["router", "uniform", "router-hard"],
                   default="router")
    p.add_argument("--flip", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--n-flip-iters", type=int, default=64)
    p.add_argument("--fast-flip", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ridge", type=float, default=1.0)
    p.add_argument("--exp-screen", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--exp-offsets", default="1,0,-1")
    p.add_argument("--pre-alpha", type=float, default=0.75)
    p.add_argument("--max-blocks", type=int, default=0, help="smoke-test cap; 0 = all")
    p.add_argument("--gpu", default="cuda:0")
    p.add_argument('--revision', help='source checkpoint revision')
    p.add_argument('--trust-remote-code', action='store_true', help='allow reviewed custom tokenizer code at an immutable remote revision')
    args = p.parse_args()

    cfg = dict(n_flip_iters=args.n_flip_iters, ridge=args.ridge, flip=args.flip,
               exp_screen=args.exp_screen, fast_flip=args.fast_flip,
               exp_offsets=tuple(int(x) for x in args.exp_offsets.split(",")),
               pre_alpha=args.pre_alpha, calib_mode=args.calib_mode)
    GPU = args.gpu
    out_dir = Path(args.out_dir)
    print(f"[macaq-moe] {args.model_id} → {out_dir}  "
          f"(calib={args.calib_mode}, n_calib={args.n_calib_tokens}, "
          f"flip={args.flip}, had_block={HAD_BLOCK})")

    from model_loading import loading_options
    source_options = loading_options(args.model_id, args.revision, args.trust_remote_code)
    if args.revision:
        from huggingface_hub import snapshot_download
        snapshot = Path(snapshot_download(args.model_id, revision=args.revision, cache_dir=CACHE_DIR))
    elif Path(args.model_id).is_dir():
        snapshot = Path(args.model_id)
    else:
        snapshot = find_snapshot_dir(args.model_id)
    weight_map = load_weight_index(snapshot)
    shards = ShardCache(snapshot)
    tok = AutoTokenizer.from_pretrained(snapshot, cache_dir=CACHE_DIR, **source_options)
    text_config = AutoConfig.from_pretrained(snapshot, cache_dir=CACHE_DIR, **source_options).text_config
    print(f"  hidden={text_config.hidden_size} layers={text_config.num_hidden_layers} "
          f"experts={text_config.num_experts} top_k={text_config.num_experts_per_tok}")

    with init_empty_weights():
        model = Qwen3_5MoeForCausalLM(text_config)
    model.eval()
    model.model.rotary_emb = Qwen3_5MoeTextRotaryEmbedding(config=text_config).to(GPU)
    for st_key in ("model.language_model.embed_tokens.weight",
                   "model.language_model.norm.weight"):
        if st_key in weight_map:
            f = shards.get(weight_map[st_key])
            set_nested_attr(model, ST_TEXT_TOPLEVEL[st_key],
                            f.get_tensor(st_key).to(GPU, dtype=torch.bfloat16))

    n_blocks = text_config.num_hidden_layers
    if args.max_blocks:
        n_blocks = min(n_blocks, args.max_blocks)

    # calibration chunks (WikiText-2 train). Dataset cache is separate from the
    # model cache (HF_DATASETS_CACHE), so a read-only HF_HUB_CACHE still works.
    train = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train",
                         cache_dir=os.environ.get("HF_DATASETS_CACHE") or None)
    ids = tok("\n\n".join(t for t in train["text"] if t.strip()),
              return_tensors="pt", truncation=True,
              max_length=args.n_calib_tokens).input_ids.to(GPU)
    calib_ids = [ids[:, i:i + args.chunk_tokens]
                 for i in range(0, ids.shape[1], args.chunk_tokens)]
    print(f"  calib {ids.shape[1]} tokens in {len(calib_ids)} chunks")

    # capture layer-0 input (+ kwargs) once via a sentinel block
    sentinel = Qwen3_5MoeDecoderLayer(text_config, layer_idx=0).to(torch.bfloat16)
    load_block_from_safetensors(sentinel, 0, weight_map, shards, ST_BLOCKS_PREFIX)
    sentinel = sentinel.to(GPU); sentinel.eval(); unfuse_moe_experts(sentinel, verbose=False)
    model.model.layers[0] = sentinel
    cached_inputs = [list(capture_layer_input(model.model, cid, layer_idx=0)) for cid in calib_ids]
    model.model.layers[0] = None
    del sentinel, ids; gc.collect(); torch.cuda.empty_cache()

    writer = ShardedSafetensorsWriter(out_dir, SHARD_BYTES)
    total_q = 0
    for li in tqdm(range(n_blocks), desc="quant"):
        fresh = Qwen3_5MoeDecoderLayer(text_config, layer_idx=li).to(torch.bfloat16)
        load_block_from_safetensors(fresh, li, weight_map, shards, ST_BLOCKS_PREFIX)
        fresh = fresh.to(GPU); fresh.eval(); unfuse_moe_experts(fresh, verbose=False)
        model.model.layers[li] = fresh

        moe = quantize_block_moe(fresh, li, cached_inputs, cfg, GPU)
        for ename, tensors in moe.items():
            for k, v in tensors.items():
                writer.add(f"model.layers.{li}.{ename}.{k}", v)
            total_q += tensors["w13_weight"].shape[0] * 2
        for name, t in collect_bf16_block_tensors(fresh, li).items():
            writer.add(name, t)

        with torch.no_grad():                       # propagate the quantized hidden
            for k, (a, kw) in enumerate(cached_inputs):
                out = fresh(*a, **kw)
                h = out[0] if isinstance(out, tuple) else out
                cached_inputs[k][0] = (h.detach().clone(),) + tuple(a[1:])

        model.model.layers[li] = None
        del fresh; gc.collect(); torch.cuda.empty_cache()

    # top-level bf16 tensors
    for st_key, mk in ST_TEXT_TOPLEVEL.items():
        if st_key in weight_map:
            f = shards.get(weight_map[st_key])
            writer.add(mk, f.get_tensor(st_key).to(torch.bfloat16).contiguous())
    shards.close()
    writer.finalize()

    cfg_out = portable_config(text_config.to_dict())
    cfg_out["architectures"] = ["Qwen3_5MoeForCausalLM"]
    cfg_out["quantization_config"] = {
        "quant_method": "mxfp4",
        "macaq_moe": True,
        "rotation_block_size": HAD_BLOCK,   # gate_up input rotation (plugin rebuilds H)
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg_out, f, indent=2)
    snap = Path(snapshot)
    for fn in os.listdir(snap):
        if fn.startswith(("tokenizer", "vocab", "merges", "special_tokens",
                          "added_tokens", "chat_template", "generation_config")):
            shutil.copy(snap / fn, out_dir / fn)
    print(f"[macaq-moe] DONE → {out_dir}  ({total_q} expert linears, {writer.idx} shards, "
          f"{writer.total/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
