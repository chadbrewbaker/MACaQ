#!/usr/bin/env python3
"""MACAQ Hessian refinement of llama.cpp GGUF codes (dense Qwen3 and hybrid Qwen3.5-family).

Input: one or more GGUFs produced by `llama-quantize` (from the MACAQ weight-only f16 export,
or from a plain f16 export), the f16 GGUF they were quantized from (the target weights), and
the HF checkpoint of the original model (for calibration activations).

For every transformer block, in order, the input Hessian  H = sum x x^T  of every Linear is
accumulated over the calibration tokens (the block is streamed from safetensors; Linears that
share an input share the XᵀX) (unquantized forward, as MACAQ's fixed-H refinement
does per linear), each linear's Hessian is carried into MACAQ's frame,

    H~ = B^T diag(p) H diag(p) B     (p = pre_scale, B = blkdiag(Hadamard); identity for non-MACAQ files)

and the tensor's level codes are refined by MACAQ's greedy flip sweeps (one flip per output row
per sweep, delta = c^2 H_jj - 2 c g, g = H~ (w~ - w^)), with the block scales frozen -- exactly
the objective and procedure `macaq_mxfp4.flip_refine_levels` uses for the MXFP4 codes.
The refined codes are packed back bit-exactly and a new GGUF is written.

Supported tensor types: IQ4_NL, IQ4_XS, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS, IQ2_XS, IQ3_S, IQ3_XXS. Others pass through.
Block mode uses native lattice VQ for IQ2/IQ3 and joint product VQ for IQ4.
"""
from __future__ import annotations

import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gguf_codecs import GGUF_PY, QT, make_codec, SUPPORTED, GridCodec  # noqa: E402
from block_refine import block_gs_refine
from product_refine import product_gs_refine  # noqa: E402
if GGUF_PY not in sys.path:
    sys.path.insert(0, GGUF_PY)
import gguf  # noqa: E402
from gguf import GGUFReader, GGUFWriter, GGUFValueType  # noqa: E402

LOG = lambda *a: print(*a, flush=True)  # noqa: E731


# ---------------------------------------------------------------------------------------------
# MACAQ frame
# ---------------------------------------------------------------------------------------------
def hadamard(b: int, device) -> torch.Tensor:
    H = torch.tensor([[(-1) ** bin(i & j).count("1") for j in range(b)] for i in range(b)],
                     dtype=torch.float32, device=device)
    return H / math.sqrt(b)


def rot(M: torch.Tensor, BH: torch.Tensor) -> torch.Tensor:
    b = BH.shape[0]
    return (M.reshape(-1, b) @ BH).reshape(M.shape)


def to_macaq_frame(H: torch.Tensor, p: torch.Tensor | None, BH: torch.Tensor | None, ridge: float) -> torch.Tensor:
    Hf = H.float()
    if p is not None:
        Hf = Hf * p[None, :] * p[:, None]
    if BH is not None:
        Hf = rot(rot(Hf, BH).T, BH).T
    Hf = Hf.contiguous()
    if ridge > 0:
        Hf.diagonal().add_(ridge * Hf.diagonal().mean())
    return Hf


# ---------------------------------------------------------------------------------------------
# the flip refinement (MACAQ's, generalised to arbitrary move sets)
# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def h_loss(vals: torch.Tensor, W_t: torch.Tensor, H: torch.Tensor) -> float:
    E = vals - W_t
    return float(((E @ H) * E).sum())


@torch.no_grad()
def flip_refine(codec, st, W_t: torch.Tensor, H: torch.Tensor, n_iters: int):
    R, N = st.vals.shape
    dev = st.vals.device
    Hd = H.diagonal()
    XTyfp = H @ W_t.T                   # (N, R)
    XTyq = H @ st.vals.T                # (N, R)
    Hflat = H.reshape(-1)
    active = torch.arange(R, device=dev)
    n_flips, it = 0, 0
    INF = float("inf")
    for it in range(n_iters):
        na = active.numel()
        g = (XTyfp[:, active] - XTyq[:, active]).T                    # (na, N)
        alt = codec.alt_values(st, active)                            # (A, na, N), nan = no such move
        c = alt - st.vals[active][None]                               # (A, na, N)
        dl = torch.nan_to_num(c * c * Hd - 2 * c * g, nan=INF)
        A = dl.shape[0]
        dl_flat = dl.permute(1, 0, 2).reshape(na, A * N)
        best, arg = dl_flat.min(dim=1)
        a = arg // N; j = arg % N
        c_sel = c.permute(1, 0, 2).reshape(na, A * N).gather(1, arg[:, None]).squeeze(1)
        use_pair = None
        if codec.has_pairs:
            j1, j2, c1, c2 = codec.pair_candidates(st, active)        # (na, M)
            Hj = Hflat[j1 * N + j2]
            dlp = c1 * c1 * Hd[j1] + c2 * c2 * Hd[j2] + 2 * c1 * c2 * Hj - 2 * (c1 * g.gather(1, j1) + c2 * g.gather(1, j2))
            bestp, mp = dlp.min(dim=1)
            use_pair = bestp < best
            best = torch.where(use_pair, bestp, best)
        improving = best < -1e-12
        if not improving.any():
            break
        ss = improving if use_pair is None else (improving & ~use_pair)
        if ss.any():
            rows_s = active[ss]; js = j[ss]; cs = c_sel[ss]
            codec.apply(st, rows_s, js, a[ss])
            XTyq[:, rows_s] += H[:, js] * cs[None, :]
        if use_pair is not None:
            sp = improving & use_pair
            if sp.any():
                rows_p = active[sp]; m = mp[sp]; ar = torch.arange(na, device=dev)[sp]
                jj1 = j1[ar, m]; jj2 = j2[ar, m]; cc1 = c1[ar, m]; cc2 = c2[ar, m]
                codec.apply_pair(st, rows_p, m)
                XTyq[:, rows_p] += H[:, jj1] * cc1[None, :] + H[:, jj2] * cc2[None, :]
        n_flips += int(improving.sum())
        active = active[improving]
    return n_flips, it + 1


@torch.no_grad()
def flip_refine_chunked(codec, st, W_t, H, n_iters, max_elems):
    """Rows are independent in the objective, so refine them in row chunks to bound the
    transient memory (the sweep holds several (A, rows, N) float32 arrays)."""
    R, N = st.vals.shape
    A = 3
    rows_per = max(256, min(R, int(max_elems // (A * N))))
    if rows_per >= R:
        return flip_refine(codec, st, W_t, H, n_iters)
    fields = {k: v for k, v in st.__dict__.items() if torch.is_tensor(v) and v.dim() >= 1 and v.shape[0] == R}
    n_flips, sweeps = 0, 0
    for r0 in range(0, R, rows_per):
        rows = torch.arange(r0, min(r0 + rows_per, R), device=st.vals.device)
        sub = type(st)(None, st.vals[rows], **{k: v[rows] for k, v in fields.items() if k != "vals"})
        f, s_ = flip_refine(codec, sub, W_t[rows], H, n_iters)
        for k in fields:
            getattr(st, k)[rows] = getattr(sub, k)
        n_flips += f; sweeps = max(sweeps, s_)
        del sub
    return n_flips, sweeps


# ---------------------------------------------------------------------------------------------
# GGUF helpers
# ---------------------------------------------------------------------------------------------
def tensor_map(reader: GGUFReader):
    return {t.name: t for t in reader.tensors}


def field_scalar(reader: GGUFReader, key: str, default=None):
    f = reader.fields.get(key)
    if f is None:
        return default
    return f.contents()


def write_refined_gguf(reader: GGUFReader, out_path: str, refined: dict[str, np.ndarray], extra_kv: dict):
    arch = field_scalar(reader, gguf.Keys.General.ARCHITECTURE)
    writer = GGUFWriter(out_path, arch, endianess=reader.endianess)
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
    for k, v in extra_kv.items():
        if isinstance(v, bool):
            writer.add_bool(k, v)
        elif isinstance(v, int):
            writer.add_uint32(k, v)
        elif isinstance(v, float):
            writer.add_float32(k, v)
        else:
            writer.add_string(k, str(v))
    for t in reader.tensors:
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for t in reader.tensors:
        data = refined.get(t.name)
        if data is None:
            data = t.data
        else:
            assert data.dtype == np.uint8 and data.nbytes == t.data.nbytes, t.name
            data = data.reshape(t.data.shape)
        writer.write_tensor_data(data, tensor_endianess=reader.endianess)
    writer.close()


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------------------
# HF block streaming (architecture adapters)
# ---------------------------------------------------------------------------------------------
# HF Linear name inside a decoder layer -> GGUF short tensor name
LINEAR_MAP = {
    "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
    "self_attn.o_proj": "attn_output",
    "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up", "mlp.down_proj": "ffn_down",
    # Qwen3.5-family Gated DeltaNet
    "linear_attn.in_proj_qkv": "attn_qkv", "linear_attn.in_proj_z": "attn_gate",
    "linear_attn.in_proj_b": "ssm_beta", "linear_attn.in_proj_a": "ssm_alpha",
    "linear_attn.out_proj": "ssm_out",
}

QUANT_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "quantize"))


sys.path.insert(0, QUANT_DIR)
from text_stream import Streamer


def linear_hessians(block, dev):
    """Hooks on every nn.Linear of the block: H[name] = Σ xᵀx (fp32, shared across shared inputs)."""
    H, handles, cache = {}, [], {}
    for name, m in block.named_modules():
        if not isinstance(m, nn.Linear):
            continue
        H[name] = torch.zeros(m.in_features, m.in_features, dtype=torch.float32, device=dev)

        def _hook(_m, inp, _out, n=name):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1])
            key = x.data_ptr()
            hit = cache.get(key)
            if hit is None:
                xf = x.float()
                xtx = xf.T @ xf
                del xf
                cache[key] = (xtx, x)          # keep x alive: a freed buffer's address gets reused
            else:
                xtx = hit[0]
            H[n].add_(xtx)
        handles.append(m.register_forward_hook(_hook))
    return H, handles, cache


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf", required=True, help="HF checkpoint of the ORIGINAL model (bf16 weights, tokenizer)")
    ap.add_argument("--f16", required=True, help="the f16/bf16 GGUF the inputs were quantized from (target weights)")
    ap.add_argument("--inputs", nargs="+", required=True, help="quantized GGUFs to refine")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--suffix", default="-mrf", help="appended to each input's basename")
    ap.add_argument("--n-calib-tokens", type=int, default=262_144)
    ap.add_argument("--chunk-tokens", type=int, default=2048)
    ap.add_argument("--n-flip-iters", type=int, default=512)
    ap.add_argument("--method", choices=["scalar", "block"], default="block",
                    help="grid types (IQ3_S, IQ3_XXS): 'block' = exact Gauss-Seidel over 4-block codewords "
                         "(+ scale re-fit), 'scalar' = one-coordinate flips. Other types always use scalar.")
    ap.add_argument("--n-block-sweeps", type=int, default=6)
    ap.add_argument("--refit-scales", action=argparse.BooleanOptionalAction, default=True,
                    help="block method: also re-fit the 4-bit sub-block scale codes and the f16 super-block scale")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ridge", type=float, default=1.0, help="ridge * mean(diag H) added to the diagonal (MACAQ default 1.0)")
    ap.add_argument("--head", action=argparse.BooleanOptionalAction, default=True,
                    help="also refine output.weight (or the tied token_embd) against the final-norm Hessian")
    ap.add_argument("--max-layers", type=int, default=None, help="dev: only refine the first N blocks")
    ap.add_argument("--skip-types", default="", help="comma-separated tensor types to leave alone")
    ap.add_argument("--skip-tensors", default="", help="comma-separated GGUF short names to leave alone (e.g. ssm_out)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verbose", action="store_true", help="log every block sweep")
    ap.add_argument("--row-chunk-elems", type=float, default=1.5e8,
                    help="scalar method: rows are processed in chunks of at most this many elements")
    ap.add_argument("--calib", required=True,
                    help="local UTF-8 calibration text")
    args = ap.parse_args()

    if not os.path.isfile(args.calib):
        ap.error("--calib must be an existing local UTF-8 file")
    if args.n_block_sweeps < 1 or args.n_calib_tokens < 1 or args.chunk_tokens < 1:
        ap.error("calibration sizes and VQ sweeps must be positive")
    dev = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    os.makedirs(args.out_dir, exist_ok=True)
    skip_types = {QT[s.strip()] for s in args.skip_types.split(",") if s.strip()}
    skip_tensors = {s.strip() for s in args.skip_tensors.split(",") if s.strip()}

    # ---- GGUFs ----------------------------------------------------------------------------
    f16r = GGUFReader(args.f16)
    f16t = tensor_map(f16r)
    readers = [GGUFReader(p) for p in args.inputs]
    tmaps = [tensor_map(r) for r in readers]
    had_block = field_scalar(f16r, "macaq.hadamard_block")
    is_macaq = had_block is not None
    BH = hadamard(int(had_block), dev) if is_macaq else None
    LOG(f"MACAQ frame: {'hadamard_block=%d, pre_scale per linear' % had_block if is_macaq else 'none (plain model)'}")

    def target(name):
        t = f16t[name]
        assert t.tensor_type in (QT.F16, QT.F32, QT.BF16), (name, t.tensor_type)
        arr = np.asarray(t.data)
        if t.tensor_type == QT.BF16:
            arr = (arr.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        return torch.as_tensor(arr.astype(np.float32), device=dev)

    def pre_scale(name):
        if not is_macaq:
            return None
        ps = f16t.get(name.replace(".weight", ".pre_scale"))
        return None if ps is None else torch.as_tensor(np.asarray(ps.data).astype(np.float32), device=dev)

    # ---- HF model + calibration ----------------------------------------------------------
    t0 = time.time()
    S = Streamer(args.hf, dev, args.n_calib_tokens, args.chunk_tokens, args.calib)
    T = S.T
    n_layers = S.n_layers if args.max_layers is None else min(args.max_layers, S.n_layers)
    LOG(f"model + calibration loaded in {time.time()-t0:.0f}s: {T} tokens in {len(S.cached)} chunks; "
        f"{S.n_layers} layers ({S.text_config.model_type})")
    ssm_perm = S.ssm_out_col_perm() if hasattr(S.text_config, "linear_num_key_heads") else None

    refined = [dict() for _ in readers]           # per input: name -> new bytes
    stats = [dict() for _ in readers]
    codecs = {}

    def refine_one(ri, name, Hf):
        t = tmaps[ri].get(name)
        if t is None or t.tensor_type not in SUPPORTED or t.tensor_type in skip_types:
            return
        codec = codecs.get(t.tensor_type)
        if codec is None:
            codec = codecs[t.tensor_type] = make_codec(t.tensor_type, dev)
        raw = np.ascontiguousarray(t.data)
        st = codec.decode(raw)
        W_t = target(name)
        assert W_t.shape == st.vals.shape, (name, W_t.shape, st.vals.shape)
        ref = max(float(((W_t @ Hf) * W_t).sum()), 1e-30)
        l0 = h_loss(st.vals, W_t, Hf)
        tq = time.time()
        extra = {}
        if isinstance(codec, GridCodec) and args.method == "block":
            bs = block_gs_refine(codec, st, W_t, Hf, n_sweeps=args.n_block_sweeps, refit_scales=args.refit_scales,
                                 seed=args.seed, log=LOG if args.verbose else None)
            n_flips, sweeps = bs["moves"], bs["sweeps"]
            extra = dict(method="block", codebook_kind="lattice", vector_size=codec.vector_size,
                         codebook_size=len(codec.grid), refit_scales=bool(args.refit_scales), code_moves=bs["code_moves"], sign_moves=bs["sign_moves"],
                         scale_moves=bs["scale_moves"], d_moves=bs["d_moves"],
                         per_sweep=[dict(s, loss=s["loss"] / ref) for s in bs["per_sweep"]])
        elif t.tensor_type in (QT.IQ4_NL, QT.IQ4_XS) and args.method == "block":
            bs = product_gs_refine(codec, st, W_t, Hf, n_sweeps=args.n_block_sweeps,
                                   seed=args.seed, log=LOG if args.verbose else None)
            n_flips, sweeps = bs["moves"], bs["sweeps"]
            extra = dict(method="block", codebook_kind="product", vector_size=2, codebook_size=256,
                         refit_scales=False, code_moves=bs["code_moves"], sign_moves=0,
                         scale_moves=0, d_moves=0,
                         per_sweep=[dict(s, loss=s["loss"] / ref) for s in bs["per_sweep"]])
        else:
            n_flips, sweeps = flip_refine_chunked(codec, st, W_t, Hf, args.n_flip_iters, args.row_chunk_elems)
            extra = dict(method="scalar")
        l1 = h_loss(st.vals, W_t, Hf)
        refined[ri][name] = codec.encode(st)
        stats[ri][name] = dict(type=t.tensor_type.name, shape=list(st.vals.shape), loss_before=l0 / ref, loss_after=l1 / ref,
                               flips=n_flips, sweeps=sweeps, seconds=time.time() - tq, **extra)
        LOG(f"    [{os.path.basename(args.inputs[ri])[:28]:28s}] {name:26s} {t.tensor_type.name:8s} "
            f"{st.vals.shape[0]:6d}x{st.vals.shape[1]:<6d} H-loss {l0/ref:.4e} -> {l1/ref:.4e} ({((1-l1/l0)*100 if l0 else 0):5.1f}% less)  "
            f"flips {n_flips:8d}  sweeps {sweeps:3d}  {time.time()-tq:5.1f}s")
        del st, W_t

    checked_orientation = False
    for li in range(n_layers):
        block = S.load_block(li)
        H, handles, cache = linear_hessians(block, dev)
        t0 = time.time()
        with torch.no_grad():
            for c in S.cached:
                cache.clear()
                block(*S._args(c[0]), **c[1])
        cache.clear()
        for h in handles:
            h.remove()
        t_h = time.time() - t0
        LOG(f"block {li}: Hessians in {t_h:.1f}s")

        for hf_name, short in LINEAR_MAP.items():
            if hf_name not in H or short in skip_tensors:
                continue
            name = f"blk.{li}.{short}.weight"
            if not any(name in tm for tm in tmaps):
                continue
            Hb = H[hf_name]
            if short == "ssm_out" and ssm_perm is not None:
                Hb = Hb[ssm_perm][:, ssm_perm]
            p = pre_scale(name)
            if not checked_orientation and p is not None:
                W_t = target(name)
                Wh = dict(block.named_modules())[hf_name].weight.float()
                if short == "ssm_out" and ssm_perm is not None:
                    Wh = Wh[:, ssm_perm]
                pred = rot(Wh / p[None, :], BH)
                rel = float((pred - W_t).norm() / W_t.norm())
                LOG(f"  orientation check on {name}: |(W/p)B - W~| / |W~| = {rel:.2e}" + ("" if rel < 2e-2 else "  <-- MISMATCH"))
                assert rel < 2e-2, "GGUF target does not match the MACAQ transform of the HF weight"
                checked_orientation = True
                del W_t, Wh, pred
            # only linears the converter transformed carry a pre_scale; the others live in the plain frame
            Hf = to_macaq_frame(Hb, p, BH if p is not None else None, args.ridge)
            for ri in range(len(readers)):
                try:
                    refine_one(ri, name, Hf)
                except torch.OutOfMemoryError:
                    raise RuntimeError(f"OOM refining {name}; no checkpoint was published")
            del Hf, Hb
        del H
        torch.cuda.empty_cache()

        S.forward_all(block, keep=True)             # unquantized forward to the next block
        del block; torch.cuda.empty_cache()

    if args.head and (args.max_layers is None or args.max_layers >= S.n_layers):
        Hh = S.final_norm_hessian()
        Hf = to_macaq_frame(Hh, None, None, args.ridge)
        for ri, tm in enumerate(tmaps):
            name = "output.weight" if "output.weight" in tm else "token_embd.weight"
            LOG(f"head: refining {name} (no MACAQ frame; final-norm Hessian)")
            try:                                  # never lose the blocks' work over the head
                refine_one(ri, name, Hf)
            except torch.OutOfMemoryError:
                raise RuntimeError(f"OOM refining {name}; no checkpoint was published")
        del Hf, Hh

    for ri, r in enumerate(readers):
        base = os.path.splitext(os.path.basename(args.inputs[ri]))[0]
        out = os.path.join(args.out_dir, base + args.suffix + ".gguf")
        kinds = sorted({s.get("codebook_kind", "scalar") for s in stats[ri].values()})
        widths = sorted({s.get("vector_size", 1) for s in stats[ri].values()})
        write_refined_gguf(r, out, refined[ri], {"macaq.flip_refined": True, "macaq.flip_iters": args.n_flip_iters,
                                                 "macaq.flip_calib_tokens": T, "macaq.flip_method": args.method,
                                                 "macaq.vq_codebook_kind": ",".join(kinds),
                                                 "macaq.vq_vector_size": widths[0] if len(widths) == 1 else 0,
                                                 "macaq.vq_sweeps": args.n_block_sweeps,
                                                 "macaq.flip_refit_scales": any(s.get("refit_scales", False) for s in stats[ri].values())})
        tot = len(stats[ri])
        js = os.path.join(args.out_dir, base + args.suffix + ".json")
        json.dump(dict(inputs=args.inputs[ri], f16=args.f16, n_calib_tokens=T, n_flip_iters=args.n_flip_iters,
                       ridge=args.ridge, method=args.method, n_block_sweeps=args.n_block_sweeps,
                       refit_scales=any(s.get("refit_scales", False) for s in stats[ri].values()), tensors=stats[ri]), open(js, "w"), indent=1)
        LOG(f"wrote {out}  ({tot} tensors refined; stats in {js})")


if __name__ == "__main__":
    main()
