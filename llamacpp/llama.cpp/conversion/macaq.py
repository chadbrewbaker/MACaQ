from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from macaq_math import macaq_hadamard, macaq_transform_weight

if TYPE_CHECKING:
    from torch import Tensor

# MACAQ is a strictly per-linear MXFP4 W4A4 post-training quantizer. Its checkpoints
# hold, for every quantized nn.Linear:
#
#   <linear>.weight_packed  (out, in//2)  uint8  OCP MXFP4 nibbles
#   <linear>.weight_scale   (out, in//32) uint8  e8m0, floor(log2 amax) - 2 + 127
#   <linear>.pre_scale      (in,)         bf16   the D1 diagonal (optional)
#
# stored in the rotated basis M_W = (W / pre_scale) @ blkdiag(H). The e8m0 convention
# and the E2M1 codes are the same as ggml's `block_mxfp4`, so conversion is a nibble
# permutation plus carrying `pre_scale` through as a sibling tensor. The Hadamard is
# not stored: llama.cpp rebuilds it from `macaq.hadamard_block`.
#
# There is a second, weight-only mode. MACAQ's calibrated part is just the D1
# diagonal; the rotation is a canonical Sylvester Hadamard and the MXFP4 packing is
# a separate, replaceable step. So given the ORIGINAL bf16 weights and the D1 from a
# MACAQ run, the same transform can be applied at full precision,
#
#     M_W = (W / pre_scale) @ blkdiag(H)          in fp32, written as f16
#
# and llama.cpp's own weight-only quantizers (the k-quants, and the IQ types with
# their non-uniform level codebooks) take it from there via `llama-quantize`. The
# runtime hook is unchanged: `build_macaq_pre` keys off the presence of `pre_scale`,
# not off the weight's type, so activations get (x*p)*B either way. The backend
# selects activation arithmetic, including Q8 kernels for some quantized weights.
# This mode does not force MXFP4 activations and also supports 3-bit weights.


class _BaseWeights:
    """Lazy reader for the ORIGINAL model's `.weight` tensors, used by weight-only
    mode. Only opened when that mode is on, and one shard at a time."""

    def __init__(self, path: Path):
        from safetensors import safe_open

        self._safe_open = safe_open
        self.path = path
        shards = sorted(path.glob("*.safetensors"))
        if not shards:
            raise ValueError(f"no *.safetensors in {path}")
        index = path / "model.safetensors.index.json"
        if index.is_file():
            wmap = json.loads(index.read_text())["weight_map"]
            self.where = {k: path / v for k, v in wmap.items()}
        else:
            self.where = {}
            for s in shards:
                with safe_open(s, framework="pt") as h:
                    for k in h.keys():
                        self.where[k] = s

    def get(self, name: str) -> Tensor:
        if name not in self.where:
            raise ValueError(
                f"{name} not found in the base model at {self.path}. Weight-only mode "
                f"needs the ORIGINAL (unquantized) checkpoint the MACAQ run started "
                f"from — its D1 is only meaningful against those weights.")
        with self._safe_open(self.where[name], framework="pt") as h:
            return h.get_tensor(name)


# Set by convert_macaq_to_gguf.py --weight-only <base model>; None = MXFP4 W4A4.
_weight_only_base: Path | None = None


def set_weight_only_base(path) -> None:
    """Switch the converter into weight-only mode, transforming the original
    weights at full precision instead of repacking MACAQ's MXFP4 nibbles."""
    global _weight_only_base
    _weight_only_base = Path(path) if path is not None else None


def macaq_repack_mxfp4(packed: Tensor, scale: Tensor) -> Tensor:
    """MACAQ's OCP MXFP4 row layout -> ggml `block_mxfp4` rows.

    MACAQ packs a row as  byte[j] = nib(2j) | nib(2j+1) << 4  (even/odd interleave,
    running over the whole row), while ggml's 32-value block is

        struct { uint8_t e; uint8_t qs[16]; }   qs[j] = nib(j) | nib(j+16) << 4

    Same scale byte, same E2M1 codes, only the nibble positions move. Returns the
    raw bytes, shape (out, n_blocks*17), for `add_tensor(..., raw_dtype=MXFP4)`.
    """
    assert packed.dtype == torch.uint8 and scale.dtype == torch.uint8
    out, n_bytes = packed.shape
    n_blocks = scale.shape[1]
    assert 2*n_bytes == 32*n_blocks, f"{packed.shape} does not match {scale.shape}"
    assert scale.shape[0] == out

    # one 4-bit code per input column, in column order
    nibbles = torch.stack((packed & 0x0F, packed >> 4), dim=-1).reshape(out, n_blocks, 32)

    # re-pair them the way ggml does: element j with element j+16
    qs = nibbles[:, :, :16] | (nibbles[:, :, 16:] << 4)

    return torch.cat((scale.reshape(out, n_blocks, 1), qs), dim=-1).reshape(out, n_blocks*17)


class MacaqModel:
    """Mixin that teaches a model class to convert MACAQ MXFP4 W4A4 checkpoints.

    Mix it into any dense architecture whose linears map 1:1 onto GGUF tensor names
    (llama.cpp keeps q/k/v and gate/up separate, so each per-linear `pre_scale`
    carries over unchanged). `ModelBase.prepare_tensors` calls
    `_generate_macaq_tensors` when `quantization_config.macaq_dense_w4a4` is set.
    """

    def _generate_macaq_tensors(self):
        # imported here, not at module scope, so that macaq_repack_mxfp4 above stays
        # importable on its own (test/test_llamacpp_export.py checks it against
        # gguf-py's MXFP4 reference without pulling in the rest of the converter)
        from .base import LazyTorchTensor, gguf, logger

        quant_config = self.hparams.get("quantization_config") or {}
        had_block = int(quant_config.get("hadamard_block", 32))
        if had_block <= 0 or had_block & (had_block - 1):
            raise ValueError(f"MACAQ hadamard_block must be a power of two, got {had_block}")

        # marks the checkpoint as MACAQ for llama.cpp and carries the rotation width
        self.gguf_writer.add_macaq_hadamard_block(had_block)

        base_weights = _BaseWeights(_weight_only_base) if _weight_only_base else None

        # The checkpoint's quant_method is "mxfp4", which makes prepare_metadata stamp
        # the file as MOSTLY_MXFP4_MOE. In weight-only mode nothing we write is MXFP4,
        # so that label would be a lie -- and llama-quantize reads it to tell the two
        # MACAQ forms apart. prepare_tensors runs before prepare_metadata, so clearing
        # it here is enough. (See llama-quant.cpp: the MXFP4 form refuses requantizing.)
        if base_weights is not None:
            self._is_mxfp4 = False

        consumed: list[str] = []
        n_quant = 0
        n_pre = 0

        for name in list(self.model_tensors.keys()):
            if not name.endswith(".weight_packed"):
                continue

            base = name.removesuffix(".weight_packed")
            scale_name = f"{base}.weight_scale"
            pre_name = f"{base}.pre_scale"
            if scale_name not in self.model_tensors:
                raise ValueError(f"{name} has no matching {scale_name}")

            packed = LazyTorchTensor.to_eager(self.model_tensors[name]())
            n_in = 2*packed.shape[1]
            new_name = self.map_tensor_name(f"{base}.weight")

            if base_weights is None:
                scale = LazyTorchTensor.to_eager(self.model_tensors[scale_name]())
                raw = macaq_repack_mxfp4(packed, scale).numpy()
                logger.info(f"Repacked {new_name} with shape "
                            f"[{raw.shape[0]}, {n_in}] and quantization MXFP4")
                self.gguf_writer.add_tensor(new_name, raw,
                                            raw_dtype=gguf.GGMLQuantizationType.MXFP4)
            else:
                # weight-only: redo the transform on the ORIGINAL weights, in fp32.
                # Handing the result back through model_tensors (rather than writing
                # it here) lets the normal path name it, apply --outtype and log it,
                # and leaves the choice of 4-bit/3-bit level scheme to llama-quantize.
                # It MUST stay lazy: the writer pulls one tensor at a time, whereas
                # transforming eagerly here would hold every fp32 weight in RAM at once.
                pre_t = LazyTorchTensor.to_eager(self.model_tensors[pre_name]()).float() \
                    if pre_name in self.model_tensors else torch.ones(n_in)

                def _transformed(base=base, pre_t=pre_t, n_in=n_in):
                    w = base_weights.get(f"{base}.weight")
                    if w.shape[1] != n_in:
                        raise ValueError(f"{base}.weight is {tuple(w.shape)}, but the "
                                         f"MACAQ checkpoint says its input width is {n_in}")
                    return macaq_transform_weight(w, pre_t, had_block)

                self.model_tensors[f"{base}.weight"] = _transformed

            consumed += [name, scale_name]
            n_quant += 1

            # D1: one value per input channel. llama.cpp multiplies the activations by it
            # before the block-Hadamard, so it must survive as its own tensor. It is
            # always written, even for --pre_mode none, because llama.cpp uses its
            # presence to decide which linears need the MACAQ transform at all.
            if pre_name in self.model_tensors:
                pre = LazyTorchTensor.to_eager(self.model_tensors[pre_name]()).float()
                if pre.shape[0] != n_in:
                    raise ValueError(f"{pre_name} has {pre.shape[0]} entries, expected {n_in}")
                consumed.append(pre_name)
                n_pre += 1
            else:
                pre = torch.ones(n_in, dtype=torch.float32)
            self.gguf_writer.add_tensor(new_name.replace(".weight", ".pre_scale"), pre.numpy())

        if n_quant == 0:
            raise ValueError("quantization_config says macaq_dense_w4a4, but the checkpoint "
                             "has no *.weight_packed tensors")

        # save_macaq_vllm writes lm_head.weight even when the model ties it to the
        # embedding. vLLM ignores the copy; llama.cpp would not -- it would get a
        # separate `output` tensor, quantized under the output rule while token_embd
        # falls back to the per-type default, so the tied and untied files would differ
        # in both size and quality. Drop it, as the base model's own conversion has none.
        if self.hparams.get("tie_word_embeddings") and "lm_head.weight" in self.model_tensors:
            logger.info("MACAQ: tie_word_embeddings is set, skipping the saved lm_head.weight copy")
            consumed.append("lm_head.weight")

        for name in consumed:
            self.model_tensors.pop(name, None)

        kind = "weight-only" if base_weights is not None else "MXFP4"
        logger.info(f"MACAQ: {n_quant} {kind} linears ({n_pre} with a pre_scale), "
                    f"hadamard_block={had_block}")
        if base_weights is not None:
            logger.info("MACAQ: weight-only GGUF written unquantized — run "
                        "`llama-quantize` on it to pick a level scheme (e.g. Q4_K_M, "
                        "IQ4_NL, Q3_K_M, IQ3_S)")
