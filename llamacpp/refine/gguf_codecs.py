"""Code-level codecs for llama.cpp quant types, for MACAQ flip refinement.

Each codec turns the raw bytes of one quantized tensor (rows x bytes_per_row) into a
`State` of device tensors -- the dequantized values plus the discrete codes behind them --
exposes the *moves* the flip refinement may make (one weight to an adjacent representable
level, a sign flip, or for IQ3_XXS a parity-coupled pair of sign flips), and packs the
codes back into bytes.  Block scales and mins are never touched: `encode` copies the
original bytes and overwrites only the code fields.

IQ2 codecs below implement eight-dimensional native lattice codewords.
IQ4 storage is scalar; joint product-codebook assignment lives in product_refine.py.

Formats:  table types (value = scale * table[code] - min): IQ4_NL, IQ4_XS, Q4_K, Q5_K, Q6_K, Q3_K
          grid types  (4 weights share a grid point, signs separate): IQ3_S, IQ3_XXS
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
import torch

GGUF_PY = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "llama.cpp", "gguf-py"))
if GGUF_PY not in sys.path:
    sys.path.insert(0, GGUF_PY)
from gguf import GGMLQuantizationType as QT, GGML_QUANT_SIZES  # noqa: E402
from gguf import quants as gq  # noqa: E402

QK_K = 256
NAN = float("nan")


class State:
    """Per-tensor refinement state. `vals` is always the exact dequantized tensor."""

    def __init__(self, raw: np.ndarray, vals: torch.Tensor, **fields):
        self.raw = raw                      # (R, nbytes) uint8, original bytes (scales are taken from here)
        self.vals = vals                    # (R, N) float32 on device
        self.__dict__.update(fields)

    @property
    def shape(self):
        return self.vals.shape


def _f16(b: np.ndarray) -> np.ndarray:
    return b.copy().view(np.float16).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# table types: value = se * T[q] - me
# ----------------------------------------------------------------------------------------------
class TableCodec:
    has_pairs = False

    def __init__(self, qtype: QT, device):
        self.qtype = qtype
        self.device = device
        self.block_size, self.type_size = GGML_QUANT_SIZES[qtype]
        tbl = {
            QT.IQ4_NL: gq.IQ4_NL.kvalues, QT.IQ4_XS: gq.IQ4_NL.kvalues,
            QT.Q4_K: range(16), QT.Q5_K: range(32), QT.Q6_K: range(-32, 32), QT.Q3_K: range(-4, 4),
        }[qtype]
        self.table = torch.tensor(list(tbl), dtype=torch.float32, device=device)
        self.n_levels = len(self.table)

    # -- unpack --------------------------------------------------------------------------------
    def decode(self, raw: np.ndarray) -> State:
        R, nbytes = raw.shape
        nb = nbytes // self.type_size
        blocks = raw.reshape(R, nb, self.type_size)
        q, se, me = getattr(self, f"_unpack_{self.qtype.name}")(blocks, R, nb)
        dev = self.device
        q = torch.as_tensor(q.astype(np.int16), device=dev)
        se = torch.as_tensor(np.ascontiguousarray(se, dtype=np.float32), device=dev)
        me = torch.as_tensor(np.ascontiguousarray(me, dtype=np.float32), device=dev)
        vals = se * self.table[q.long()] - me
        return State(raw, vals, q=q, se=se, me=me)

    def _unpack_IQ4_NL(self, b, R, nb):
        d = _f16(b[:, :, :2].reshape(R, nb, 2)).reshape(R, nb)
        qs = b[:, :, 2:]                                              # (R, nb, 16)
        q = np.concatenate([qs & 0x0F, qs >> 4], axis=-1).reshape(R, nb * 32)
        se = np.repeat(d, 32, axis=1)
        return q, se, np.zeros_like(se)

    def _unpack_IQ4_XS(self, b, R, nb):
        d = _f16(b[:, :, 0:2]).reshape(R, nb)
        sh = b[:, :, 2:4].copy().view(np.uint16).reshape(R, nb)
        sl = b[:, :, 4:8]                                             # (R, nb, 4)
        qs = b[:, :, 8:].reshape(R, nb, 8, 16)
        s = np.arange(8)
        lo = (sl[:, :, s // 2] >> (4 * (s % 2))) & 0x0F               # (R, nb, 8)
        hi = ((sh[:, :, None] >> (2 * s)) & 3).astype(np.uint8)
        sc = (lo | (hi << 4)).astype(np.int8).astype(np.float32) - 32
        q = np.concatenate([qs & 0x0F, qs >> 4], axis=-1).reshape(R, nb * 256)
        se = np.repeat((d[:, :, None] * sc).reshape(R, nb * 8), 32, axis=1)
        return q, se, np.zeros_like(se)

    @staticmethod
    def _scale_min(scales):                                          # Q4_K / Q5_K 12-byte packing
        sc = scales.reshape(*scales.shape[:-1], 3, 4)
        d, m, md = sc[..., 0, :], sc[..., 1, :], sc[..., 2, :]
        s = np.concatenate([d & 0x3F, (md & 0x0F) | ((d >> 2) & 0x30)], axis=-1)
        mn = np.concatenate([m & 0x3F, (md >> 4) | ((m >> 2) & 0x30)], axis=-1)
        return s.astype(np.float32), mn.astype(np.float32)

    def _unpack_Q4_K(self, b, R, nb):
        d = _f16(b[:, :, 0:2]).reshape(R, nb); dmin = _f16(b[:, :, 2:4]).reshape(R, nb)
        sc, mn = self._scale_min(b[:, :, 4:16])
        qs = b[:, :, 16:].reshape(R, nb, 4, 32)
        q = np.stack([qs & 0x0F, qs >> 4], axis=3).reshape(R, nb * 256)   # (p, nib) -> sub-block 2p+nib
        se = np.repeat((d[:, :, None] * sc).reshape(R, nb * 8), 32, axis=1)
        me = np.repeat((dmin[:, :, None] * mn).reshape(R, nb * 8), 32, axis=1)
        return q, se, me

    def _unpack_Q5_K(self, b, R, nb):
        d = _f16(b[:, :, 0:2]).reshape(R, nb); dmin = _f16(b[:, :, 2:4]).reshape(R, nb)
        sc, mn = self._scale_min(b[:, :, 4:16])
        qh = b[:, :, 16:48]                                           # (R, nb, 32)
        qs = b[:, :, 48:].reshape(R, nb, 4, 32)
        ql = np.stack([qs & 0x0F, qs >> 4], axis=3).reshape(R, nb, 8, 32)
        hb = ((qh[:, :, None, :] >> np.arange(8)[None, None, :, None]) & 1).astype(np.uint8)  # (R, nb, 8, 32)
        q = (ql | (hb << 4)).reshape(R, nb * 256)
        se = np.repeat((d[:, :, None] * sc).reshape(R, nb * 8), 32, axis=1)
        me = np.repeat((dmin[:, :, None] * mn).reshape(R, nb * 8), 32, axis=1)
        return q, se, me

    def _unpack_Q6_K(self, b, R, nb):
        ql = b[:, :, 0:128].reshape(R, nb, 2, 64)
        qh = b[:, :, 128:192].reshape(R, nb, 2, 32)
        sc = b[:, :, 192:208].copy().view(np.int8).astype(np.float32)   # (R, nb, 16)
        d = _f16(b[:, :, 208:210]).reshape(R, nb)
        lo = np.stack([ql & 0x0F, ql >> 4], axis=3).reshape(R, nb, 2, 128)            # w = h*128 + nib*64 + j
        hi = np.stack([(qh >> s) & 3 for s in (0, 2, 4, 6)], axis=3).reshape(R, nb, 2, 128)  # w = h*128 + k*32 + j
        q = (lo | (hi << 4)).reshape(R, nb * 256)                     # 0..63, table = q - 32
        se = np.repeat((d[:, :, None] * sc).reshape(R, nb * 16), 16, axis=1)
        return q, se, np.zeros_like(se)

    def _unpack_Q3_K(self, b, R, nb):
        hm = b[:, :, 0:32]; qs = b[:, :, 32:96].reshape(R, nb, 2, 32); scales = b[:, :, 96:108]
        d = _f16(b[:, :, 108:110]).reshape(R, nb)
        lo = np.stack([(qs >> s) & 3 for s in (0, 2, 4, 6)], axis=3).reshape(R, nb, 256)      # w = h*128 + k*32 + j
        hb = np.stack([(hm >> k) & 1 for k in range(8)], axis=2).reshape(R, nb, 256)        # w = k*32 + j
        q = ((lo.astype(np.int16) - 4 * (1 - hb.astype(np.int16))) + 4).reshape(R, nb * 256)  # 0..7, table = q - 4
        i = np.arange(16)
        l = (scales[:, :, i % 8] >> (4 * (i // 8))) & 0x0F
        h = (scales[:, :, 8 + i % 4] >> (2 * (i // 4))) & 0x03
        sc = (l | (h << 4)).astype(np.int8).astype(np.float32) - 32
        se = np.repeat((d[:, :, None] * sc).reshape(R, nb * 16), 16, axis=1)
        return q, se, np.zeros_like(se)

    # -- moves ---------------------------------------------------------------------------------
    def alt_values(self, st: State, rows: torch.Tensor) -> torch.Tensor:
        q = st.q[rows].long(); se = st.se[rows]; me = st.me[rows]
        up = torch.where(q + 1 < self.n_levels, se * self.table[(q + 1).clamp(max=self.n_levels - 1)] - me, torch.full_like(se, NAN))
        dn = torch.where(q - 1 >= 0, se * self.table[(q - 1).clamp(min=0)] - me, torch.full_like(se, NAN))
        return torch.stack([up, dn])

    def apply(self, st: State, rows, cols, a):
        step = torch.where(a == 0, 1, -1).to(torch.int16)
        st.q[rows, cols] += step
        st.vals[rows, cols] = st.se[rows, cols] * self.table[st.q[rows, cols].long()] - st.me[rows, cols]

    # -- pack ----------------------------------------------------------------------------------
    def encode(self, st: State) -> np.ndarray:
        raw = st.raw.copy()
        R, nbytes = raw.shape; nb = nbytes // self.type_size
        b = raw.reshape(R, nb, self.type_size)
        q = st.q.cpu().numpy().astype(np.uint8)
        getattr(self, f"_pack_{self.qtype.name}")(b, q, R, nb)
        return raw

    def _pack_IQ4_NL(self, b, q, R, nb):
        q = q.reshape(R, nb, 32)
        b[:, :, 2:] = q[:, :, :16] | (q[:, :, 16:] << 4)

    def _pack_IQ4_XS(self, b, q, R, nb):
        q = q.reshape(R, nb, 8, 32)
        b[:, :, 8:] = (q[:, :, :, :16] | (q[:, :, :, 16:] << 4)).reshape(R, nb, 128)

    def _pack_Q4_K(self, b, q, R, nb):
        q = q.reshape(R, nb, 4, 2, 32)
        b[:, :, 16:] = (q[:, :, :, 0, :] | (q[:, :, :, 1, :] << 4)).reshape(R, nb, 128)

    def _pack_Q5_K(self, b, q, R, nb):
        q = q.reshape(R, nb, 8, 32)
        lo = (q & 0x0F).reshape(R, nb, 4, 2, 32)
        b[:, :, 48:] = (lo[:, :, :, 0, :] | (lo[:, :, :, 1, :] << 4)).reshape(R, nb, 128)
        hb = (q >> 4) & 1                                              # (R, nb, 8, 32)
        b[:, :, 16:48] = sum((hb[:, :, k, :] << k) for k in range(8)).astype(np.uint8)

    def _pack_Q6_K(self, b, q, R, nb):
        q = q.reshape(R, nb, 2, 2, 64)                                 # (h, nib, j) for the low nibbles
        lo = q & 0x0F
        b[:, :, 0:128] = (lo[:, :, :, 0, :] | (lo[:, :, :, 1, :] << 4)).reshape(R, nb, 128)
        hi = ((q >> 4) & 3).reshape(R, nb, 2, 4, 32)                   # (h, k, j)
        b[:, :, 128:192] = sum((hi[:, :, :, k, :] << (2 * k)) for k in range(4)).astype(np.uint8).reshape(R, nb, 64)

    def _pack_Q3_K(self, b, q, R, nb):
        q = q.reshape(R, nb, 256).astype(np.int16)                     # 0..7
        lo = (q & 3).astype(np.uint8).reshape(R, nb, 2, 4, 32)         # (h, k, j)
        hb = (q >> 2).astype(np.uint8).reshape(R, nb, 8, 32)           # (k, j)
        b[:, :, 32:96] = sum((lo[:, :, :, k, :] << (2 * k)) for k in range(4)).astype(np.uint8).reshape(R, nb, 64)
        b[:, :, 0:32] = sum((hb[:, :, k, :] << k) for k in range(8)).astype(np.uint8)


# ----------------------------------------------------------------------------------------------
# grid types: 4 weights share a grid point; value = se * grid[idx][k] * sign
# ----------------------------------------------------------------------------------------------
def _grid_of(cls) -> np.ndarray:
    cls.init_grid()
    return cls.grid.reshape(cls.grid_shape).astype(np.float32)


def build_neighbors(grid: np.ndarray):
    """For grid point i and coordinate k: index of the grid point equal in the other three
    coordinates with the next larger (up) / smaller (dn) value in k, or -1."""
    G, K = grid.shape
    up = -np.ones((G, K), dtype=np.int64); dn = -np.ones((G, K), dtype=np.int64)
    for k in range(K):
        groups = defaultdict(list)
        others = [kk for kk in range(K) if kk != k]
        for i in range(G):
            groups[tuple(grid[i, others])].append(i)
        for members in groups.values():
            members.sort(key=lambda i: grid[i, k])
            for a, i in enumerate(members):
                if a + 1 < len(members): up[i, k] = members[a + 1]
                if a > 0: dn[i, k] = members[a - 1]
    return up, dn


class GridCodec:
    vector_size = 4
    scale_group_size = 32

    def __init__(self, qtype: QT, device):
        self.qtype = qtype
        self.device = device
        self.block_size, self.type_size = GGML_QUANT_SIZES[qtype]
        cls = {QT.IQ3_S: gq.IQ3_S, QT.IQ3_XXS: gq.IQ3_XXS}[qtype]
        grid = _grid_of(cls)                                           # (G, 4)
        up, dn = build_neighbors(grid)
        self.grid = torch.as_tensor(grid, device=device)
        self.nb_up = torch.as_tensor(up, device=device)
        self.nb_dn = torch.as_tensor(dn, device=device)
        self.has_pairs = qtype == QT.IQ3_XXS
        if self.has_pairs:
            self.ksigns = np.frombuffer(gq.IQ2_XXS.ksigns, dtype=np.uint8)

    def decode(self, raw: np.ndarray) -> State:
        R, nbytes = raw.shape; nb = nbytes // self.type_size
        b = raw.reshape(R, nb, self.type_size)
        d = _f16(b[:, :, 0:2]).reshape(R, nb)
        if self.qtype == QT.IQ3_S:
            qs = b[:, :, 2:66].astype(np.int64); qh = b[:, :, 66:74]; signs = b[:, :, 74:106]; scales = b[:, :, 106:110]
            g = np.arange(64)
            idx = qs | (((qh[:, :, g // 8] >> (g % 8)) & 1).astype(np.int64) << 8)      # (R, nb, 64)
            w = np.arange(256)
            neg = (signs[:, :, w // 8] >> (w % 8)) & 1                                    # (R, nb, 256)
            s = np.arange(8)
            sc = (scales[:, :, s // 2] >> (4 * (s % 2))) & 0x0F
            db = d[:, :, None] * (1 + 2 * sc.astype(np.float32))                          # (R, nb, 8)
        else:  # IQ3_XXS
            qs = b[:, :, 2:66].astype(np.int64)
            sas = b[:, :, 66:98].copy().view(np.uint32).reshape(R, nb, 8)                 # per 32-sub-block
            idx = qs
            sc = (sas >> 28).astype(np.float32)
            db = d[:, :, None] * (0.5 + sc) * 0.5
            sidx = np.stack([(sas >> (7 * t)) & 0x7F for t in range(4)], axis=3)         # (R, nb, 8, 4) -> group t of sub-block
            sbyte = self.ksigns[sidx]                                                     # (R, nb, 8, 4)
            neg = np.stack([(sbyte >> r) & 1 for r in range(8)], axis=4).reshape(R, nb, 256)
        dev = self.device
        idx_t = torch.as_tensor(idx.reshape(R, nb * 64), device=dev)
        sgn = torch.as_tensor(np.where(neg.reshape(R, nb * 256) == 1, -1.0, 1.0).astype(np.float32), device=dev)
        sc_t = torch.as_tensor(sc.reshape(R, nb * 8).astype(np.int64), device=dev)          # 4-bit sub-block scale codes
        d_t = torch.as_tensor(d.astype(np.float32), device=dev)                              # (R, nb) f16 super-block scale
        se = torch.repeat_interleave(self.sub_scale(d_t, sc_t), 32, dim=1)                   # (R, N)
        mag = self.grid[idx_t].reshape(R, nb * 256)
        vals = se * mag * sgn
        return State(raw, vals, idx=idx_t, sgn=sgn, se=se, sc=sc_t, d=d_t)

    # scale codes -> per-32-weight sub-block scale
    def scale_factor(self, sc: torch.Tensor) -> torch.Tensor:
        if self.qtype == QT.IQ3_S:
            return (1 + 2 * sc).to(torch.float32)
        return (0.5 + sc.to(torch.float32)) * 0.5

    def sub_scale(self, d: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
        R, nb = d.shape
        return torch.repeat_interleave(d, 8, dim=1) * self.scale_factor(sc)                  # (R, nb*8)

    # -- moves ---------------------------------------------------------------------------------
    def alt_values(self, st: State, rows) -> torch.Tensor:
        idx = st.idx[rows]                                             # (na, G)
        R_, G = idx.shape; N = G * 4
        k = torch.arange(4, device=idx.device)
        up_i = self.nb_up[idx]                                         # (na, G, 4)
        dn_i = self.nb_dn[idx]
        gk = self.grid.T                                               # (4, G_grid)
        mag_up = torch.where(up_i >= 0, gk[k, up_i.clamp(min=0)], torch.full_like(up_i, 0, dtype=torch.float32).fill_(NAN)).reshape(R_, N)
        mag_dn = torch.where(dn_i >= 0, gk[k, dn_i.clamp(min=0)], torch.full_like(dn_i, 0, dtype=torch.float32).fill_(NAN)).reshape(R_, N)
        se = st.se[rows]; sgn = st.sgn[rows]
        alts = [se * mag_up * sgn, se * mag_dn * sgn]
        if not self.has_pairs:                                         # IQ3_S: single sign flips are free moves
            alts.append(-st.vals[rows])
        return torch.stack(alts)

    def apply(self, st: State, rows, cols, a):
        g = cols // 4; k = cols % 4
        cur = st.idx[rows, g]
        new = torch.where(a == 0, self.nb_up[cur, k], torch.where(a == 1, self.nb_dn[cur, k], cur))
        st.idx[rows, g] = new
        flip = a == 2
        st.sgn[rows[flip], cols[flip]] *= -1
        st.vals[rows, cols] = st.se[rows, cols] * self.grid[st.idx[rows, g], k] * st.sgn[rows, cols]

    def pair_candidates(self, st: State, rows):
        """IQ3_XXS: signs come in 8-groups with even parity, so a flip of weight r<7 forces weight 7."""
        v = st.vals[rows]; R_, N = v.shape
        t = torch.arange(N // 8, device=v.device)
        j1 = (t[:, None] * 8 + torch.arange(7, device=v.device)[None, :]).reshape(-1)     # (M,)
        j2 = (t * 8 + 7).repeat_interleave(7)
        j1 = j1[None, :].expand(R_, -1); j2 = j2[None, :].expand(R_, -1)
        c1 = -2 * v.gather(1, j1); c2 = -2 * v.gather(1, j2)
        return j1, j2, c1, c2

    def apply_pair(self, st: State, rows, m):
        t = m // 7; r = m % 7
        j1 = t * 8 + r; j2 = t * 8 + 7
        st.sgn[rows, j1] *= -1; st.sgn[rows, j2] *= -1
        st.vals[rows, j1] *= -1; st.vals[rows, j2] *= -1

    # -- pack ----------------------------------------------------------------------------------
    def encode(self, st: State) -> np.ndarray:
        raw = st.raw.copy(); R, nbytes = raw.shape; nb = nbytes // self.type_size
        b = raw.reshape(R, nb, self.type_size)
        idx = st.idx.cpu().numpy().reshape(R, nb, 64)
        neg = (st.sgn.cpu().numpy() < 0).astype(np.uint8).reshape(R, nb, 256)
        sc = st.sc.cpu().numpy().reshape(R, nb, 8).astype(np.uint32)
        b[:, :, 0:2] = st.d.cpu().numpy().astype(np.float16).view(np.uint8).reshape(R, nb, 2)
        b[:, :, 2:66] = (idx & 0xFF).astype(np.uint8)
        if self.qtype == QT.IQ3_S:
            hi = (idx >> 8).astype(np.uint8).reshape(R, nb, 8, 8)
            b[:, :, 66:74] = sum((hi[:, :, :, i] << i) for i in range(8)).astype(np.uint8)
            n8 = neg.reshape(R, nb, 32, 8)
            b[:, :, 74:106] = sum((n8[:, :, :, i] << i) for i in range(8)).astype(np.uint8)
            b[:, :, 106:110] = (sc[:, :, 0::2] | (sc[:, :, 1::2] << 4)).astype(np.uint8)
        else:
            n8 = neg.reshape(R, nb, 8, 4, 8)                            # sub-block, group, r
            sidx = sum((n8[:, :, :, :, r].astype(np.uint32) << r) for r in range(7))       # 7 bits; parity implied
            new = (sc << 28) | sum((sidx[:, :, :, t] << (7 * t)) for t in range(4)).astype(np.uint32)
            b[:, :, 66:98] = new.astype(np.uint32).view(np.uint8).reshape(R, nb, 32)
        return raw


class IQ2GridCodec(GridCodec):
    """Native eight-dimensional IQ2 lattices with parity-constrained signs."""
    vector_size = 8

    def __init__(self, qtype, device):
        self.qtype, self.device = qtype, device
        self.block_size, self.type_size = GGML_QUANT_SIZES[qtype]
        cls = {QT.IQ2_XXS: gq.IQ2_XXS, QT.IQ2_XS: gq.IQ2_XS}[qtype]
        grid = _grid_of(cls)
        self.grid = torch.as_tensor(grid, device=device)
        self.nb_up = self.nb_dn = None
        self.has_pairs = True
        self.ksigns = np.frombuffer(gq.IQ2_XXS.ksigns, dtype=np.uint8)
        self.scale_group_size = 32 if qtype == QT.IQ2_XXS else 16

    def alt_values(self, st, rows):
        raise ValueError("IQ2 refinement requires method=block")

    def scale_factor(self, sc):
        return (0.5 + sc.to(torch.float32)) * 0.25

    def sub_scale(self, d, sc):
        return d.repeat_interleave(256 // self.scale_group_size, dim=1) * self.scale_factor(sc)

    def decode(self, raw):
        R, nbytes = raw.shape
        nb = nbytes // self.type_size
        b = raw.reshape(R, nb, self.type_size)
        d = _f16(b[:, :, :2]).reshape(R, nb)
        if self.qtype == QT.IQ2_XXS:
            words = b[:, :, 2:].copy().view(np.uint32).reshape(R, nb, 8, 2)
            idx = np.stack([(words[..., 0] >> (8*i)) & 255 for i in range(4)], axis=-1)
            sc = words[..., 1] >> 28
            signs = np.stack([(words[..., 1] >> (7*i)) & 127 for i in range(4)], axis=-1)
        else:
            words = b[:, :, 2:66].copy().view(np.uint16).reshape(R, nb, 32)
            idx = words & 511
            signs = words >> 9
            scales = b[:, :, 66:74]
            sc = np.stack([scales & 15, scales >> 4], axis=-1)
        idx = torch.as_tensor(idx.reshape(R, nb*32).astype(np.int64), device=self.device)
        sign_bytes = self.ksigns[signs.astype(np.int64)]
        negative = np.stack([(sign_bytes >> i) & 1 for i in range(8)], axis=-1).reshape(R, nb*256)
        sgn = torch.as_tensor(np.where(negative, -1., 1.).astype(np.float32), device=self.device)
        d = torch.as_tensor(d, device=self.device)
        sc = torch.as_tensor(sc.reshape(R,-1).astype(np.int64), device=self.device)
        se = self.sub_scale(d,sc).repeat_interleave(self.scale_group_size,dim=1)
        vals = se * self.grid[idx].reshape(R,nb*256) * sgn
        return State(raw,vals,idx=idx,sgn=sgn,se=se,sc=sc,d=d)

    def encode(self, st):
        raw=st.raw.copy()
        R,nbytes=raw.shape
        nb=nbytes//self.type_size
        b=raw.reshape(R,nb,self.type_size)
        b[:,:,:2]=st.d.cpu().numpy().astype(np.float16).view(np.uint8).reshape(R,nb,2)
        idx=st.idx.cpu().numpy().astype(np.uint32).reshape(R,nb,32)
        neg=(st.sgn.cpu().numpy()<0).astype(np.uint32).reshape(R,nb,32,8)
        if np.any(neg.sum(-1)%2):
            raise ValueError('IQ2 sign groups require even parity')
        signs=sum(neg[...,i] << i for i in range(7))
        sc=st.sc.cpu().numpy().astype(np.uint32)
        if self.qtype == QT.IQ2_XXS:
            groups=idx.reshape(R,nb,8,4)
            first=sum(groups[...,i] << (8*i) for i in range(4))
            signs=signs.reshape(R,nb,8,4)
            second=(sc.reshape(R,nb,8)<<28)|sum(signs[...,i]<<(7*i) for i in range(4))
            b[:,:,2:]=np.stack([first,second],axis=-1).astype(np.uint32).view(np.uint8).reshape(R,nb,64)
        else:
            b[:,:,2:66]=(idx | (signs<<9)).astype(np.uint16).view(np.uint8).reshape(R,nb,64)
            sc=sc.reshape(R,nb,16)
            b[:,:,66:74]=(sc[:,:,0::2] | (sc[:,:,1::2]<<4)).astype(np.uint8)
        return raw


SUPPORTED = {QT.IQ2_XXS, QT.IQ2_XS, QT.IQ4_NL, QT.IQ4_XS, QT.Q4_K, QT.Q5_K, QT.Q6_K, QT.Q3_K, QT.IQ3_S, QT.IQ3_XXS}


def make_codec(qtype: QT, device):
    if qtype in (QT.IQ2_XXS, QT.IQ2_XS):
        return IQ2GridCodec(qtype, device)
    if qtype in (QT.IQ3_S, QT.IQ3_XXS):
        return GridCodec(qtype, device)
    if qtype in SUPPORTED:
        return TableCodec(qtype, device)
    return None
