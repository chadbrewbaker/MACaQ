"""Fused activation transforms for the native GGUF runtime. Device code is Triton."""
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice, gdc_launch_dependents, gdc_wait


@triton.jit
def _butterfly32(x, lane: tl.constexpr, groups: tl.constexpr):
    for step in tl.static_range(5):
        bit = 1 << step
        other = tl.gather(x, tl.broadcast_to((lane ^ bit)[None, :], (groups, 32)), 1)
        x = tl.where((lane[None, :] & bit) == 0, x + other, other - x)
    return x


@triton.jit
def _store_q8(y, values, block, valid, lane: tl.constexpr):
    maximum = tl.max(tl.abs(values), 1)
    scale = maximum / 127.0
    total = tl.sum(values, 1)
    # Match the runtime's fast division and round-away-from-zero Q8 conversion.
    q = libdevice.round(libdevice.fast_dividef(values, tl.where(scale > 0, scale, 1)[:, None])).to(tl.int8)
    tl.store(y + block[:, None] * 36 + 4 + lane[None, :], q, valid[:, None])
    tl.store((y + block * 36).to(tl.pointer_type(tl.float16)), scale.to(tl.float16), valid)
    tl.store((y + block * 36 + 2).to(tl.pointer_type(tl.float16)), total.to(tl.float16), valid)


@triton.jit
def macaq_pre(x, prescale, y, packed, width, rows, padded,
              H: tl.constexpr, GROUPS: tl.constexpr, Q8: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        gdc_launch_dependents()
    row = tl.program_id(1).to(tl.int64)
    groups = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    lane = tl.arange(0, 32)
    offsets = groups[:, None] * H + lane[None, :]
    active = (offsets < width) & (row < rows)
    norm: tl.constexpr = H ** -0.5
    if PDL:
        gdc_wait()
    a = tl.load(x + row * width + offsets, active, 0) * tl.load(prescale + offsets, active, 0)
    a = _butterfly32(a * norm, lane, GROUPS)
    if H == 64:
        b = tl.load(x + row * width + offsets + 32, active, 0) * tl.load(prescale + offsets + 32, active, 0)
        b = _butterfly32(b * norm, lane, GROUPS)
        lo, hi = a + b, a - b
    else:
        lo = a
    # Preserve the floating-point graph output for callbacks and other consumers.
    tl.store(y + row * width + offsets, lo, active)
    if H == 64:
        tl.store(y + row * width + offsets + 32, hi, active)
    if Q8:
        block = row * (padded // 32) + groups * (H // 32)
        # The launch covers the padded width, including zero activation blocks.
        valid = (groups * H < padded) & (row < rows)
        _store_q8(packed, lo, block, valid, lane)
        if H == 64:
            _store_q8(packed, hi, block + 1, valid, lane)


@triton.jit
def macaq_moe_pre(x, prescale, ids, y, packed, width, tokens, routes,
                  input_routes, padded, id_stride, experts,
                  GROUPS: tl.constexpr, Q8: tl.constexpr, PDL: tl.constexpr = False):
    if PDL:
        gdc_launch_dependents()
    row = tl.program_id(1).to(tl.int64)
    token, route = row // routes, row % routes
    groups = tl.program_id(0) * GROUPS + tl.arange(0, GROUPS)
    lane = tl.arange(0, 32)
    offsets = groups[:, None] * 64 + lane[None, :]
    if PDL:
        gdc_wait()
    expert = tl.load(ids + token * id_stride + route, token < tokens, 0).to(tl.int64)
    active = (offsets < width) & (token < tokens) & (expert >= 0) & (expert < experts)
    input_row = token * input_routes + route % input_routes
    a = tl.load(x + input_row * width + offsets, active, 0)
    a *= tl.load(prescale + expert * width + offsets, active, 0)
    b = tl.load(x + input_row * width + offsets + 32, active, 0)
    b *= tl.load(prescale + expert * width + offsets + 32, active, 0)
    a = _butterfly32(a * 0.125, lane, GROUPS)
    b = _butterfly32(b * 0.125, lane, GROUPS)
    lo, hi = a + b, a - b
    tl.store(y + row * width + offsets, lo, active)
    tl.store(y + row * width + offsets + 32, hi, active)
    if Q8:
        block = row * (padded // 32) + groups * 2
        valid = (groups * 64 < padded) & (token < tokens)
        _store_q8(packed, lo, block, valid, lane)
        _store_q8(packed, hi, block + 1, valid, lane)
