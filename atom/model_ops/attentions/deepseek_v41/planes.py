# SPDX-License-Identifier: MIT
"""Views and window I/O for flat [all data | all scales] KV sources."""

import torch
import triton
import triton.language as tl

from .packed_rows import _read_plane


def split_source(source, dim, fp4=False):
    """Zero-copy views of one flat [all data | all scales] source tensor."""
    data_width = dim // 2 if fp4 else dim
    scale_width = dim // (16 if fp4 else 32)
    if source.dtype != torch.uint8 or source.ndim != 1 or not source.is_contiguous():
        raise ValueError("A KV source must be one contiguous flat uint8 tensor")
    rows, remainder = divmod(source.numel(), data_width + scale_width)
    if remainder:
        raise ValueError("KV source allocation must contain complete data/scale rows")
    cut = rows * data_width
    return source[:cut].view(rows, data_width), source[cut:].view(rows, scale_width)


@triton.jit
def _stage_window(
    data, scale, output, output_scale, slots, cu, RING: tl.constexpr, D: tl.constexpr
):
    batch = tl.program_id(1)
    # Graph padding has no slot to read. All live requests use batch-local
    # rows, so the temporary source never scales with cache.num_slots.
    if tl.load(cu + batch + 1) == tl.load(cu + batch):
        return
    slot = tl.load(slots + batch).to(tl.int64)
    if slot < 0:  # a dummy query prefix may be live without a cache request
        return
    row = tl.program_id(0) * 16 + tl.arange(0, 16)
    d = tl.arange(0, D)
    g = tl.arange(0, D // 32)
    old = slot * RING + row
    new = batch.to(tl.int64) * RING + row
    tl.store(
        output + new[:, None] * D + d[None, :],
        tl.load(data + old[:, None] * D + d[None, :], (row < RING)[:, None], other=0),
        (row < RING)[:, None],
    )
    tl.store(
        output_scale + new[:, None] * (D // 32) + g[None, :],
        tl.load(
            scale + old[:, None] * (D // 32) + g[None, :],
            (row < RING)[:, None],
            other=0,
        ),
        (row < RING)[:, None],
    )


def prefill_window(history, step, ring):
    """One SWA source: this batch's rings followed by space for current KV.

    The producer writes current KV directly to the tail views. The persistent
    rings remain unchanged until attention completes and write_window commits.
    Reuse the step's reserved storage, but derive plane and tail offsets from
    its live request count. Token graphs keep this beside the live CSR build.
    """
    dim = history[0].shape[-1]
    batches = step.cu_seqlens_q.numel() - 1
    rows = batches * ring
    size = (rows + step.running_tokens) * (dim + dim // 32)
    source = step.prefill_window_staging
    if source is None:
        raise ValueError("Prefill window staging must be reserved before preparation")
    if source.numel() < size:
        raise ValueError("Prefill window staging is smaller than the live step")
    planes = split_source(source[:size], dim)
    if batches:
        if source.is_cuda:
            _stage_window[(triton.cdiv(ring, 16), batches)](
                *history, *planes, step.slots, step.cu_seqlens_q, ring, dim
            )
        else:
            for batch in range(batches):
                if step.cu_seqlens_q[batch + 1] == step.cu_seqlens_q[batch]:
                    continue
                slot = int(step.slots[batch])
                if slot < 0:
                    continue
                for dst, old in zip(planes, history):
                    dst[batch * ring : (batch + 1) * ring].copy_(
                        old[slot * ring : (slot + 1) * ring]
                    )
    return planes, tuple(plane[rows:] for plane in planes)


@triton.jit
def _reset_windows(
    ptrs, slots, NSLOTS: tl.constexpr, RING: tl.constexpr, D: tl.constexpr
):
    pair = tl.program_id(0)
    layer, request = pair // NSLOTS, pair % NSLOTS
    slot = tl.load(slots + request).to(tl.int64)
    data = tl.load(ptrs + layer * 2).to(tl.pointer_type(tl.uint8))
    scale = tl.load(ptrs + layer * 2 + 1).to(tl.pointer_type(tl.uint8))
    offset = tl.program_id(1) * 1024 + tl.arange(0, 1024)
    tl.store(data + slot * RING * D + offset, 0, offset < RING * D)
    tl.store(scale + slot * RING * (D // 32) + offset, 0, offset < RING * (D // 32))


def reset_windows(sources, ptrs, slots, ring, dim):
    """Reset selected slots across every layer's data and scale planes."""
    if not slots.numel() or not sources:
        return
    if slots.is_cuda:
        # The layer/slot product can exceed the y-axis limit of 65535.
        _reset_windows[(len(sources) * slots.numel(), triton.cdiv(ring * dim, 1024))](
            ptrs, slots, slots.numel(), ring, dim
        )
    else:
        for source in sources:
            for plane in split_source(source, dim):
                plane.view(plane.shape[0] // ring, -1).index_fill_(0, slots, 0)


@triton.jit
def _gather_windows(
    ptrs,
    layers,
    slots,
    output,
    NSLOTS: tl.constexpr,
    RING: tl.constexpr,
    D: tl.constexpr,
):
    pair = tl.program_id(1)
    layer = tl.load(layers + pair // NSLOTS)
    slot = tl.load(slots + pair % NSLOTS).to(tl.int64)
    values = tl.load(ptrs + layer * 2).to(tl.pointer_type(tl.uint8))
    scales = tl.load(ptrs + layer * 2 + 1).to(tl.pointer_type(tl.uint8))
    i = tl.program_id(0) * 16 + tl.arange(0, 16)
    dims = tl.arange(0, D)
    result = _read_plane(values, scales, slot * RING + i, dims, i < RING, D, False)
    tl.store(
        output + (pair.to(tl.int64) * RING + i)[:, None] * D + dims[None, :],
        result,
        (i < RING)[:, None],
    )


def gather_windows(sources, ptrs, layer_ids, layers, slots, ring, dim):
    single = isinstance(layers, int)
    layers = layer_ids[layers : layers + 1] if single else layers.flatten()
    output = torch.empty(
        (layers.numel(), slots.numel(), ring, dim),
        dtype=torch.bfloat16,
        device=slots.device,
    )
    if not output.numel():
        return output[0] if single else output
    if output.is_cuda:
        _gather_windows[(triton.cdiv(ring, 16), layers.numel() * slots.numel())](
            ptrs,
            layers,
            slots,
            output,
            slots.numel(),
            ring,
            dim,
        )
    else:
        from atom.model_ops.blockscale_kernels.quantization import FP8_DTYPE

        rows = slots.long()[:, None] * ring + torch.arange(ring)
        for at, layer in enumerate(layers.tolist()):
            data, scale = split_source(sources[layer], dim)
            raw, codes = data[rows], scale[rows]
            steps = torch.exp2(codes.float() - 127)
            steps[codes == 255] = float("nan")
            output[at].copy_(
                raw.view(FP8_DTYPE).float() * steps.repeat_interleave(32, -1)
            )
    return output[0] if single else output


@triton.jit
def _write_planes(
    values,
    scales,
    vs,
    ss,
    data,
    scale,
    positions,
    batches,
    cu,
    slots,
    cached,
    CACHED: tl.constexpr,
    RING: tl.constexpr,
    D: tl.constexpr,
):
    t = tl.program_id(0)
    batch = tl.load(batches + t)
    if batch < 0:
        return
    if t < tl.load(cu + batch + 1) - RING:
        return
    pos = tl.load(positions + t)
    if CACHED:  # noqa: SIM102
        if pos < tl.load(cached + batch):
            return
    row = tl.load(slots + batch).to(tl.int64) * RING + pos % RING
    d = tl.arange(0, D)
    g = tl.arange(0, D // 32)
    tl.store(data + row * D + d, tl.load(values + t * vs + d))
    tl.store(scale + row * (D // 32) + g, tl.load(scales + t * ss + g))


def write_window_planes(values, scales, planes, step, window, dim, cached_until=None):
    data, scale = planes
    if not data.is_cuda:
        for t in range(step.running_tokens):
            batch = int(step.batch_ids[t])
            if batch < 0 or t < int(step.cu_seqlens_q[batch + 1]) - window.ring_slots:
                continue
            pos = int(step.positions[t])
            if cached_until is not None and pos < int(cached_until[batch]):
                continue
            row = window.index(int(step.slots[batch]), pos)
            data[row].copy_(values.flatten(0, -2)[t].view(torch.uint8))
            scale[row].copy_(scales.flatten(0, -2)[t].view(torch.uint8))
        return
    if step.running_tokens:
        _write_planes[(step.running_tokens,)](
            values.view(torch.uint8),
            scales.view(torch.uint8),
            values.stride(-2),
            scales.stride(-2),
            data,
            scale,
            step.positions,
            step.batch_ids,
            step.cu_seqlens_q,
            step.slots,
            cached_until,
            cached_until is not None,
            window.ring_slots,
            dim,
        )


@triton.jit
def _gather_window_planes(
    ptrs, layers, slots, data, scale, B: tl.constexpr, R: tl.constexpr, D: tl.constexpr
):
    pair = tl.program_id(1)
    layer = tl.load(layers + pair // B)
    slot = tl.load(slots + pair % B).to(tl.int64)
    src = tl.load(ptrs + layer * 2).to(tl.pointer_type(tl.uint8))
    ss = tl.load(ptrs + layer * 2 + 1).to(tl.pointer_type(tl.uint8))
    i = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    tl.store(
        data + pair.to(tl.int64) * R * D + i,
        tl.load(src + slot * R * D + i, i < R * D, other=0),
        i < R * D,
    )
    tl.store(
        scale + pair.to(tl.int64) * R * (D // 32) + i,
        tl.load(ss + slot * R * (D // 32) + i, i < R * (D // 32), other=0),
        i < R * (D // 32),
    )


def gather_window_planes(ptrs, layers, slots, ring, dim):
    """Raw MXFP8 draft windows; slot IDs stay live during graph replay."""
    shape = (layers.numel(), slots.numel(), ring)
    source = torch.empty(
        layers.numel() * slots.numel() * ring * (dim + dim // 32),
        dtype=torch.uint8,
        device=slots.device,
    )
    planes = split_source(source, dim)
    if source.numel():
        _gather_window_planes[(triton.cdiv(ring * dim, 1024), shape[0] * shape[1])](
            ptrs, layers, slots, *planes, shape[1], ring, dim
        )
    return tuple(p.view(*shape, p.shape[-1]) for p in planes)
