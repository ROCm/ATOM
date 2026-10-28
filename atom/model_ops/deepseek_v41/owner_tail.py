# SPDX-License-Identifier: MIT
"""Gather only the replay frontier, preserving BF16 and FP32 state bits."""

import torch
import triton
import triton.language as tl
from aiter.dist.parallel_state import get_tp_group

from atom.model_ops.deepseek_v41.owner_tail_plan import plan_tail_ownership
from atom.model_ops.deepseek_v41.row_sharded_mhc import v41_mhc_gather_rows
from atom.utils import upload_numpy


@triton.jit
def _pack_tail(
    rows,
    sources,
    packet,
    WIDTHS: tl.constexpr,
    ROW_WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slot = tl.program_id(0)
    row = tl.load(rows + slot).to(tl.int64)
    word = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    base = 0
    for i in tl.static_range(len(WIDTHS)):
        width = WIDTHS[i]
        mask = (word >= base) & (word < base + width)
        bits = tl.load(
            sources[i] + row * width + word - base, mask=mask & (row >= 0), other=0
        )
        tl.store(packet + slot * ROW_WORDS + word, bits, mask=mask)
        base += width


@triton.jit
def _unpack_tail(
    rows,
    packet,
    outputs,
    WIDTHS: tl.constexpr,
    ROW_WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    out = tl.program_id(0)
    row = tl.load(rows + out).to(tl.int64)
    word = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    bits = tl.load(packet + row * ROW_WORDS + word, mask=word < ROW_WORDS, other=0)
    base = 0
    for i in tl.static_range(len(WIDTHS)):
        width = WIDTHS[i]
        mask = (word >= base) & (word < base + width)
        tl.store(outputs[i] + out * width + word - base, bits, mask=mask)
        base += width


def _words(value):
    return value.view(value.shape[1], -1).view(torch.int32)


def gather_owner_tail(host_rows, *tensors, original_rows, transport="packet", out=None):
    """One frontier map and lossless pack/gather/unpack for all seam fields.

    Inputs are contiguous [1, owner_capacity, ...] tensors, including the
    optional FP8 input and byte-view scales from fused MHC quantization. Packet
    transport needs one byte collective; fields is an explicit control using
    the same geometry and bitwise pack/unpack with one collective per field.
    """
    if not tensors or transport not in ("packet", "fields"):
        raise ValueError("Invalid owner-tail fields or transport")
    group = get_tp_group()
    plan = plan_tail_ownership(host_rows, original_rows, group.world_size)
    if any(
        t.shape[:2] != (1, plan.owner_capacity)
        or not t.is_contiguous()
        or t.dtype
        not in (torch.bfloat16, torch.float32, torch.float8_e4m3fn, torch.uint8)
        or (t[0, 0].numel() * t.element_size()) % 4
        for t in tensors
    ):
        raise ValueError("Owner-tail input lost its contiguous whole-word row layout")
    device = tensors[0].device
    outputs = (
        tuple(t.new_empty(1, len(host_rows), *t.shape[2:]) for t in tensors)
        if out is None
        else tuple(out)
    )
    if len(outputs) != len(tensors) or any(
        dst.shape != (1, len(host_rows), *src.shape[2:])
        or dst.dtype != src.dtype
        or dst.device != src.device
        or not dst.is_contiguous()
        for src, dst in zip(tensors, outputs)
    ):
        raise ValueError("Owner-tail destination differs from the native graph input")
    if not len(host_rows):
        return outputs
    local = upload_numpy(plan.local_rows[group.rank_in_group], device)
    gathered = upload_numpy(plan.gathered_rows, device)
    sources = tuple(_words(t) for t in tensors)
    targets = tuple(_words(t) for t in outputs)
    widths = tuple(s.shape[1] for s in sources)
    groups = (
        (tuple(range(len(tensors))),)
        if transport == "packet"
        else tuple((i,) for i in range(len(tensors)))
    )
    for fields in groups:
        field_widths = tuple(widths[i] for i in fields)
        row_words = sum(field_widths)
        packet = torch.empty(
            (plan.packet_rows, row_words * 4), dtype=torch.uint8, device=device
        )
        _pack_tail[(plan.packet_rows, triton.cdiv(row_words, 1024))](
            local,
            tuple(sources[i] for i in fields),
            packet.view(torch.int32),
            WIDTHS=field_widths,
            ROW_WORDS=row_words,
            BLOCK=1024,
        )
        full = v41_mhc_gather_rows(packet)
        _unpack_tail[(len(host_rows), triton.cdiv(row_words, 1024))](
            gathered,
            full.view(torch.int32),
            tuple(targets[i] for i in fields),
            WIDTHS=field_widths,
            ROW_WORDS=row_words,
            BLOCK=1024,
        )
    return outputs
