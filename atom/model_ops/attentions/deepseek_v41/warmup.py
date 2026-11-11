# SPDX-License-Identifier: MIT
"""Prepare ragged prefill addressing with the allocated serving pool geometry."""

from types import SimpleNamespace

import torch

from .indices import build_indices, fill_step_indptrs
from .packed_rows import write_packed_window


@torch.no_grad()
def warmup_prefill_addressing(cache, topology, max_tokens, table_stride):
    """Launch index/window variants without writing a request's PAGE or STATE.

    The initial model warmup uses a small throwaway pool. Its offsets can be
    int32 where serving requires int64, and its aligned maximum token count
    misses ragged scans and smaller scan blocks. Dead token rows exercise the
    serving signatures while leaving the allocated pool untouched.
    """
    device, geometry = cache.indptr_device, cache.geometry
    widths = {min(max_tokens, 1023), min(max_tokens, 1025)}
    widths.update(1 << i for i in range(11) if (1 << i) <= max_tokens)
    # Triton specializes runtime integers by divisibility. Each scan block
    # needs both its aligned and ragged token count, even at the same BLOCK.
    widths.update((1 << i) - 1 for i in range(4, 11) if (1 << i) - 1 <= max_tokens)
    width = max(widths)
    batches = torch.full((width,), -1, dtype=torch.int32, device=device)
    positions = torch.zeros(width, dtype=torch.int64, device=device)
    cu = torch.zeros(2, dtype=torch.int32, device=device)
    slots = torch.zeros(1, dtype=torch.int32, device=device)
    tables = torch.zeros((1, table_stride), dtype=torch.int32, device=device)
    floor = torch.zeros(1, dtype=torch.int32, device=device)
    groups = {}
    for spec in topology:
        first = spec.index_group_start if spec.index_group_size > 1 else spec.layer_id
        groups.setdefault(first, spec)
    for rows in sorted(widths):
        step = SimpleNamespace(
            width=rows,
            batch_ids=batches[:rows],
            positions=positions[:rows],
            cu_seqlens_q=cu,
            slots=slots,
            block_tables=tables,
            decode=False,
            max_q_len=rows,
            swa_replay_start=floor,
            swa_cached_until=floor,
        )
        indptrs = fill_step_indptrs(step, geometry, cache.indptr_buffers)
        for first, spec in groups.items():
            layers = spec.index_group_size
            window = geometry.window(first, cache.num_pages)
            stride = (
                0
                if layers == 1
                else geometry.window(first + 1, cache.num_pages).ring_start
                - window.ring_start
            )
            topk = geometry.batch_topk(spec.ratio)
            selected = (
                torch.full((rows, topk), -1, dtype=torch.int32, device=device)
                if topk
                else None
            )
            build_indices(
                selected,
                indptrs[spec.ratio],
                step,
                geometry,
                window,
                spec.kv_owner,
                spec.ratio,
                layers,
                stride,
            )
        if geometry.window_fp8:
            dim = geometry.head_dim
            packed = torch.zeros(
                (rows, geometry.row_bytes), dtype=torch.uint8, device=device
            )
            values, scales = packed[:, :dim], packed[:, dim : dim + dim // 32]
            for spec in topology:
                window = geometry.window_bytes(spec.layer_id, cache.num_pages)
                for cached in (None, floor):
                    write_packed_window(
                        values, scales, cache.backing, step, window, dim, cached
                    )
    torch.cuda.synchronize(device)
