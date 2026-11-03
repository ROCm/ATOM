# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The map between a token axis and the sequence axis.

Every other per-token array in this package is indexed by token; almost every
quantity the scheduler hands over is indexed by sequence. This converts between
them, so a kernel can resolve `per_seq[batch_id[t]]`.

There are TWO axes, named after the `cu_seqlens_q` / `cu_seqlens_k` each follows:
query lengths in gives a `batch_id_per_q_token`, context lengths a
`batch_id_per_k_token`. They are not interchangeable, and mixing them does not
fault -- it silently resolves a token to another request's row.

Which builder depends on where the result must live. `build_batch_ids` (host)
covers any width bounded by `max_num_batched_tokens`, which a pinned
`CpuGpuBuffer` can hold; every q-axis caller is one. `build_batch_ids_device`
covers the `total_kv` width, which nothing caps -- no fixed buffer fits it, a
pageable upload that size would sit on the critical path, and its input is
already on the device. `build_batch_ids_from_cu` is that same device result
built from the step's uploaded `cu_seqlens` in one launch
(`ATOM_FAST_PREFILL_BATCH_IDS`), where `repeat_interleave` walks each
sequence's whole length on one wavefront.
"""

from __future__ import annotations

import numpy as np
import torch

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:
    triton = None
    tl = None

# Tokens one program of `build_batch_ids_from_cu` writes (16 KiB of int32).
_IDS_CHUNK = 4096


def build_batch_ids(
    seqlens: np.ndarray,
    pad_to: int | None = None,
    pad: int = -1,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Which sequence each token of a flat token axis belongs to.

    Sequence `i` contributes `seqlens[i]` consecutive entries of value `i`, in
    sequence order -- the layout attention reads through the matching
    `cu_seqlens_*`.

    `pad_to` widens the result to a CUDAGraph bucket, filling the tail with
    `pad`. The tail is NOT sequence 0: a captured step runs at the bucket width
    whatever the batch, so a fabricated token naming a real sequence would have
    every consumer resolve it to that request's row and quietly attend on its
    behalf. `-1` is what the kernels bail on.
    """
    total = int(seqlens.sum())
    width = total if pad_to is None else pad_to
    # Asserted, not raised: numpy refuses the write below on its own, so this
    # only trades its message for a better one. The dtype guard in `.prefill`
    # is a `raise` because there numpy would answer, wrongly and in silence.
    assert width >= total, f"pad_to={pad_to} is under the {total} tokens scheduled"
    out = np.empty(width, dtype=np.int32) if out is None else out[:width]
    out[:total] = np.repeat(np.arange(len(seqlens), dtype=np.int32), seqlens)
    out[total:] = pad
    return out


def build_batch_ids_device(seqlens: torch.Tensor, *, total: int) -> torch.Tensor:
    """`build_batch_ids` for a result too wide to stage through a host buffer.

    No `pad_to`: the gather it feeds is exactly this long, so there is no
    CUDAGraph bucket to pad out to and no sentinel to write.

    `total` is `seqlens.sum()`, and required rather than optional: without it
    `repeat_interleave` synchronizes the device to read its own output length
    back, once per forward between the metadata and the first layer. Every
    caller already holds the sum on the host -- omitting it would only be
    slower, which is the kind of thing a default hides.
    """
    return torch.repeat_interleave(
        torch.arange(seqlens.numel(), dtype=torch.int32, device=seqlens.device),
        seqlens.to(torch.int64),
        output_size=total,
    )


if triton is not None:

    @triton.jit
    def _batch_ids_from_cu_kernel(cu_seqlens, out, CHUNK: tl.constexpr):
        # Program (sequence i, chunk c) writes `i` over its share of
        # [cu[i], cu[i + 1]). The grid covers the longest sequence, so the
        # programs past a shorter one's end exit without a store.
        i = tl.program_id(0)
        lo = tl.load(cu_seqlens + i).to(tl.int64)
        hi = tl.load(cu_seqlens + i + 1).to(tl.int64)
        start = lo + tl.program_id(1).to(tl.int64) * CHUNK
        if start < hi:
            offs = start + tl.arange(0, CHUNK)
            tl.store(out + offs, tl.full([CHUNK], i, tl.int32), mask=offs < hi)

else:
    _batch_ids_from_cu_kernel = None


def build_batch_ids_from_cu(
    cu_seqlens: torch.Tensor, *, total: int, max_seqlen: int
) -> torch.Tensor:
    """`build_batch_ids_device` from the prefix sum the step already uploaded.

    The same int32 result -- `cu_seqlens[i + 1] - cu_seqlens[i]` entries of `i`,
    in sequence order, `total` wide -- in one launch. `repeat_interleave`'s
    kernel gives each sequence one wavefront that walks its whole length, so a
    prefill over a few long cached contexts (32 sequences, 2.6M tokens) runs
    ~270 us on 3 workgroups; this one spreads every sequence over
    `max_seqlen / 4096` programs.

    `cu_seqlens` is `num_seqs + 1` wide, non-decreasing and starts at 0, with
    `cu_seqlens[-1] == total`; a zero-length (padding) row contributes nothing.
    `max_seqlen` bounds every `cu_seqlens[i + 1] - cu_seqlens[i]` and only sizes
    the grid. Both host ints, so there is no device read; the grid depends on
    them alone, which keeps the launch graph-capturable.
    """
    out = torch.empty(total, dtype=torch.int32, device=cu_seqlens.device)
    num_seqs = cu_seqlens.numel() - 1
    if total == 0 or num_seqs <= 0 or max_seqlen <= 0:
        return out
    if _batch_ids_from_cu_kernel is None:
        raise RuntimeError("build_batch_ids_from_cu requires Triton")
    _batch_ids_from_cu_kernel[(num_seqs, triton.cdiv(max_seqlen, _IDS_CHUNK))](
        cu_seqlens, out, CHUNK=_IDS_CHUNK, num_warps=4
    )
    return out
