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
already on the device.
"""

from __future__ import annotations

import numpy as np
import torch


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
