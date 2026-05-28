# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import torch

import triton

from .utils import tensor_cache


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_rebased_cu_seqlens(
    cu_seqlens: torch.LongTensor,
    num_decodes: int,
    num_decode_tokens: int,
) -> torch.LongTensor:
    """Cached helper that produces the per-chunk-fwd-call kernel-facing
    `cu_seqlens` when the GDN dispatcher feeds prefill the original
    non-spec `cu_seqlens` alongside `num_decodes`/`num_decode_tokens`.

    Caller passes the metadata's original (cache-stable) `cu_seqlens`;
    this returns `cu_seqlens[num_decodes:] - num_decode_tokens`, which the
    chunk kernels use to index into the prefill-slice of q/k/v/etc. The
    `@tensor_cache` keys off the tensor identity + the two ints, so as
    long as the metadata reuses its `cu_seqlens` tensor across forward
    calls (the normal scheduler invariant), the subtraction kernel only
    fires once per (cu_seqlens_id, num_decodes, num_decode_tokens) tuple.

    When `num_decodes == 0` and `num_decode_tokens == 0` (pure-prefill
    batch) this returns the input tensor unchanged.
    """
    if num_decodes == 0 and num_decode_tokens == 0:
        return cu_seqlens
    return cu_seqlens[num_decodes:] - num_decode_tokens


def _prefill_lens(
    cu_seqlens: torch.LongTensor,
    num_decodes: int,
) -> torch.LongTensor:
    """Per-sequence token counts for the prefill slice, computed directly
    from the original (cache-stable) `cu_seqlens` without materialising an
    intermediate sliced view. Equivalent to
    `prepare_lens(cu_seqlens[num_decodes:])` but skips the call to
    `prepare_lens` on a sliced tensor (which would miss its own cache).
    """
    if num_decodes == 0:
        return prepare_lens(cu_seqlens)
    return cu_seqlens[num_decodes + 1 :] - cu_seqlens[num_decodes:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
) -> torch.LongTensor:
    """Prologue helper for chunked varlen kernels. When the caller is the
    GDN mixed-batch dispatcher, `num_decodes`/`num_decode_tokens` indicate
    that the prefill slice starts at `cu_seqlens[num_decodes]` rather than
    `cu_seqlens[0]`; we account for that internally so the cache key stays
    `(metadata_cu_seqlens_id, chunk_size, num_decodes, num_decode_tokens)`
    — stable across forward calls when the scheduler reuses the metadata.
    `num_decode_tokens` does not affect the output (the chunk indices are
    rebase-invariant) but is part of the cache key for symmetry with
    `prepare_chunk_offsets` / `prepare_rebased_cu_seqlens`.
    """
    _ = num_decode_tokens  # in cache key only
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(
                _prefill_lens(cu_seqlens, num_decodes), chunk_size
            ).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
    num_decodes: int = 0,
    num_decode_tokens: int = 0,
) -> torch.LongTensor:
    """See `prepare_chunk_indices` for the slicing semantics."""
    _ = num_decode_tokens  # in cache key only
    return torch.cat(
        [
            cu_seqlens.new_tensor([0]),
            triton.cdiv(_prefill_lens(cu_seqlens, num_decodes), chunk_size),
        ]
    ).cumsum(-1)
