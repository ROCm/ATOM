# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501
import functools
import importlib
import logging

import numpy as np
import torch

import triton

from .utils import tensor_cache

logger = logging.getLogger("atom")

# Host copy of a device `cu_seqlens`, set by the metadata builder on the one
# tensor object every layer of a forward receives. Per object, not per buffer:
# the next forward slices a new view, which carries none until set again.
_HOST_ATTR = "_atom_host_cu_seqlens"
_CHUNK_ATTR = "_atom_chunk_indices"


def attach_host_cu_seqlens(cu_seqlens: torch.Tensor, host: np.ndarray) -> None:
    """Let `prepare_chunk_indices` build from `host`, which must hold the
    values of `cu_seqlens`, rather than read them back from the device. That
    read waits for everything queued on the stream ahead of it."""
    setattr(cu_seqlens, _HOST_ATTR, np.asarray(host))
    setattr(cu_seqlens, _CHUNK_ATTR, {})


def _host_chunk_indices(cu_seqlens: torch.Tensor, chunk_size: int):
    host = getattr(cu_seqlens, _HOST_ATTR, None)
    if host is None:
        return None
    built = getattr(cu_seqlens, _CHUNK_ATTR)
    out = built.get(chunk_size)
    if out is None:
        from atom.utils import upload_numpy

        n = -(-np.diff(host.astype(np.int64)) // chunk_size)
        indices = np.arange(int(n.sum()), dtype=np.int64) - np.repeat(
            np.cumsum(n) - n, n
        )
        # Same pairs as the device path, including its sequence numbering,
        # which counts only sequences that have a chunk.
        seq = np.cumsum(indices == 0) - 1
        dtype = torch.empty(0, dtype=cu_seqlens.dtype).numpy().dtype
        pairs = np.ascontiguousarray(np.stack([seq, indices], 1).astype(dtype))
        out = built[chunk_size] = upload_numpy(pairs, cu_seqlens.device)
    return out


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return cu_seqlens[1:] - cu_seqlens[:-1]


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    host_built = _host_chunk_indices(cu_seqlens, chunk_size)
    if host_built is not None:
        return host_built
    indices = torch.cat(
        [
            torch.arange(n)
            for n in triton.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor, chunk_size: int
) -> torch.LongTensor:
    return torch.cat(
        [cu_seqlens.new_tensor([0]), triton.cdiv(prepare_lens(cu_seqlens), chunk_size)]
    ).cumsum(-1)


def _host_aware(device_path):
    @functools.wraps(device_path)
    def wrapper(cu_seqlens, chunk_size):
        host_built = _host_chunk_indices(cu_seqlens, chunk_size)
        if host_built is not None:
            return host_built
        return device_path(cu_seqlens, chunk_size)

    return wrapper


def _host_seq_bounds(device_path):
    @functools.wraps(device_path)
    def wrapper(cu_seqlens):
        host = getattr(cu_seqlens, _HOST_ATTR, None)
        if host is None:
            return device_path(cu_seqlens)
        b = [int(x) for x in host]
        return tuple((b[i], b[i + 1]) for i in range(len(b) - 1))

    return wrapper


def _uploading_segments(build):
    """aiter builds FlashKDA's segment descriptors straight onto the device,
    an upload that waits for the stream to drain. Build them on the host and
    upload without waiting; cached on the same key as aiter's builder."""

    @functools.lru_cache(maxsize=64)
    @functools.wraps(build)
    def wrapper(seqs, C, chunks_per_seg, device):
        if torch.device(device).type == "cpu":
            return build(seqs, C, chunks_per_seg, device)
        from atom.utils import upload_numpy

        desc, off, num_segs, max_per_seq = build(
            seqs, C, chunks_per_seg, torch.device("cpu")
        )
        return (
            upload_numpy(desc.numpy(), device),
            upload_numpy(off.numpy(), device),
            num_segs,
            max_per_seq,
        )

    return wrapper


# aiter's KDA chunk kernels import their own copies of these host readers.
_AITER_KDA = "aiter.ops.triton._triton_kernels.kimi_delta_attn"
_AITER_KDA_PATCHES = (
    *(
        (module, "prepare_chunk_indices", _host_aware)
        for module in (
            "chunk_fwd",
            "intra_attn",
            "wy_fast",
            "gla_output",
            "utils.cumsum",
            "flash_kda",
        )
    ),
    ("flash_kda", "_seq_bounds", _host_seq_bounds),
    ("flash_kda", "_build_segments", _uploading_segments),
)


def install_aiter_kda_host_chunk_indices() -> int:
    """Route aiter's KDA host reads of `cu_seqlens` through
    `attach_host_cu_seqlens` data. Idempotent; returns how many module
    bindings were replaced."""
    replaced = 0
    for module, attr, wrap in _AITER_KDA_PATCHES:
        try:
            mod = importlib.import_module(f"{_AITER_KDA}.{module}")
        except ImportError:
            continue
        fn = getattr(mod, attr, None)
        if fn is None or getattr(fn, "_atom_host_aware", False):
            continue
        wrapped = wrap(fn)
        wrapped._atom_host_aware = True
        setattr(mod, attr, wrapped)
        replaced += 1
    if replaced:
        logger.info(
            "KDA chunk indices: %d aiter bindings build from host cu_seqlens", replaced
        )
    return replaced
