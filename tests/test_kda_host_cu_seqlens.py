# SPDX-License-Identifier: MIT
"""Chunk metadata built from a host copy of `cu_seqlens` matches the device path.

The device path reads `cu_seqlens` back to the host, which waits for every
kernel queued ahead of it; under pipeline parallelism that is the previous
microbatch's whole forward. These run on CPU tensors and pin the host-built
values to what the device path produced.
"""

import numpy as np
import pytest
import torch

index = pytest.importorskip("atom.model_ops.fla_ops.index")

LENS = [
    [5],
    [64],
    [65],
    [0, 7],
    [3, 0, 0, 129],
    [8191, 1, 30, 2304, 13],
    *[
        list(np.random.default_rng(seed).integers(0, 5000, size=1 + seed % 17))
        for seed in range(40)
    ],
]


def device_chunk_indices(cu_seqlens, chunk_size):
    """The device path, kept as the oracle."""
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    n = ((lens + chunk_size - 1) // chunk_size).tolist()
    indices = torch.cat([torch.arange(x) for x in n])
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


def host_and_device(lens, dtype):
    host = np.concatenate([[0], np.cumsum(lens)]).astype(np.int32)
    return host, torch.from_numpy(host.astype(np.int64)).to(dtype)


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("chunk_size", [16, 32, 64])
@pytest.mark.parametrize("lens", LENS)
def test_host_chunk_indices_match_device(lens, dtype, chunk_size):
    host, cu = host_and_device(lens, dtype)
    ref = device_chunk_indices(cu, chunk_size)
    index.attach_host_cu_seqlens(cu, host)
    got = index._host_chunk_indices(cu, chunk_size)
    assert got.dtype == ref.dtype and torch.equal(got, ref)
    assert index._host_chunk_indices(cu, chunk_size) is got
    wrapped = index._host_aware(device_chunk_indices)
    assert torch.equal(wrapped(cu, chunk_size), ref)


def test_unattached_tensor_takes_device_path():
    _, cu = host_and_device([3, 70], torch.int32)
    assert index._host_chunk_indices(cu, 64) is None
    assert index._host_chunk_indices(cu[:2], 64) is None


@pytest.mark.parametrize("lens", LENS[:12])
def test_host_seq_bounds_match_device(lens):
    def device_bounds(cu):
        b = cu.tolist()
        return tuple((b[i], b[i + 1]) for i in range(len(b) - 1))

    host, cu = host_and_device(lens, torch.int32)
    wrapped = index._host_seq_bounds(device_bounds)
    assert wrapped(cu) == device_bounds(cu)
    index.attach_host_cu_seqlens(cu, host)
    assert wrapped(cu) == device_bounds(cu)


def test_segments_are_built_on_host_and_cached():
    built_on = []

    def build(seqs, C, chunks_per_seg, device):
        built_on.append(torch.device(device).type)
        nch = [-(-(e - b) // C) for b, e in seqs]
        desc = torch.tensor(
            [nch, [b for b, _ in seqs]], dtype=torch.int32, device=device
        )
        off = torch.tensor([0, *np.cumsum(nch)], dtype=torch.int32, device=device)
        return desc, off, len(seqs), max(nch)

    seqs = ((0, 100), (100, 1000), (1000, 1001))
    wrapped = index._uploading_segments(build)
    got = wrapped(seqs, 32, 8, torch.device("meta"))
    ref = build(seqs, 32, 8, "cpu")
    assert built_on == ["cpu", "cpu"]
    assert got[0].device.type == "meta" and got[0].shape == ref[0].shape
    assert got[1].shape == ref[1].shape and got[2:] == ref[2:]
    assert wrapped(seqs, 32, 8, torch.device("meta")) is got
    assert built_on == ["cpu", "cpu"]
