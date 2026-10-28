"""Program map for varlen causal conv, built on the sequence tensor's device."""

import numpy as np
import pytest
import torch

from atom.model_ops.attentions.gdn_attn import compute_causal_conv1d_metadata
from atom.model_ops.mamba_ops.causal_conv1d import BLOCK_M


def _host_tables(seqlens: np.ndarray):
    nums = -(-seqlens.astype(np.int64) // BLOCK_M)
    seq_ids = np.repeat(np.arange(len(nums), dtype=np.int64), nums)
    offsets = np.empty(int(nums.sum()), dtype=np.int64)
    cursor = 0
    for count in nums:
        count = int(count)
        offsets[cursor : cursor + count] = np.arange(count)
        cursor += count
    return seq_ids, offsets


def test_device_indices_match_the_old_host_tables():
    seqlens = np.array([1, 7, 8, 9, 17, 3, 0, 32], dtype=np.int64)
    cu = np.zeros(len(seqlens) + 1, dtype=np.int64)
    cu[1:] = np.cumsum(seqlens)
    query = torch.tensor(cu, dtype=torch.int32)
    batch_ptr, offsets, num_chunks = compute_causal_conv1d_metadata(query)
    seq_ids, host_offsets = _host_tables(seqlens)
    assert num_chunks == len(seq_ids)
    np.testing.assert_array_equal(batch_ptr.numpy(), seq_ids)
    np.testing.assert_array_equal(offsets.numpy(), host_offsets)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_indices_match_the_old_host_tables():
    seqlens = np.array([1, 7, 8, 9, 17, 3, 0, 32], dtype=np.int64)
    cu = np.zeros(len(seqlens) + 1, dtype=np.int64)
    cu[1:] = np.cumsum(seqlens)
    query = torch.tensor(cu, dtype=torch.int32, device="cuda")
    batch_ptr, offsets, num_chunks = compute_causal_conv1d_metadata(query)
    seq_ids, host_offsets = _host_tables(seqlens)
    assert num_chunks == len(seq_ids)
    np.testing.assert_array_equal(batch_ptr.cpu().numpy(), seq_ids)
    np.testing.assert_array_equal(offsets.cpu().numpy(), host_offsets)
