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


def _check(query: torch.Tensor, seqlens: np.ndarray):
    nums_dict, batch_ptr, offsets = compute_causal_conv1d_metadata(query)
    seq_ids, host_offsets = _host_tables(seqlens)
    entry = nums_dict[BLOCK_M]
    assert entry["tot"] == entry["mlist_len"] == len(seq_ids)
    got_batch = batch_ptr.detach().cpu().numpy()
    got_off = offsets.detach().cpu().numpy()
    np.testing.assert_array_equal(got_batch, seq_ids)
    np.testing.assert_array_equal(got_off, host_offsets)


def test_device_indices_match_the_old_host_tables():
    seqlens = np.array([1, 7, 8, 9, 17, 3, 0, 32], dtype=np.int64)
    cu = np.zeros(len(seqlens) + 1, dtype=np.int64)
    cu[1:] = np.cumsum(seqlens)
    _check(torch.tensor(cu, dtype=torch.int32), seqlens)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_gpu_indices_match_the_old_host_tables():
    seqlens = np.array([1, 7, 8, 9, 17, 3, 0, 32], dtype=np.int64)
    cu = np.zeros(len(seqlens) + 1, dtype=np.int64)
    cu[1:] = np.cumsum(seqlens)
    _check(torch.tensor(cu, dtype=torch.int32, device="cuda"), seqlens)
