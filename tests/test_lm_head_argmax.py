# SPDX-License-Identifier: MIT

import pytest
import torch

pytest.importorskip("aiter")

from atom.model_ops.lm_head_argmax import (
    lm_head_argmax_pack,
    lm_head_argmax_pick,
    lm_head_argmax_units,
)

if not torch.cuda.is_available():
    pytest.skip("Triton kernels need a GPU", allow_module_level=True)


def _reference(logits: torch.Tensor, vocab_start_idx: int) -> torch.Tensor:
    max_val, local_idx = logits.max(dim=-1)
    return torch.stack([max_val.float(), (local_idx + vocab_start_idx).float()], dim=-1)


@pytest.mark.parametrize("rows", [1, 7, 15])
def test_small_batches_match_torch(rows):
    """Batches below the old cutoff execute the fused Triton path."""
    torch.manual_seed(rows)
    logits = torch.randn((rows, 8192), dtype=torch.bfloat16, device="cuda")
    vocab_start_idx = 32000

    actual = lm_head_argmax_pack(logits, vocab_start_idx)
    expected = _reference(logits, vocab_start_idx)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_ties_choose_lowest_global_index():
    logits = torch.tensor(
        [[1.0, 3.0, 3.0, 2.0], [5.0, 5.0, 4.0, 5.0]],
        dtype=torch.float32,
        device="cuda",
    )
    vocab_start_idx = 128

    actual = lm_head_argmax_pack(logits, vocab_start_idx)
    expected = _reference(logits, vocab_start_idx)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_vocab_larger_than_one_block_is_fused():
    block_size = 131072
    logits = torch.zeros((2, block_size + 17), dtype=torch.float32, device="cuda")
    logits[0, 4] = 3.0
    logits[0, block_size + 5] = 4.0
    logits[1, 9] = 5.0
    logits[1, block_size + 1] = 5.0
    vocab_start_idx = 32000

    actual = lm_head_argmax_pack(logits, vocab_start_idx)
    expected = _reference(logits, vocab_start_idx)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def _ar_gather_argmax(logits: torch.Tensor, world: int) -> torch.Tensor:
    """ATOM_ARGMAX_HEAD_AR_GATHER on one device: each rank's byte image of its
    vocab shard, summed rank by rank in bf16 as the all-reduce does, then the
    pick."""
    shard = logits.shape[1] // world
    reduced = None
    for rank in range(world):
        units = lm_head_argmax_units(
            logits[:, rank * shard : (rank + 1) * shard], rank * shard, rank, world
        )
        reduced = units if reduced is None else reduced + units
    out = torch.full((logits.shape[0],), -1, dtype=torch.int32, device="cuda")
    return lm_head_argmax_pick(reduced, world, out)


@pytest.mark.parametrize("rows", [1, 4, 25, 80])
@pytest.mark.parametrize("world", [2, 4])
def test_ar_gather_picks_the_full_vocab_argmax(rows, world):
    torch.manual_seed(rows * world)
    logits = torch.randn((rows, 4096 * world), dtype=torch.bfloat16, device="cuda")
    expected = logits.float().argmax(dim=-1).to(torch.int32)
    actual = _ar_gather_argmax(logits, world)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_ar_gather_ties_and_negative_maxima():
    """Ties across ranks go to the lowest rank (the lowest index), as the
    all-gather path's argmax over the packs; a negative max keeps its sign."""
    logits = torch.full((3, 64), -9.0, dtype=torch.float32, device="cuda")
    logits[0, [20, 40, 50]] = 2.5  # ranks 1, 2 and 3 tie
    logits[1, [3, 17]] = -0.75  # every value negative
    logits[2, 63] = 1.0
    out = _ar_gather_argmax(logits, 4)
    assert out.tolist() == [20, 3, 63]
