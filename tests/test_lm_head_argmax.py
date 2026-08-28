# SPDX-License-Identifier: MIT

import pytest
import torch

pytest.importorskip("aiter")

from atom.model_ops.lm_head_argmax import lm_head_argmax_pack

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
