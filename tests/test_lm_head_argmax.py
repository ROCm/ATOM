import pytest
import torch

from atom.model_ops.lm_head_argmax import lm_head_argmax_pack


def _torch_lm_head_argmax_pack(
    logits: torch.Tensor,
    vocab_start_idx: int,
) -> torch.Tensor:
    local_max_val, local_idx = logits.max(dim=-1)
    global_idx = local_idx + vocab_start_idx
    return torch.stack([local_max_val.float(), global_idx.float()], dim=-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("shape", [(1, 8), (7, 1024), (16, 8192)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_lm_head_argmax_pack(shape, dtype):
    torch.manual_seed(0)
    logits = torch.randn(shape, dtype=dtype, device="cuda")
    vocab_start_idx = 32000

    expected = _torch_lm_head_argmax_pack(logits, vocab_start_idx)
    actual = lm_head_argmax_pack(logits, vocab_start_idx)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_lm_head_argmax_pack_tie_breaks_lowest_global_idx():
    logits = torch.tensor(
        [[1.0, 3.0, 3.0, 2.0], [5.0, 5.0, 4.0, 5.0]],
        dtype=torch.float32,
        device="cuda",
    )
    vocab_start_idx = 128

    actual = lm_head_argmax_pack(logits, vocab_start_idx)
    expected = _torch_lm_head_argmax_pack(logits, vocab_start_idx)

    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
