"""Numeric parity: ATOM's QSA indexer projections vs the vLLM reference.

Covers the two orderings that fail silently if transposed: Q is normalized
then rotated at its own position, while K stays raw until after pooling and is
rotated at its group's FIRST token position.
"""

import math
from types import SimpleNamespace

import pytest
import torch

from tests.qwen3_8_flash_next.parity_harness import init_single_rank

# Real Qwen3.8-Flash-Next indexer geometry: ATOM's GemmaRMSNorm has no torch
# fallback and its aiter kernel aborts the process on unusual shapes.
HIDDEN, N_HEADS, HEAD_DIM = 2560, 4, 128
# head_dim 256 x partial_rotary_factor 0.25, as in the checkpoint.
ROTARY_DIM = 64
DTYPE = torch.bfloat16
BUDGET, COMPRESS = 2048, 4


def _rotary():
    """The real indexer RoPE: aiter's 1D rotary at the indexer head size.

    `apply_qsa_rope` reshapes by `head_size`, so the instance has to be built
    for the indexer's 128 rather than the attention layer's 256 -- the cos/sin
    cache depends only on `rotary_dim` and is identical either way.
    """
    from aiter.rotary_embedding import get_rope

    return get_rope(
        head_size=HEAD_DIM,
        rotary_dim=ROTARY_DIM,
        max_position=4096,
        base=10_000_000,
        is_neox_style=True,
        rope_scaling=None,
    )


def _config():
    return SimpleNamespace(
        hidden_size=HIDDEN,
        indexer_n_heads=N_HEADS,
        indexer_kv_heads=1,
        indexer_head_dim=HEAD_DIM,
        indexer_budget=BUDGET,
        indexer_compress_ratio=COMPRESS,
        rms_norm_eps=1e-6,
    )


@pytest.fixture(scope="module")
def indexer():
    init_single_rank()
    from atom.model_ops.qwen3_8_flash_next.indexer import Qwen3_8FlashNextIndexer

    rotary = _rotary()
    # The aiter Gemma-RMSNorm kernel requires weight and activation to
    # share a dtype, so materialize the module in the serving dtype.
    module = (
        Qwen3_8FlashNextIndexer(_config(), rotary_emb=rotary, prefix="idx").cuda().to(DTYPE)
    )
    torch.manual_seed(0)
    with torch.no_grad():
        module.index_qk_proj.weight.copy_(
            (torch.randn((N_HEADS + 1) * HEAD_DIM, HIDDEN) * 0.05).cuda().to(DTYPE)
        )
        module.q_layernorm.weight.copy_((torch.randn(HEAD_DIM) * 0.1).cuda().to(DTYPE))
        module.k_layernorm.weight.copy_((torch.randn(HEAD_DIM) * 0.1).cuda().to(DTYPE))
    return module


def _gemma_rmsnorm(x, weight, eps):
    variance = x.float().square().mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * (1.0 + weight.float())).to(
        x.dtype
    )


def test_project_qk_normalizes_q_and_keeps_k_raw(indexer):
    torch.manual_seed(1)
    tokens = 12
    hidden = torch.randn(tokens, HIDDEN).cuda().to(DTYPE)
    positions = torch.arange(tokens).cuda()

    with torch.no_grad():
        q, k = indexer.project_qk(hidden, positions)
        qk = torch.nn.functional.linear(hidden, indexer.index_qk_proj.weight)
        q_raw, k_raw = qk.split((N_HEADS * HEAD_DIM, HEAD_DIM), dim=-1)

    # K must come through the projection untouched -- no norm, no rotation.
    torch.testing.assert_close(k, k_raw.reshape(-1, 1, HEAD_DIM), rtol=0, atol=0)
    # Q must be Gemma-normalized per head, then rotated.
    expected_q = _gemma_rmsnorm(
        q_raw.reshape(-1, HEAD_DIM), indexer.q_layernorm.weight, 1e-6
    ).reshape(tokens, N_HEADS, HEAD_DIM)
    assert not torch.allclose(q, expected_q), "rope was not applied to Q"
    assert q.shape == (tokens, N_HEADS, HEAD_DIM)


def test_pooling_happens_before_normalization(indexer):
    """Pool raw, then norm -- normalizing first gives different keys."""
    torch.manual_seed(2)
    tokens = 16
    raw = torch.randn(tokens, 1, HEAD_DIM).cuda().to(DTYPE)

    pooled_then_normed = _gemma_rmsnorm(
        indexer.pool_key_groups(raw, COMPRESS).reshape(-1, HEAD_DIM),
        indexer.k_layernorm.weight,
        1e-6,
    )
    normed_then_pooled = indexer.pool_key_groups(
        _gemma_rmsnorm(
            raw.reshape(-1, HEAD_DIM), indexer.k_layernorm.weight, 1e-6
        ).reshape(tokens, 1, HEAD_DIM),
        COMPRESS,
    ).reshape(-1, HEAD_DIM)

    assert pooled_then_normed.shape == (tokens // COMPRESS, HEAD_DIM)
    assert not torch.allclose(
        pooled_then_normed.float(), normed_then_pooled.float(), atol=1e-3
    )


def test_pool_key_groups_drops_the_ragged_tail(indexer):
    raw = torch.arange(4 * COMPRESS + 3, dtype=torch.float32).cuda().to(DTYPE)
    raw = raw.reshape(-1, 1, 1).expand(-1, 1, HEAD_DIM).contiguous()
    pooled = indexer.pool_key_groups(raw, COMPRESS)
    assert pooled.shape == (4, 1, HEAD_DIM)
    # First group averages tokens 0..COMPRESS-1.
    assert math.isclose(pooled[0, 0, 0].item(), (COMPRESS - 1) / 2, rel_tol=1e-6)


def test_selection_widths_follow_the_contract(indexer):
    assert indexer.block_topk == BUDGET // COMPRESS
    assert indexer.output_width == BUDGET + COMPRESS - 1
