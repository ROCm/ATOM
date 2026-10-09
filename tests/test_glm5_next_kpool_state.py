# SPDX-License-Identifier: MIT

"""GLM-5.3 k-pool state contracts (CPU plus ROCm-only kernel cases)."""

from types import SimpleNamespace

import pytest
import torch

from atom.utils import forward_context

GLM5 = pytest.importorskip(
    "atom.models.glm5_next",
    reason="the GLM-5.3 model imports the AITER runtime",
    exc_type=ImportError,
)
from atom.model_ops.glm5_next import indexer as KPOOL_INDEXER


def _config_for_normalize(**overrides):
    values = {
        "num_hidden_layers": 3,
        "num_attention_heads": 8,
        "linear_attn_config": {"kda_layers": [0, 2]},
        "layer_types": [
            "linear_attention",
            "deepseek_sparse_attention",
            "linear_attention",
        ],
        "n_routed_experts": 4,
        "qk_nope_head_dim": 256,
        "qk_rope_head_dim": 0,
        "kv_lora_rank": 512,
        "index_head_dim": 128,
        "index_kpool": 4,
        "index_topk": 2048,
        "index_kpool_always_select_tail": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_normalize_derives_a_missing_full_attention_list():
    config = _config_for_normalize()

    GLM5._normalize_glm5_next_config(config)

    assert config.glm5_kda_layers == [0, 2]
    assert config.glm5_full_attn_layers == [1]


def test_normalize_rejects_conflicting_attention_layouts():
    config = _config_for_normalize(
        linear_attn_config={
            "kda_layers": [0, 2],
            "full_attn_layers": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="disjoint"):
        GLM5._normalize_glm5_next_config(config)


def test_normalize_rejects_non_nope_geometry():
    config = _config_for_normalize(qk_rope_head_dim=64)

    with pytest.raises(ValueError, match="qk_rope_head_dim"):
        GLM5._normalize_glm5_next_config(config)


def test_dummy_custom_op_result_is_fresh_fp32_like_its_fake(monkeypatch):
    """Runtime and fake implementations must have the same alias/dtype contract."""
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata=None,
            context=SimpleNamespace(is_dummy_run=True),
        ),
    )
    weights = torch.ones((2, 3), dtype=torch.bfloat16)
    out = KPOOL_INDEXER._sparse_attn_indexer_kpool(
        hidden_states=torch.empty((2, 4)),
        kv_cache=torch.empty(1),
        q_fp8=torch.empty((2, 1, 4)),
        k=torch.empty((2, 4)),
        gate_score=torch.empty((2, 4)),
        weights=weights,
        compress_ape=torch.empty((4, 4)),
        tail_cache=torch.empty((1, 2, 4, 4)),
        state_slot_idx_in=torch.zeros(2, dtype=torch.int32),
        state_slot_idx=torch.zeros(2, dtype=torch.int32),
        positions=torch.arange(2),
        sparse_kv_indices_buffer=torch.empty(1),
        topk_tokens=4,
        index_kpool=4,
        head_dim=4,
        max_model_len=16,
        topk_out_width=128,
        scale_fmt="ue8m0",
        stable_topk=False,
    )

    assert out.dtype == torch.float32
    assert out.data_ptr() != weights.data_ptr()
    assert torch.equal(out, weights.float())


def test_one_row_cached_chunk_can_close_a_pool_from_the_input_tail(monkeypatch):
    """A chunk starting at absolute position 103 closes pool 100..103 at row 0."""
    seen = {}

    def pool_and_rotate(pool_k, pool_gate, _ape):
        seen["pool_k"] = pool_k.clone()
        seen["pool_gate"] = pool_gate.clone()
        return pool_k.sum(dim=1)

    def pool_slot_mapping(_bt, pool_ids, _req_idx, _rows):
        seen["pool_ids"] = pool_ids.clone()
        return torch.where(pool_ids >= 0, torch.full_like(pool_ids, 7), pool_ids)

    def cache_write(pooled, _cache, slots, *_args, **_kwargs):
        seen["pooled"] = pooled.clone()
        seen["slots"] = slots.clone()

    monkeypatch.setattr(KPOOL_INDEXER.kpool, "pool_and_rotate", pool_and_rotate)
    monkeypatch.setattr(KPOOL_INDEXER.kpool, "pool_slot_mapping", pool_slot_mapping)
    monkeypatch.setattr(KPOOL_INDEXER, "indexer_k_quant_and_cache", cache_write)

    tail = torch.zeros((12, 2, 4, 2), dtype=torch.bfloat16)
    for phase in range(3):
        tail[5, 0, phase] = 100 + phase
        tail[5, 1, phase] = 200 + phase

    KPOOL_INDEXER._kpool_write_completed_pools(
        kv_cache=torch.empty(1),
        k=torch.full((1, 2), 103, dtype=torch.bfloat16),
        gate_score=torch.full((1, 2), 203, dtype=torch.bfloat16),
        positions=torch.tensor([103]),
        pool_bt=torch.empty((1, 1), dtype=torch.int32),
        req_idx=torch.tensor([0]),
        compress_ape=torch.zeros((4, 2)),
        index_kpool=4,
        head_dim=2,
        scale_fmt="ue8m0",
        pool_rows=16,
        chunk_start=torch.tensor([103]),
        tail_cache=tail,
        state_slot_idx_in=torch.tensor([5]),
        state_slot_idx=torch.tensor([9]),
    )

    assert seen["pool_k"][:, :, 0].tolist() == [[100, 101, 102, 103]]
    assert seen["pool_gate"][:, :, 0].tolist() == [[200, 201, 202, 203]]
    assert seen["pool_ids"].tolist() == [25]
    assert seen["slots"].tolist() == [7]


def test_invalid_output_slot_cannot_publish_a_completed_pool(monkeypatch):
    seen = {}
    monkeypatch.setattr(
        KPOOL_INDEXER.kpool,
        "pool_and_rotate",
        lambda pool_k, _pool_gate, _ape: pool_k.sum(dim=1),
    )

    def pool_slot_mapping(_bt, pool_ids, _req_idx, _rows):
        seen["pool_ids"] = pool_ids.clone()
        return pool_ids

    monkeypatch.setattr(KPOOL_INDEXER.kpool, "pool_slot_mapping", pool_slot_mapping)
    monkeypatch.setattr(
        KPOOL_INDEXER, "indexer_k_quant_and_cache", lambda *_args, **_kwargs: None
    )

    KPOOL_INDEXER._kpool_write_completed_pools(
        kv_cache=torch.empty(1),
        k=torch.ones((4, 2), dtype=torch.bfloat16),
        gate_score=torch.ones((4, 2), dtype=torch.bfloat16),
        positions=torch.arange(4),
        pool_bt=torch.empty((1, 1), dtype=torch.int32),
        req_idx=torch.zeros(4, dtype=torch.int64),
        compress_ape=torch.zeros((4, 2)),
        index_kpool=4,
        head_dim=2,
        scale_fmt="ue8m0",
        pool_rows=16,
        state_slot_idx=torch.tensor([-1]),
    )

    assert seen["pool_ids"].tolist() == [-1, -1, -1, -1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="exercises Triton tail copy")
def test_prefill_tail_fork_copies_prior_rows_into_the_output_slot():
    device = torch.device("cuda")
    tail = torch.zeros((12, 2, 4, 128), dtype=torch.bfloat16, device=device)
    for phase in range(2):
        tail[5, 0, phase] = 100 + phase
        tail[5, 1, phase] = 200 + phase

    KPOOL_INDEXER.kpool.kpool_seed_tail(
        tail,
        torch.full((1, 128), 102, dtype=torch.bfloat16, device=device),
        torch.full((1, 128), 202, dtype=torch.bfloat16, device=device),
        torch.tensor([102], dtype=torch.int64, device=device),
        torch.tensor([0, 1], dtype=torch.int32, device=device),
        torch.tensor([9], dtype=torch.int32, device=device),
        4,
        slot_idx_in=torch.tensor([5], dtype=torch.int32, device=device),
    )

    assert tail[9, 0, :, 0].cpu().tolist() == [100, 101, 102, 0]
    assert tail[9, 1, :, 0].cpu().tolist() == [200, 201, 202, 0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="exercises Triton tail copy")
def test_decode_tail_fork_reads_input_and_materializes_output_slot():
    device = torch.device("cuda")
    tail = torch.zeros((12, 2, 4, 128), dtype=torch.bfloat16, device=device)
    for phase in range(3):
        tail[5, 0, phase] = phase + 1
        tail[5, 1, phase] = phase + 11

    out = KPOOL_INDEXER.kpool.kpool_decode_stash_and_pool(
        tail,
        torch.full((1, 128), 4, dtype=torch.bfloat16, device=device),
        torch.full((1, 128), 14, dtype=torch.bfloat16, device=device),
        torch.tensor([3], dtype=torch.int64, device=device),
        torch.tensor([9], dtype=torch.int32, device=device),
        torch.zeros((4, 128), dtype=torch.float32, device=device),
        4,
        slot_idx_in=torch.tensor([5], dtype=torch.int32, device=device),
    )

    assert out.shape == (1, 128)
    assert tail[9, 0, :, 0].cpu().tolist() == [1, 2, 3, 4]
    assert tail[9, 1, :, 0].cpu().tolist() == [11, 12, 13, 14]
