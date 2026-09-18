# SPDX-License-Identifier: MIT
"""Pinned model math, isolated per differential test."""

import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from .reference import load_reference


@pytest.fixture
def reference():
    directory = os.environ.get("ATOM_DSV41_REFERENCE")
    if not directory:
        pytest.skip("Set ATOM_DSV41_REFERENCE for pinned model methods")
    with load_reference(directory) as module:
        yield module


@pytest.fixture
def single_rank(monkeypatch):
    from atom.models.deepseek_v41 import attention, model

    from atom.model_ops import embed_head, layernorm, linear
    from atom.model_ops import moe as fused_moe

    group = SimpleNamespace(rank_in_group=0, world_size=1)
    # `fused_moe`, not the V4.1 `moe` module: the routed experts are V4's
    # `FusedMoE`, which reads the group where it lives. `layers` is absent
    # because the output projection reduces through `RowParallelLinear` now,
    # which reads the group in `linear`.
    for module in (linear, attention, model, fused_moe, embed_head):
        monkeypatch.setattr(module, "get_tp_group", lambda: group)
    monkeypatch.setattr(
        layernorm, "get_tensor_model_parallel_world_size", lambda: group.world_size
    )
    return group


@pytest.fixture
def small_config():
    """Small model graph retaining D=512 and eight local attention heads."""
    from atom.models.deepseek_v41.config import DeepseekV41TextConfig

    return DeepseekV41TextConfig(
        hidden_size=64,
        head_dim=512,
        num_attention_heads=8,
        q_lora_rank=32,
        o_groups=1,
        o_lora_rank=32,
        rms_norm_eps=1e-20,
        sliding_window=4,
        index_n_heads=32,
        index_head_dim=32,
        index_topk=4,
        candidate_block_size=2,
        candidate_topk_blocks=4,
        max_position_embeddings=32,
        num_hidden_layers=5,
        num_nextn_predict_layers=0,
        compress_ratios=(0, 2, 2, 1, 1),
        kv_source_layer_ids=(1, 3),
        index_source_layer_ids=(1, 3, 4),
        candidate_source_layer_id=3,
    )


@pytest.fixture
def attention_contract(monkeypatch, reference):
    """Check the real backend before controlling its rounding for graph tests.

    FP8 projection quantization is discontinuous at bin boundaries. Validate
    each actual BF16 attention output against upstream independently, then feed
    the upstream output to the remaining graph so its assertions isolate model
    composition. This is not an end-to-end numerical acceptance test.
    """
    import torch
    from atom.models.deepseek_v41 import attention

    captured = []
    original = reference.sparse_attn

    def capture(*args):
        output = original(*args)
        captured.append(output.detach().clone())
        return output

    monkeypatch.setattr(reference, "sparse_attn", capture)

    @contextmanager
    def check(outputs):
        remaining = iter(outputs)

        def checked(function):
            def call(*args, **kwargs):
                actual = function(*args, **kwargs)
                expected = next(remaining).flatten(0, 1).to(actual.device)
                assert torch.isfinite(actual).all()
                error = (actual.float() - expected.float()).norm()
                assert error <= 3e-3 * expected.float().norm().clamp_min(1e-30)
                return expected.clone()

            return call

        with pytest.MonkeyPatch.context() as patch:
            for name in ("sparse_attn_v4_paged_decode", "sparse_attn_v4_paged_prefill"):
                patch.setattr(attention, name, checked(getattr(attention, name)))
            yield
        assert next(remaining, None) is None

    return captured, check
