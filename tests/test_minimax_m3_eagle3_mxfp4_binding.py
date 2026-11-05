# SPDX-License-Identifier: MIT
"""How an EAGLE3 draft's attention gets MXFP4 pages under ``--kv-cache-dtype fp4``.

The pool is the DRAFT builder's, not the target builder's: for an EAGLE3 draft
(an MHA llama) ``draft_kv_builder`` returns a ``DraftKvBuilder`` (the aiter backend
owns a draft's pool), ``ModelRunner`` binds the draft's modules through it first,
and the target builder's walk never holds a row for them. What the draft's module
ends up with is ``Mxfp4KvLayout``'s pages. Also pinned: the layer picks the MXFP4
impl only for fp4. Needs aiter like the
other ``aiter_attention`` builder tests, so the CPU CI job skips it.
"""

import types

import pytest
import torch
from torch import nn

pytest.importorskip("aiter")
from aiter import dtypes

from atom.model_ops.attentions import aiter_attention as mod
from atom.model_ops.attentions.pool_layout.pool_rows import KvGeometry
from atom.model_ops.minimax_m3.mxfp4_pages import PAGE_BYTES, Mxfp4KvLayout
from atom.spec_decode import draft_kv

M3_HF = types.SimpleNamespace(architectures=["MiniMaxM3SparseForConditionalGeneration"])
LAYER_OFFSET = 62  # the target's layer count: the draft's layer_num starts there
DRAFT_HEADS = 4  # MiniMax-M3-EAGLE3-GQA: 64 q heads over 4 kv heads
DRAFT_HF = types.SimpleNamespace(num_key_value_heads=DRAFT_HEADS, head_dim=128)


class _Attn(nn.Module):
    """One paged, non-MLA attention module, as ``Eagle3LlamaAttention.attn`` is."""

    def __init__(self, layer_num, heads, head_dim=128):
        super().__init__()
        self.base_attention = True
        self.use_mla = False
        self.layer_num = layer_num
        self.num_kv_heads, self.head_dim = heads, head_dim
        self.impl = types.SimpleNamespace(is_indexed_sparse_attention=False)


class _Norm(nn.Module):
    """A sibling the binding must leave alone."""


def config(kv="fp4"):
    return types.SimpleNamespace(
        kv_cache_dtype=kv,
        index_cache_dtype="fp8",
        hf_config=M3_HF,
        max_model_len=4096,
    )


def draft_stack():
    return nn.Sequential(_Norm(), _Attn(LAYER_OFFSET, DRAFT_HEADS), _Norm())


@pytest.fixture
def mha_backend(monkeypatch):
    monkeypatch.setattr(dtypes, "d_dtypes", {"fp8": torch.uint8})
    monkeypatch.setattr("atom.utils.selector.attn_family", lambda _hf: "mha")
    monkeypatch.setattr(
        "atom.utils.selector.get_attn_backend", lambda _family: mod.AiterBackend
    )


def runner(draft_model, target_model=None, kv="fp4"):
    return types.SimpleNamespace(
        config=config(kv),
        world_size=1,
        model=target_model if target_model is not None else nn.Sequential(),
        drafter=types.SimpleNamespace(model=draft_model),
        attn_metadata_builder=types.SimpleNamespace(block_size=128),
        # EagleProposer._build_draft_model sets runner.draft_kv_builder, so the
        # draft does not share the target's pool
        draft_shares_kv_pool=lambda: False,
    )


# --- the draft builder owns the draft's pool ---------------------------------


def test_an_eagle3_draft_gets_a_draft_kv_builder(mha_backend):
    """The llama draft resolves to the MHA backend, which owns a draft's pool."""
    assert mod.AiterBackend.DRAFT_OWNS_KV_POOL is True
    builder = draft_kv.draft_kv_builder(runner(draft_stack()), DRAFT_HF)
    assert isinstance(builder, draft_kv.DraftKvBuilder)


def test_the_draft_module_binds_to_mxfp4_pages_of_its_own_pool(mha_backend):
    stack = draft_stack()
    builder = draft_kv.draft_kv_builder(runner(stack), DRAFT_HF)
    pool = builder.kv_pool
    assert pool.layout == Mxfp4KvLayout() and pool.scale_fields == []
    pool.allocate(4, "cpu")

    attn = stack[1]
    bound = builder.build_kv_cache_tensor(attn)
    assert bound is not None and bound.layer_num == LAYER_OFFSET
    for pages in (bound.k_cache, bound.v_cache):
        assert pages.dtype == torch.uint8
        assert tuple(pages.shape) == (4, DRAFT_HEADS, PAGE_BYTES)
        assert pages.stride(-1) == 1
    # the module the impl reads is bound to the same views, with no scale planes
    assert attn.k_cache is bound.k_cache and attn.v_cache is bound.v_cache
    assert bound.k_scale is None and bound.v_scale is None
    # K and V are separate fields: one head's page is 8704 B, not shared
    assert bound.k_cache.data_ptr() != bound.v_cache.data_ptr()

    # the siblings are not the builder's, so the runner falls through to the target's
    assert builder.build_kv_cache_tensor(stack[0]) is None


def _runners_answer(has_draft_builder):
    """``ModelRunner.draft_shares_kv_pool`` as the runner itself computes it."""
    model_runner = pytest.importorskip("atom.model_engine.model_runner")
    fake = types.SimpleNamespace(
        config=types.SimpleNamespace(speculative_config=object()), drafter=object()
    )
    if has_draft_builder:
        fake.draft_kv_builder = object()
    return model_runner.ModelRunner.draft_shares_kv_pool(fake)


def test_a_draft_with_a_builder_does_not_share_the_target_pool():
    # EagleProposer._build_draft_model sets runner.draft_kv_builder for an EAGLE3
    # draft; without one the draft's layers would land in the target's pools
    assert _runners_answer(True) is False
    assert _runners_answer(False) is True


def test_the_target_builder_holds_no_row_for_the_draft(mha_backend):
    """The draft's attention is not a row of the target's pool: the draft builder
    claims it, and the target builder's walk covers the target's model only."""
    target_attn = _Attn(0, 2)
    b = object.__new__(mod.AiterAttentionMetadataBuilder)
    b.block_size, b.block_ratio, b._has_sparse_attention = 128, 1, False
    b.model_runner = runner(draft_stack(), target_model=nn.Sequential(target_attn))
    b.model_runner.draft_shares_kv_pool = lambda: _runners_answer(True)
    geometries = {k: n for k, n in b.row_counts().items() if isinstance(k, KvGeometry)}
    assert geometries == {KvGeometry(2, 128, Mxfp4KvLayout()): 1}  # no 4-head row


def test_a_draft_whose_heads_are_not_128_wide_is_refused_at_the_pool(mha_backend):
    hf = types.SimpleNamespace(num_key_value_heads=DRAFT_HEADS, head_dim=64)
    builder = draft_kv.draft_kv_builder(runner(draft_stack()), hf)
    with pytest.raises(ValueError, match="MXFP4 KV page"):
        _ = builder.kv_pool


# --- the layer: which impl --------------------------------------------------


def _eagle3_llama(monkeypatch):
    llama = pytest.importorskip("atom.models.eagle3_llama")
    monkeypatch.setattr(llama, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(llama, "QKVParallelLinear", lambda **kw: nn.Identity())
    monkeypatch.setattr(llama, "RowParallelLinear", lambda **kw: nn.Identity())
    monkeypatch.setattr(llama, "get_rope", lambda *a, **kw: nn.Identity())
    seen = {}

    def attention(*args, **kwargs):
        seen.update(kwargs)
        return nn.Identity()

    monkeypatch.setattr(llama, "Attention", attention)
    return llama, seen


def _attention(llama, cache_config):
    hf = types.SimpleNamespace(
        head_dim=128,
        rope_theta=5_000_000,
        max_position_embeddings=4096,
        use_sliding_window=False,
    )
    return llama.Eagle3LlamaAttention(
        config=hf,
        hidden_size=6144,
        num_heads=64,
        num_kv_heads=DRAFT_HEADS,
        cache_config=cache_config,
        layer_num=LAYER_OFFSET,
    )


def test_only_fp4_takes_the_mxfp4_impl(monkeypatch):
    llama, seen = _eagle3_llama(monkeypatch)
    from atom.model_ops.minimax_m3.eagle3_attn_mxfp4 import (
        Eagle3Mxfp4PagedAttentionImpl,
    )

    _attention(llama, "fp4")
    assert seen["impl_cls"] is Eagle3Mxfp4PagedAttentionImpl
    assert seen["kv_cache_dtype"] == "fp4" and seen["layer_num"] == LAYER_OFFSET
    for dtype in ("bf16", "fp8"):
        seen.clear()
        _attention(llama, dtype)
        assert "impl_cls" not in seen and seen["kv_cache_dtype"] == dtype
