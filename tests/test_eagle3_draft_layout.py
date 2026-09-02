# SPDX-License-Identifier: MIT
"""What ``Eagle3DraftBuilder`` actually binds, given the layout decision.

The decision itself lives in `atom.spec_decode.draft_kv_layout` and is covered by
`tests/test_draft_kv_layout.py`, which runs on CI. This file covers the other
half -- the binder acting on it -- and needs the atom import environment, so on
CI it skips.

Both levels are needed: without these, deleting the one line that applies the
decision leaves every predicate test green.

Getting it wrong is silent both ways: every layout holds the same element count,
so a mismatched pair reads transposed data without faulting.

No GPU: the pool is a CPU tensor and no kernel runs.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

eagle3_kv_builder = pytest.importorskip(
    "atom.spec_decode.eagle3_kv_builder",
    reason="requires the full atom import environment (aiter)",
)
Eagle3DraftBuilder = eagle3_kv_builder.Eagle3DraftBuilder

L, BLOCKS, BS, NKV, HD = 2, 4, 16, 2, 128
FP8 = eagle3_kv_builder.dtypes.d_dtypes["fp8"]


def _impl(triton: bool):
    """A draft attention impl with only the fields the predicate reads.

    `use_triton_attn` is what `PagedAttentionImpl.__init__` derives from the env,
    the sliding window and head_dim; the binder only sees the result.
    """
    return SimpleNamespace(
        rotary_emb=object(),
        q_norm=None,
        k_norm=None,
        use_triton_attn=triton,
        use_flash_layout=False,
    )


def _builder_and_module(impl, kv_dtype="bf16"):
    """A builder wired to a real (CPU) pool, and one module for it to bind."""
    dtype = torch.bfloat16 if kv_dtype == "bf16" else FP8
    b = object.__new__(Eagle3DraftBuilder)
    b.block_size, b.num_layers, b.num_blocks = BS, L, BLOCKS
    b.num_kv_heads, b.head_dim, b._next_layer_id = NKV, HD, 0
    b.model_runner = SimpleNamespace(
        eagle3_kv_cache=torch.zeros(2, L, BLOCKS, BS, NKV, HD, dtype=dtype),
        eagle3_kv_scale=torch.zeros(2, L, BLOCKS, NKV, BS, dtype=torch.float32),
        config=SimpleNamespace(max_model_len=4096, kv_cache_dtype=kv_dtype),
    )
    module = SimpleNamespace(base_attention=True, use_mla=False, impl=impl)
    return b, module


def test_binder_gives_flash_a_4d_view():
    """The flash arm must bind the allocation unviewed, and record it."""
    impl = _impl(triton=True)
    b, module = _builder_and_module(impl)
    kvt = b.build_kv_cache_tensor(0, module)

    assert impl.use_flash_layout is True
    assert tuple(kvt.k_cache.shape) == (BLOCKS, BS, NKV, HD)
    assert tuple(kvt.v_cache.shape) == (BLOCKS, BS, NKV, HD)
    assert kvt.k_cache.dim() == 4 and kvt.v_cache.dim() == 4


@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8"])
def test_binder_gives_non_flash_the_shuffle_views(kv_dtype):
    """The other arm keeps 5D SHUFFLE K and the 4D V its writer produces.

    Both dtypes, because x = 16 // itemsize reshapes K: 8 for bf16, 16 for fp8.
    """
    impl = _impl(triton=False)
    b, module = _builder_and_module(impl, kv_dtype)
    kvt = b.build_kv_cache_tensor(0, module)

    x = 16 // b.model_runner.eagle3_kv_cache.element_size()
    assert impl.use_flash_layout is False
    assert tuple(kvt.k_cache.shape) == (BLOCKS, NKV, HD // x, BS, x)
    assert tuple(kvt.v_cache.shape) == (BLOCKS, NKV, HD, BS)


@pytest.mark.parametrize("on", [True, False], ids=["flash", "shuffle"])
def test_bound_rank_always_agrees_with_the_flag(on):
    """The invariant a mismatch violates: a 4D K is exactly a flash pool.

    Swapping the two arms' bodies, or dropping the assignment that records the
    choice, breaks this without changing any element count -- which is the
    failure this whole module guards against.
    """
    impl = _impl(triton=on)
    b, module = _builder_and_module(impl)
    kvt = b.build_kv_cache_tensor(0, module)
    assert (kvt.k_cache.dim() == 4) is impl.use_flash_layout


@pytest.mark.parametrize("on", [True, False], ids=["flash", "shuffle"])
def test_both_arms_view_the_same_bytes(on):
    """Whichever arm runs, the binding is a view of the pool, never a copy."""
    impl = _impl(triton=on)
    b, module = _builder_and_module(impl)
    pool = b.model_runner.eagle3_kv_cache
    kvt = b.build_kv_cache_tensor(0, module)
    assert kvt.k_cache.data_ptr() == pool[0, 0].data_ptr()
    assert kvt.v_cache.data_ptr() == pool[1, 0].data_ptr()


def test_the_builder_publishes_its_choice_for_the_proposer():
    """`uses_flash_layout` is what `propose`'s freshness guard keys off.

    ORed across layers: the predicate is per-layer, so a later SHUFFLE layer
    must not clear a flag an earlier flash layer needed.
    """
    b, module = _builder_and_module(_impl(triton=True))
    b.build_kv_cache_tensor(0, module)
    assert b.uses_flash_layout is True

    b2, _ = _builder_and_module(_impl(triton=False))
    b2.build_kv_cache_tensor(
        0, SimpleNamespace(base_attention=True, use_mla=False, impl=_impl(triton=True))
    )
    b2.build_kv_cache_tensor(
        1, SimpleNamespace(base_attention=True, use_mla=False, impl=_impl(triton=False))
    )
    assert b2.uses_flash_layout is True
