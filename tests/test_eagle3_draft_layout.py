# SPDX-License-Identifier: MIT
"""Which KV layout ``Eagle3DraftBuilder`` gives an Eagle3 MHA draft.

The draft's pool holds one set of bytes that a writer in ``rope_cache`` fills
and an attention kernel reads back. Three writers can run, and they do not all
emit the same V:

    fused_qk_norm_rope_cache_quant_shuffle   -> SHUFFLE V   (re-views V itself)
    fused_qk_rope_reshape_and_cache          -> 4D V        (flash_layout=False)
    reshape_and_cache(asm_layout=True)       -> SHUFFLE V

Only the middle one emits a V no prefill reader consumes, and reading a cached
prefix out of it cost a whole-pool conversion. ``_use_flash_layout`` picks those
modules out and gives them a flash pool instead.

Getting it wrong is silent both ways -- every layout holds the same element
count, so a mismatched pair reads transposed data without faulting. A draft that
today gets a working SHUFFLE V (no ATOM_FORCE_ATTN_TRITON, head_dim 128, no
sliding window -- the Kimi-K2.5 shape) must keep it.

Two levels, because the predicate being right is not the binder acting on it:

  * ``_use_flash_layout`` over a fake impl -- the decision.
  * ``build_kv_cache_tensor`` over a real torch pool -- the bound shapes and the
    flag. Without these, deleting the one line that applies the decision leaves
    every test here green.

No GPU: the pool is a CPU tensor, no kernel runs. The module under test imports
aiter, so this still skips on a CPU-only runner.
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
_use_flash_layout = Eagle3DraftBuilder._use_flash_layout


def _impl(*, rotary=True, qk_norm=False, sliding_window=-1, head_dim=128):
    """A draft attention impl with only the fields the predicate reads."""
    norm = object() if qk_norm else None
    return SimpleNamespace(
        rotary_emb=object() if rotary else None,
        q_norm=norm,
        k_norm=norm,
        sliding_window=sliding_window,
        head_dim=head_dim,
        use_flash_layout=False,
    )


@pytest.fixture
def force_triton(monkeypatch):
    """ATOM_FORCE_ATTN_TRITON is read through envs.__getattr__ on every access,
    so setting the variable is enough -- no import-time value to patch."""

    def _set(on: bool):
        monkeypatch.setenv("ATOM_FORCE_ATTN_TRITON", "1" if on else "0")

    return _set


# --- the decision: the two shapes that exist in CI today ---------------------


def test_minimax_m3_draft_gets_flash(force_triton):
    """ATOM_FORCE_ATTN_TRITON=1 sends rope_cache down the 4D-V writer."""
    force_triton(True)
    assert _use_flash_layout(_impl()) is True


def test_kimi_k25_draft_keeps_its_layout(force_triton):
    """No force-triton, head_dim 128, no sliding window: the draft already gets
    a SHUFFLE V from reshape_and_cache(asm_layout=True) and must keep it."""
    force_triton(False)
    assert _use_flash_layout(_impl()) is False


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param({"sliding_window": 4096}, id="sliding_window"),
        pytest.param({"head_dim": 64}, id="head_dim_not_128"),
    ],
)
def test_use_triton_attn_without_the_env(force_triton, kwargs):
    """`use_triton_attn` is an OR of three terms; the env is only one of them."""
    force_triton(False)
    assert _use_flash_layout(_impl(**kwargs)) is True


def test_qk_norm_draft_is_left_alone(force_triton):
    """rope_cache's first branch re-views V to SHUFFLE on its own, so this
    module never sees the 4D V and must not be handed a flash pool."""
    force_triton(True)
    assert _use_flash_layout(_impl(qk_norm=True)) is False


def test_only_one_norm_still_counts_as_the_4d_writer(force_triton):
    """rope_cache's first branch requires BOTH norms; one alone falls through to
    the 4D-V writer, so the predicate must not treat it as the fused path."""
    force_triton(True)
    impl = _impl()
    impl.q_norm = object()  # k_norm stays None
    assert _use_flash_layout(impl) is True


def test_non_attention_module(force_triton):
    """build_kv_cache_tensor passes getattr(module, "impl", None)."""
    force_triton(True)
    assert _use_flash_layout(None) is False


def test_rope_less_draft_is_a_known_gap(force_triton):
    """A KNOWN GAP, pinned so it is not mistaken for a deliberate choice.

    The predicate returns False without rotary_emb. That is right on its own --
    but not when `use_triton_attn` also holds: rope_cache then falls through to
    reshape_and_cache,
    which takes asm_layout=False for a 4D V and writes the very shape this
    predicate exists to avoid. It therefore keeps paying the
    whole-pool permute().contiguous() on every prefix-carrying prefill.

    No such draft exists in-tree today (Eagle3LlamaAttention always builds a
    rotary_emb), which is why it is recorded rather than fixed here -- switching
    that third writer has its own blast radius. If you are here because a
    rope-less draft OOMs on a whole-pool convert: this assertion is the bug, not
    the contract.
    """
    force_triton(True)
    assert _use_flash_layout(_impl(rotary=False)) is False


# --- the binder: what actually gets bound ------------------------------------

L, BLOCKS, BS, NKV, HD = 2, 4, 16, 2, 128


FP8 = eagle3_kv_builder.dtypes.d_dtypes["fp8"]


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


def test_binder_gives_flash_a_4d_view(force_triton):
    """The flash arm must bind the allocation unviewed, and record it."""
    force_triton(True)
    impl = _impl()
    b, module = _builder_and_module(impl)
    kvt = b.build_kv_cache_tensor(0, module)

    assert impl.use_flash_layout is True
    assert tuple(kvt.k_cache.shape) == (BLOCKS, BS, NKV, HD)
    assert tuple(kvt.v_cache.shape) == (BLOCKS, BS, NKV, HD)
    assert kvt.k_cache.dim() == 4 and kvt.v_cache.dim() == 4


@pytest.mark.parametrize("kv_dtype", ["bf16", "fp8"])
def test_binder_gives_non_flash_the_shuffle_views(force_triton, kv_dtype):
    """The other arm keeps 5D SHUFFLE K and the 4D V its writer produces.

    Both dtypes, because x = 16 // itemsize reshapes K: 8 for bf16, 16 for fp8.
    """
    force_triton(False)
    impl = _impl()
    b, module = _builder_and_module(impl, kv_dtype)
    kvt = b.build_kv_cache_tensor(0, module)

    x = 16 // b.model_runner.eagle3_kv_cache.element_size()
    assert impl.use_flash_layout is False
    assert tuple(kvt.k_cache.shape) == (BLOCKS, NKV, HD // x, BS, x)
    assert tuple(kvt.v_cache.shape) == (BLOCKS, NKV, HD, BS)


@pytest.mark.parametrize("on", [True, False], ids=["flash", "shuffle"])
def test_bound_rank_always_agrees_with_the_flag(force_triton, on):
    """The invariant a mismatch violates: a 4D K is exactly a flash pool.

    Swapping the two arms' bodies, or dropping the assignment that records the
    choice, breaks this without changing any element count -- which is the
    failure this whole module guards against.
    """
    force_triton(on)
    impl = _impl()
    b, module = _builder_and_module(impl)
    kvt = b.build_kv_cache_tensor(0, module)
    assert (kvt.k_cache.dim() == 4) is impl.use_flash_layout


@pytest.mark.parametrize("on", [True, False], ids=["flash", "shuffle"])
def test_both_arms_view_the_same_bytes(force_triton, on):
    """Whichever arm runs, the binding is a view of the pool, never a copy."""
    force_triton(on)
    impl = _impl()
    b, module = _builder_and_module(impl)
    pool = b.model_runner.eagle3_kv_cache
    kvt = b.build_kv_cache_tensor(0, module)
    assert kvt.k_cache.data_ptr() == pool[0, 0].data_ptr()
    assert kvt.v_cache.data_ptr() == pool[1, 0].data_ptr()
