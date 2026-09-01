# SPDX-License-Identifier: MIT
"""Which KV layout ``Eagle3DraftBuilder`` gives an Eagle3 MHA draft.

The draft's pool holds one set of bytes that a writer in ``rope_cache`` fills
and an attention kernel reads back. Three writers can run, and they do not all
emit the same V:

    fused_qk_norm_rope_cache_quant_shuffle   -> SHUFFLE V   (re-views V itself)
    fused_qk_rope_reshape_and_cache          -> VHD V       (flash_layout=False)
    reshape_and_cache(asm_layout=True)       -> SHUFFLE V

Only the middle one produces a V that no prefill reader consumes, which is what
made reading a cached prefix cost a whole-pool conversion. ``_use_flash_layout``
picks those modules out and hands them a flash pool instead -- same writer, one
flag flipped, and unified_attention reads it in prefill and decode alike.

Getting that predicate wrong is silent both ways: every layout holds the same
element count, so a mismatched pair reads transposed data and corrupts
attention without faulting. In particular a draft that today gets a working
SHUFFLE V (no ATOM_FORCE_ATTN_TRITON, head_dim 128, no sliding window -- the
Kimi-K2.5 shape) must keep it. These cases pin that.

The predicate is a pure function of the impl, so no GPU is needed -- but the
module under test imports aiter, so this skips on a CPU-only runner.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

eagle3_kv_builder = pytest.importorskip(
    "atom.spec_decode.eagle3_kv_builder",
    reason="requires the full atom import environment (aiter)",
)

_use_flash_layout = eagle3_kv_builder.Eagle3DraftBuilder._use_flash_layout


def _impl(*, rotary=True, qk_norm=False, sliding_window=-1, head_dim=128):
    """A draft attention impl with only the fields the predicate reads."""
    norm = object() if qk_norm else None
    return SimpleNamespace(
        rotary_emb=object() if rotary else None,
        q_norm=norm,
        k_norm=norm,
        sliding_window=sliding_window,
        head_dim=head_dim,
    )


@pytest.fixture
def force_triton(monkeypatch):
    """ATOM_FORCE_ATTN_TRITON is read through envs.__getattr__ on every access,
    so setting the variable is enough -- no import-time value to patch."""

    def _set(on: bool):
        monkeypatch.setenv("ATOM_FORCE_ATTN_TRITON", "1" if on else "0")

    return _set


# --- the two shapes that exist in CI today ----------------------------------


def test_minimax_m3_draft_gets_flash(force_triton):
    """ATOM_FORCE_ATTN_TRITON=1 sends rope_cache down the VHD writer."""
    force_triton(True)
    assert _use_flash_layout(_impl()) is True


def test_kimi_k25_draft_keeps_its_layout(force_triton):
    """No force-triton, head_dim 128, no sliding window: the draft already gets
    a SHUFFLE V from reshape_and_cache(asm_layout=True) and must keep it."""
    force_triton(False)
    assert _use_flash_layout(_impl()) is False


# --- the other two ways rope_cache reaches the VHD writer -------------------


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


# --- modules the predicate must not claim -----------------------------------


def test_qk_norm_draft_is_left_alone(force_triton):
    """rope_cache's first branch re-views V to SHUFFLE on its own, so this
    module never sees a VHD pool and must not be handed a flash one."""
    force_triton(True)
    assert _use_flash_layout(_impl(qk_norm=True)) is False


def test_no_rope_draft_is_left_alone(force_triton):
    """Without rotary_emb, rope_cache falls through to reshape_and_cache."""
    force_triton(True)
    assert _use_flash_layout(_impl(rotary=False)) is False


def test_non_attention_module(force_triton):
    """build_kv_cache_tensor passes getattr(module, "impl", None), which is None
    for anything that is not an attention layer."""
    force_triton(True)
    assert _use_flash_layout(None) is False


def test_only_one_norm_still_counts_as_the_vhd_writer(force_triton):
    """rope_cache's first branch requires BOTH norms; one alone falls through to
    the VHD writer, so the predicate must not treat it as the fused path."""
    force_triton(True)
    impl = _impl()
    impl.q_norm = object()  # k_norm stays None
    assert _use_flash_layout(impl) is True
