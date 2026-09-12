# SPDX-License-Identifier: MIT
"""Which backend `_kv_b_proj_gather` picks, and when it asks.

Dispatch policy only -- no device. The FlyDSL gather serves one corner (gfx950,
page_size 1, fp8 cache and fp8 weight); a bf16 cache, an unquantized weight or
an MXFP4 one belong to the Triton op, which takes the same arguments. Getting
that wrong is not a slow path, it is a `ValueError` out of aiter mid-forward
that kills the engine, which is how three CI accuracy jobs died.

Both aiter functions are stubbed: what is under test is that ATOM consults the
predicate rather than the import, and consults it once.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("triton", reason="attention_mla defines @triton.jit kernels")
pytest.importorskip("aiter", reason="attention_mla imports the AITER runtime")

from atom.model_ops import attention_mla as mla


def _impl(**over):
    """The slice of MLAAttention that `_kv_b_proj_gather` touches."""
    fields = {
        "kv_b_proj": SimpleNamespace(weight=SimpleNamespace(is_shuffled=True)),
        "_k_scale": None,
        "use_flydsl_gather_kv_b_proj": True,
        "_flydsl_gather_ok": None,
    }
    return SimpleNamespace(**{**fields, **over})


@pytest.fixture
def spies(monkeypatch):
    calls = SimpleNamespace(flydsl=0, triton=0, asked=0)

    def supported(*_a, **_k):
        calls.asked += 1
        return calls.answer

    monkeypatch.setattr(mla, "_FLYDSL_GATHER_AVAILABLE", True)
    monkeypatch.setattr(mla, "gather_kv_b_proj_flydsl_supported", supported)
    monkeypatch.setattr(
        mla,
        "gather_kv_b_proj_flydsl",
        lambda *a, **k: setattr(calls, "flydsl", calls.flydsl + 1),
    )
    monkeypatch.setattr(
        mla,
        "gather_kv_b_proj",
        lambda *a, **k: setattr(calls, "triton", calls.triton + 1),
    )
    monkeypatch.setattr(mla, "_maybe_view_mxfp4_weight_for_gather", lambda _proj, w: w)
    return calls


def _gather(impl):
    mla.MLAAttention._kv_b_proj_gather(impl, None, None, None, None, None, None)


@pytest.mark.parametrize("answer, flydsl, triton", [(True, 1, 0), (False, 0, 1)])
def test_backend_follows_the_predicate_not_the_import(spies, answer, flydsl, triton):
    """A shape the kernel declines must reach the Triton op, not raise."""
    spies.answer = answer
    _gather(_impl())
    assert (spies.flydsl, spies.triton) == (flydsl, triton)


def test_the_predicate_is_asked_once(spies):
    """Its terms are fixed by the weights and the cache, so caching is sound;
    asking per gather would put a Python-level shape walk on every layer."""
    spies.answer = True
    impl = _impl()
    for _ in range(5):
        _gather(impl)
    assert spies.asked == 1
    assert spies.flydsl == 5


def test_env_off_never_asks(spies):
    """ATOM_USE_FLYDSL_GATHER_KV_B_PROJ=0 is a decision, not a question."""
    spies.answer = True
    impl = _impl(use_flydsl_gather_kv_b_proj=False)
    _gather(impl)
    assert (spies.flydsl, spies.triton) == (0, 1)
    assert spies.asked == 0
