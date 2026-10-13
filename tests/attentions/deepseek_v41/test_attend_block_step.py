# SPDX-License-Identifier: MIT
"""`_attend_block` attends the Step its caller chose.

Bounded replay re-enters attention with a *tail* Step -- the rows the late
layers still owe -- while the cache goes on holding the whole forward's Step,
because the row-view context swaps `metadata.step` and nothing touches
`cache.current_step`. An `_attend_block` that read the cache therefore gave
the tail's rows a Step describing every row: 128 rows attending under a Step
that says 1024, which reaches `rope_quant_window`'s reshape by `step.width`.

Decoder bounded replay is on by default (`enable_decoder_swa_bounded_replay`),
so that is the ordinary native path, not a flag someone opts into.

The substitution itself is still needed, but only where a capture freezes
this call's arguments, so it lives in the plugin's own wrapper. Both
directions are asserted: core must not substitute, and the plugin's wrapper
must -- a core that simply stopped reading the cache would pass the first
test while leaving every graph replay attending the warmup batch's Step.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

_ABSENT = object()


def _attention_stub(seen: list):
    """The pieces `_attend_block` calls, each a no-op that records nothing
    except which Step reached the attention call."""
    stub = SimpleNamespace(
        heads=1,
        head_dim=1,
        _fork_compress=lambda *a, **k: None,
        project_qkv=lambda *a, **k: (None, None),
        qk_norm=lambda *a, **k: (None, None, None),
        _fork_select=lambda *a, **k: None,
        wq_b=lambda *a, **k: SimpleNamespace(unflatten=lambda *_: None),
    )
    stub._attend = lambda attn_out, query, kv, sel, cache, step, rope, queries=None: (
        seen.append(step)
    )
    return stub


def _call(step, cache_step, attend_block):
    seen: list = []
    cache = SimpleNamespace(current_step=cache_step)
    attend_block(_attention_stub(seen), None, None, None, cache, step, None)
    assert len(seen) == 1
    return seen[0]


def test_core_attends_the_step_it_was_given():
    from atom.models.deepseek_v41.attention import Attention

    tail, whole = SimpleNamespace(width=128), SimpleNamespace(width=1024)
    assert _call(tail, whole, Attention._attend_block) is tail, (
        "core substituted the cache's Step for the caller's; bounded replay's "
        "tail rows would attend under the whole forward's Step"
    )


def test_the_plugin_wrapper_substitutes_the_live_step():
    """The capture binds this call's arguments once, so a replay's `step`
    argument is the one recorded. Reading the cache inside the region -- which
    does rerun -- is what makes a replay see its own step."""
    pytest.importorskip("vllm")
    from atom.models.deepseek_v41.attention import Attention
    from atom.plugin.vllm.v41_core_overrides import apply_v41_eager_break

    # Save the *values*, not just whether the names exist: core defines
    # `_attn_out_buffer` too (returning None), so a presence check would leave
    # the plugin's allocating override installed for every later test in the
    # session. That is exactly what it did -- `test_runtime_contract` then saw
    # a buffer where core promises none, and only in a full run.
    saved = {
        name: vars(Attention).get(name, _ABSENT)
        for name in ("_attend_block", "_attn_out_buffer")
    }
    try:
        with patch.dict("os.environ", {"VLLM_USE_BREAKABLE_CUDAGRAPH": "1"}):
            if not apply_v41_eager_break():
                pytest.skip("this vLLM has no breakable-capture decorator")
        frozen, live = SimpleNamespace(width=8), SimpleNamespace(width=4)
        assert _call(frozen, live, Attention._attend_block) is live
    finally:
        for name, value in saved.items():
            if value is _ABSENT:
                if name in vars(Attention):
                    delattr(Attention, name)
            else:
                setattr(Attention, name, value)
