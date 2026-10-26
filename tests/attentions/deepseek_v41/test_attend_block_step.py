# SPDX-License-Identifier: MIT
"""The broken-out attention region reads the step the cache holds now.

A capture binds this call's arguments once, when the graph is recorded, and
reruns no host code to rebind them -- so a `step` passed as an argument is
the one the capture saw: a warmup batch's lengths and cursors, for the life
of that graph. The cache outlives every step and `begin_step` leaves the
current one on it, so reading it from there -- inside the region, which does
rerun -- is what lets a replay see its own step.

That substitution belongs to this frontend alone, and an earlier version put
it in the shared `attention.py`, where it overrode callers that meant the
step they passed. Bounded replay re-enters attention with a *tail* Step while
the cache still holds the whole forward's, so a 1024-token prefill with a
128-token window handed attention 128 rows under a Step describing 1024 --
on the native path, with `enable_decoder_swa_bounded_replay` defaulting to
True. Core now carries no copy of this at all.

Both directions are asserted. Core must not substitute -- it has nothing to
substitute with, which is the stronger statement -- and the plugin's copy
must, because one that simply passed its argument through would pass every
other test here while replaying the warmup's step forever.
"""

from contextlib import suppress
from types import SimpleNamespace

import pytest


def test_core_attention_has_no_step_substitution():
    """`atom/models/deepseek_v41/attention.py` is untouched by this PR, so the
    substitution cannot be reintroduced there without this going red."""
    pytest.importorskip("aiter", reason="the V4.1 attention module reaches AITER")
    import pathlib

    import atom.models.deepseek_v41.attention as core_attention

    source = pathlib.Path(core_attention.__file__).read_text()
    assert "current_step" not in source, (
        "core's attention reads `cache.current_step` again; that overrides "
        "bounded replay's tail Step on the native path"
    )
    assert not hasattr(core_attention.Attention, "_attn_out_buffer")


def test_the_plugin_region_reads_the_live_step():
    pytest.importorskip("vllm")
    from atom.plugin.vllm import v41_core_overrides as mod

    seen = []
    stub = SimpleNamespace(
        heads=1,
        head_dim=1,
        _fork_compress=lambda *a, **k: seen.append(a[2]) or None,
        project_qkv=lambda *a, **k: (None, None),
        qk_norm=lambda *a, **k: (None, None, None),
        _fork_select=lambda *a, **k: None,
        wq_b=lambda *a, **k: SimpleNamespace(unflatten=lambda *_: None),
    )
    frozen, live = SimpleNamespace(width=8), SimpleNamespace(width=4)
    cache = SimpleNamespace(current_step=live)
    # Suppressed rather than required: what is pinned is which step reached
    # the first use, not that the stubbed tensors eventually fail.
    with suppress(Exception):
        mod._attend_block(stub, None, None, None, cache, frozen, None, None)
    assert seen and seen[0] is live, (
        "the region used the step it was handed, so a replay would attend "
        "under the step the capture recorded"
    )


def test_the_plugin_falls_back_when_no_pool_is_bound_yet():
    """`current_step` is absent until `begin_step` has run once -- the window
    the memory-profiling forward lives in."""
    pytest.importorskip("vllm")
    from atom.plugin.vllm import v41_core_overrides as mod

    seen = []
    stub = SimpleNamespace(
        heads=1,
        head_dim=1,
        _fork_compress=lambda *a, **k: seen.append(a[2]) or None,
        project_qkv=lambda *a, **k: (None, None),
        qk_norm=lambda *a, **k: (None, None, None),
        _fork_select=lambda *a, **k: None,
        wq_b=lambda *a, **k: SimpleNamespace(unflatten=lambda *_: None),
    )
    given = SimpleNamespace(width=8)
    with suppress(Exception):
        mod._attend_block(stub, None, None, None, SimpleNamespace(), given, None, None)
    assert seen and seen[0] is given
