# SPDX-License-Identifier: MIT
"""What the rapidserve kvcache bootstrap bundle has to carry, and why.

Decode owns no KV memory, so its `get_num_blocks()` short-circuits and it never
sizes a pool. Every number its `BlockManager` is built from therefore has to
ride the bundle from prefill. That transport exists and works; what broke it
once was a rename on the far end -- `num_per_req_cache_groups` and friends
became the per-class entry table, the bundle kept shipping the old names, and
both ends defaulted the misses to 0. Decode then came up believing the model
had no per-request state, which is not an error anywhere: it reads as a
stateless model and refuses admission much later, with "Cannot allocate
prefill" on a pool that is entirely healthy.

So the keys are checked against what sizing actually returns, rather than
against a second copy of the list.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

RUNNER = (
    pathlib.Path(__file__).resolve().parent.parent / "atom/model_engine/model_runner.py"
)
CORE = (
    pathlib.Path(__file__).resolve().parent.parent / "atom/model_engine/engine_core.py"
)


def _method(path: pathlib.Path, owner: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(), filename=str(path))
    found = [
        fn
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef) and cls.name == owner
        for fn in cls.body
        if isinstance(fn, ast.FunctionDef) and fn.name == name
    ]
    assert len(found) == 1, f"{owner}.{name} is defined {len(found)} times"
    return found[0]


def _sched_dim_keys() -> tuple[str, ...]:
    from aiter_stub import stubbed_aiter

    with stubbed_aiter():
        from atom.model_engine.engine_core import _SCHED_DIM_KEYS

    return _SCHED_DIM_KEYS


def _returned_keys() -> set[str]:
    """Keys of the dict `ModelRunner.get_num_blocks` returns.

    Read statically: the module imports aiter, and the regression this guards
    lands on CI, which has no aiter build.
    """
    fn = _method(RUNNER, "ModelRunner", "get_num_blocks")
    returns = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
    ]
    assert len(returns) == 1, "get_num_blocks should return one dict literal"
    return {
        k.value
        for k in returns[0].value.keys
        if isinstance(k, ast.Constant) and isinstance(k.value, str)
    }


# ── The keys name real sizing output ─────────────────────────────────────


def test_every_shipped_key_is_something_sizing_produces():
    """The invariant the old names lost. `get_num_blocks` is the only place
    these numbers exist, so a key that is not in its result cannot be filled by
    anyone and will ship whatever the default is."""
    missing = sorted(set(_sched_dim_keys()) - _returned_keys())
    assert not missing, (
        f"_SCHED_DIM_KEYS names {missing}, which ModelRunner.get_num_blocks "
        "does not return. Nothing would fill them on the prefill side."
    )


def test_the_state_slot_count_is_among_them():
    """Named rather than implied by the test above: `pool_entries` is what
    `BlockManager` reads `num_state_slots` out of, so it is the one key whose
    absence turns a stateful model into a silently stateless one."""
    assert "pool_entries" in _sched_dim_keys()


def test_the_plan_travels_whole_and_not_only_in_pieces():
    """`BlockManager` reads the two dicts; a *backend* reads the plan.

    DeepSeek-V4's `num_state_slots` is
    `model_runner.pool_plan.entries[STATE_SLOT_CLASS]`, so a decode runner left
    holding `PoolPlan.empty()` sizes the class at zero however complete the
    dicts beside it are -- it builds an arena with no slot views and dies in
    warmup describing a copy with no source rows. Shipping the pieces is not
    shipping the plan.
    """
    assert "pool_plan" in _sched_dim_keys()
    assert "pool_plan" in _returned_keys()


def test_the_backend_count_and_the_manager_count_have_one_source():
    """Both spellings must come off the same object. Two independently shipped
    numbers is how they disagree, and a disagreement here is a pool built at
    one size and addressed at another."""
    fn = _method(RUNNER, "RapidServeModelRunner", "import_kv_cache_ipc_handle")
    src = ast.get_source_segment(RUNNER.read_text(), fn)
    assert src is not None
    assert "self.pool_plan = pool_plan" in src
    assert "pool_plan.entries" in src, (
        "config.pool_entries must be derived from the installed plan, not "
        "shipped alongside it"
    )


# ── Neither end may default a miss ───────────────────────────────────────


def test_the_send_side_takes_no_getattr_default():
    """A third argument to `getattr` is what let the rename rot: the keys
    stopped existing, every lookup returned 0, and the bundle looked full."""
    fn = _method(CORE, "PrefillEngineCore", "_send_ready_signal")
    defaulted = [
        node
        for node in ast.walk(fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "getattr"
        and len(node.args) > 2
    ]
    assert not defaulted, (
        "getattr(..., <default>) in the bundle send turns a key that stopped "
        "existing into a silently shipped default"
    )


def test_the_receive_side_raises_on_a_missing_key():
    """Decode's half of the same rule. It cannot recover from a short bundle --
    there is no second source for these numbers -- so it must refuse loudly
    rather than build a BlockManager from an empty table."""
    fn = _method(CORE, "DecodeEngineCore", "__init__")
    # A membership test against the bundle, anywhere in the comprehension that
    # collects what is absent. Matched structurally rather than by message text
    # so rewording the error does not fail this.
    checks_membership = any(
        isinstance(node, ast.Compare)
        and any(isinstance(op, ast.NotIn) for op in node.ops)
        and any(
            isinstance(c, ast.Name) and c.id == "bundle" for c in node.comparators
        )
        for node in ast.walk(fn)
    )
    assert checks_membership, (
        "DecodeEngineCore.__init__ no longer tests the bundle for the keys it "
        "is about to read"
    )
    assert any(isinstance(node, ast.Raise) for node in ast.walk(fn)), (
        "...and no longer raises when they are absent"
    )


# ── The STATE floor is sized for the side that holds the slots ───────────


class TestStateFloorConcurrency:
    """A state slot is held for a request's whole life: decode assigns it,
    prefill writes it, decode reads it. One request never needs two, so the
    floor covers decode's concurrency -- which is not prefill's whenever
    `--disagg-prefill-max-num-seqs` narrows the prefill batch."""

    @staticmethod
    def _fn():
        mr = pytest.importorskip("atom.model_engine.model_runner")
        return mr._state_floor_num_seqs

    @staticmethod
    def _config(max_num_seqs, decode_max=None):
        import types

        return types.SimpleNamespace(
            max_num_seqs=max_num_seqs,
            disagg_decode_max_num_seqs=decode_max,
        )

    def test_plain_launch_uses_its_own_bound(self):
        assert self._fn()(self._config(64)) == 64

    def test_prefill_sizes_for_decodes_concurrency(self):
        """The case that motivated it: prefill batches 8 at a time while decode
        admits 256, and the slots belong to the 256."""
        assert self._fn()(self._config(8, decode_max=256)) == 256

    def test_a_wider_prefill_still_defers_to_decode(self):
        """Not a max(): the number of slots is the number of live requests, and
        decode is what bounds that. A prefill batch wider than decode's bound
        is prefill running ahead through slots decode already owns."""
        assert self._fn()(self._config(512, decode_max=256)) == 256


def test_config_defaults_the_decode_bound_to_unset():
    """None, not 0 -- `or max_num_seqs` has to fall through for every
    non-rapidserve launch, and 0 would too, but only by accident."""
    from atom.config import Config

    field = {f.name: f for f in __import__("dataclasses").fields(Config)}
    assert field["disagg_decode_max_num_seqs"].default is None
