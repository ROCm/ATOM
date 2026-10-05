# SPDX-License-Identifier: MIT
"""Who charges a rapidserve request for its prompt.

A request's prefix-cache reuse is found by whichever BlockManager admits it. In
a rapidserve pair that is decode's, always: the prefill process runs a
`PrefillScheduler` with `block_manager = None` and is handed the hit count in
`BlockAssignment.num_cached_tokens` rather than discovering it.

So decode does the lookup, decode does the accounting, and prefill does neither
-- which is the same split `_METRICS_ROLE` already encodes, where the
aggregator drops the prefill rank's counts so an in-flight request is not
counted on both sides.

The bug this guards: decode found the hits and skipped the work, and nothing
called `update_cache`, so the engine line reported `n/a` forever. `n/a` is
"never measured", and it is indistinguishable at a glance from a cache that is
working and unmeasured -- which is exactly what it was.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

SCHEDULER = (
    pathlib.Path(__file__).resolve().parent.parent / "atom/model_engine/scheduler.py"
)


def _method(owner: str, name: str) -> ast.FunctionDef:
    tree = ast.parse(SCHEDULER.read_text(), filename=str(SCHEDULER))
    found = [
        fn
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef) and cls.name == owner
        for fn in cls.body
        if isinstance(fn, ast.FunctionDef) and fn.name == name
    ]
    assert len(found) == 1, f"{owner}.{name} is defined {len(found)} times"
    return found[0]


def _calls(fn: ast.FunctionDef) -> set[str]:
    return {
        node.func.attr
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }


# ── Decode accounts what decode looked up ────────────────────────────────


def test_decode_admission_accounts_the_reuse_it_found():
    """`allocate_waiting` is the only place a rapidserve request is admitted,
    and `block_manager.allocate` there is the only place its hit count is
    computed. If the accounting is not in that method it happens nowhere."""
    assert "_record_cache_reuse" in _calls(
        _method("DecodeScheduler", "allocate_waiting")
    )


def test_it_is_charged_only_on_a_successful_allocation():
    """Counted after a refusal, a request that was never admitted enters the
    window as zero reuse and drags the windowed rate toward 0%. The call has to
    sit under the `allocate` that filled the fields it reads, above the
    `popleft` every exit path shares."""
    src = ast.get_source_segment(
        SCHEDULER.read_text(), _method("DecodeScheduler", "allocate_waiting")
    )
    assert src is not None
    record = src.index("self._record_cache_reuse(seq)")
    popleft = src.index("self.waiting.popleft()")
    assert record < popleft, (
        "_record_cache_reuse must precede the popleft, inside the branch where "
        "block_manager.allocate() succeeded"
    )


def test_the_fields_it_reads_are_filled_by_the_allocate_above_it():
    """Ordering that is easy to break by moving the call 'somewhere tidier':
    `num_cached_tokens` and the hit-block counts are set by
    `block_manager.allocate`, so accounting before it reads the previous
    request's numbers."""
    src = ast.get_source_segment(
        SCHEDULER.read_text(), _method("DecodeScheduler", "allocate_waiting")
    )
    assert src is not None
    assert src.index("self.block_manager.allocate(") < src.index(
        "self._record_cache_reuse(seq)"
    )


def test_the_hit_count_reaches_allocate():
    """`allocate` defaults `num_cached_blocks` to 0 -- "0 if caller didn't call
    it" -- so comparing `can_allocate`'s result to `< 0` and dropping the value
    tells it every block is fresh. It then claims the whole prompt and records
    `num_cached_tokens = 0`, and no request reuses anything however full the
    cache is. The ordinary path passes it; this one has to as well."""
    src = ast.get_source_segment(
        SCHEDULER.read_text(), _method("DecodeScheduler", "allocate_waiting")
    )
    assert src is not None
    assert "self.block_manager.allocate(seq, num_cached_blocks)" in src, (
        "DecodeScheduler.allocate_waiting must forward can_allocate's hit "
        "count to allocate, not just test its sign"
    )


def test_both_allocate_callers_pass_the_hit_count():
    """Stated across the pair, because the bug was the two disagreeing: the
    ordinary scheduler passed it and this one did not, and nothing compared
    them."""
    text = SCHEDULER.read_text()
    bare = [
        node
        for node in ast.walk(ast.parse(text))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "allocate"
        and isinstance(node.func.value, ast.Attribute)
        and node.func.value.attr == "block_manager"
        and len(node.args) < 2
    ]
    assert not bare, (
        f"{len(bare)} call(s) to block_manager.allocate() omit the hit count "
        "and so silently allocate every block fresh"
    )


# ── ...and prefill accounts nothing ──────────────────────────────────────


def test_the_prefill_side_does_not_double_count():
    """It could not do the lookup, and must not claim the result. Both halves
    matter: `PrefillScheduler` has no BlockManager to read a hit off, and the
    aggregator already folds the pair's counts together."""
    tree = ast.parse(SCHEDULER.read_text(), filename=str(SCHEDULER))
    cls = next(
        c
        for c in ast.walk(tree)
        if isinstance(c, ast.ClassDef) and c.name == "PrefillScheduler"
    )
    assert "_record_cache_reuse" not in _calls(cls), (
        "the rapidserve prefill process must not account cache reuse; its "
        "decode peer already did"
    )


def test_prefill_still_declares_itself_the_prefill_role():
    """The other half of the no-double-count rule, and the one the metrics
    aggregator reads."""
    src = SCHEDULER.read_text()
    cls = next(
        c
        for c in ast.walk(ast.parse(src))
        if isinstance(c, ast.ClassDef) and c.name == "PrefillScheduler"
    )
    roles = [
        n.value.value
        for n in cls.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_METRICS_ROLE" for t in n.targets)
        and isinstance(n.value, ast.Constant)
    ]
    assert roles == ["prefill"]


# ── The rate is None until something is measured ─────────────────────────


class TestUnmeasuredIsNotZero:
    @staticmethod
    def _stats(**kw):
        mod = pytest.importorskip("atom.model_engine.engine_stats")
        return mod.EngineStats(enable_prefix_caching=True, **kw)

    def test_an_engine_that_never_accounted_reports_none(self):
        """Which the line renders `n/a`. Reporting 0.0% instead would be a
        claim about reuse that nobody measured."""
        assert self._stats().recent_cache_hit_rate is None

    def test_one_admission_is_enough_to_make_it_a_number(self):
        """And so the fix is observable: a single accounted admission moves the
        rate off `n/a`, whatever its value."""
        stats = self._stats()
        stats.update_cache(512, 1024, 0, 512, 896)
        assert stats.recent_cache_hit_rate is not None
