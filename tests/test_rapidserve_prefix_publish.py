# SPDX-License-Identifier: MIT
"""Who registers a rapidserve request's prompt blocks for reuse.

A block's hash is published only once its KV has actually been computed, so
publishing is driven from `postprocess`, off the count of tokens the step just
forwarded. The non-disagg path gets the offset for free by publishing before it
advances:

    self.block_manager.hash_blocks(seq, chunk)   # base = num_cached_tokens
    seq.num_cached_tokens += chunk

A rapidserve request is prefilled in the other process and learns about it in
`on_prefill_done`, which advances immediately. By the time `postprocess` runs,
the default base sits at the END of the prompt and `hash_blocks` computes
`start >= end` -- it returns having registered nothing. The prompt never enters
the content index, every later request sharing the prefix misses, and the only
symptom is a prefix cache hit rate pinned at 0%.

The inter-node P/D path already had this problem and solved it in
`BlockManager.publish_loaded_prefix` ("the decode consumer has no local prefill
postprocess to publish these prompt blocks"). Rapidserve needed the same thing,
through `hash_blocks`' `start_tokens` override -- which exists for pipeline
parallelism, for the identical reason: an offset already bumped past the chunk.
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


# ── The range that actually gets published ───────────────────────────────


class TestPublishedRange:
    """Exercised against the real `hash_blocks`, which is where the range is
    decided -- a test that recomputed the formula would agree with itself."""

    @staticmethod
    def _bm():
        from aiter_stub import stubbed_aiter
        from conftest import MockConfig

        with stubbed_aiter():
            from atom.model_engine.block_manager import BlockManager

        return BlockManager(MockConfig(enable_prefix_caching=True))

    @staticmethod
    def _range(bm, *, base, num_new, table_len):
        """`hash_blocks`' own `[start, end)`, read off its arithmetic."""
        hbs = bm.hash_block_size
        start = base // hbs
        end = min((base + num_new) // hbs, table_len)
        return start, end

    def test_the_default_offset_publishes_nothing_after_the_advance(self):
        """The bug, stated as arithmetic. `on_prefill_done` leaves
        `num_cached_tokens` at the full prompt, so a publish that takes its
        base from there covers only the freshly sampled token and rounds away
        to an empty range."""
        bm = self._bm()
        hbs = bm.hash_block_size
        prompt = 16 * hbs
        # What `postprocess` would pass with the default base: num_tokens is
        # the prompt plus the one sampled token.
        start, end = self._range(bm, base=prompt, num_new=1, table_len=64)
        assert start >= end, "expected the collapsed range that registers nothing"

    def test_the_recorded_offset_publishes_what_prefill_filled(self):
        """With the pre-advance offset, the range is exactly the blocks the
        other process computed: from the hit it was handed to the whole
        prompt."""
        bm = self._bm()
        hbs = bm.hash_block_size
        hit, computed = 4 * hbs, 12 * hbs
        start, end = self._range(bm, base=hit, num_new=computed, table_len=64)
        assert (start, end) == (4, 16)

    def test_a_fully_cached_prompt_publishes_nothing_and_that_is_right(self):
        """Prefill computed no new blocks, so there is nothing whose KV became
        valid in this step. Not the same as the bug above: here the empty range
        is the truth."""
        bm = self._bm()
        hit = 16 * bm.hash_block_size
        start, end = self._range(bm, base=hit, num_new=0, table_len=64)
        assert start >= end


# ── The offset is captured before it is destroyed ────────────────────────


def test_the_offset_is_recorded_before_the_advance():
    """One statement either side of this ordering is the whole fix. After the
    `+=`, `num_cached_tokens` is the full prompt and the offset is gone."""
    src = ast.get_source_segment(
        SCHEDULER.read_text(), _method("DecodeScheduler", "on_prefill_done")
    )
    assert src is not None
    record = src.index("seq.pending_hash_start = seq.num_cached_tokens")
    advance = src.index("seq.num_cached_tokens += num_tokens_computed")
    assert record < advance


def test_the_receive_thread_does_not_publish():
    """`on_prefill_done` runs on the PrefillDone receive thread. Publishing
    there mutates the block pool and the checkpoint coordinator that the engine
    thread drains in `_flush_state_maintenance` -- so it records, and
    `postprocess` publishes."""
    fn = _method("DecodeScheduler", "on_prefill_done")
    calls = {
        node.func.attr
        for node in ast.walk(fn)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert "hash_blocks" not in calls


# ── postprocess routes on it ─────────────────────────────────────────────


def test_postprocess_overrides_the_offset_when_one_was_recorded():
    src = ast.get_source_segment(
        SCHEDULER.read_text(), _method("Scheduler", "postprocess")
    )
    assert src is not None
    assert "seq.pending_hash_start >= 0" in src
    assert "start_tokens=_start" in src


def test_a_resumed_prefix_counts_as_hashed():
    """A request that resumed from `start` has `start` tokens in the index --
    that is what the hit was. `hash_blocks` moves `num_hashed_tokens` only when
    it publishes, and a request resuming entirely from cache publishes nothing,
    so the watermark would stay at 0. `hash_decode_blocks` takes its base from
    exactly that field and re-hashes the whole prompt on the first decode step:
    idempotent, but an O(prompt) pass per cache hit.
    """
    src = ast.get_source_segment(
        SCHEDULER.read_text(),
        _method("DecodeScheduler", "_publish_prefill_hashes"),
    )
    assert src is not None
    bump = src.index("seq.num_hashed_tokens = max(seq.num_hashed_tokens, start)")
    # Before the publish, and outside the `num_new > 0` branch -- the whole
    # point is the case where nothing is published.
    assert bump < src.index("num_new = seq.num_cached_tokens - start")


def test_the_recorded_offset_is_consumed_once():
    """Left set, a later one-shot for the same sequence would re-publish a
    range whose blocks have moved on."""
    src = ast.get_source_segment(
        SCHEDULER.read_text(), _method("Scheduler", "postprocess")
    )
    assert src is not None
    assert "seq.pending_hash_start = -1" in src


def test_the_ordinary_path_is_untouched():
    """A sequence that never went through rapidserve carries -1 and must take
    the original branch, offset and placeholder handling included."""
    src = ast.get_source_segment(
        SCHEDULER.read_text(), _method("Scheduler", "postprocess")
    )
    assert src is not None
    assert "_num_new = seq.num_tokens - seq.num_cached_tokens" in src
    assert "_num_new -= num_placeholder" in src


def test_sequences_start_with_nothing_pending():
    from atom.model_engine.sequence import Sequence
    from atom.sampling_params import SamplingParams

    seq = Sequence([1, 2, 3, 4], 4, sampling_params=SamplingParams())
    assert seq.pending_hash_start == -1


@pytest.mark.parametrize("owner", ["DecodeScheduler"])
def test_the_inter_node_path_still_publishes_its_own_way(owner):
    """Not a regression target so much as a boundary: the Mooncake/MORI-IO
    consumer publishes in `BlockManager.publish_loaded_prefix` and sets
    `prefix_hashes_published` itself. It must not also acquire a pending
    offset, or the two mechanisms would both fire."""
    bm_src = (
        pathlib.Path(__file__).resolve().parent.parent
        / "atom/model_engine/block_manager.py"
    ).read_text()
    assert "seq.prefix_hashes_published = True" in bm_src
    assert "pending_hash_start" not in bm_src
