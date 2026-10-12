"""vLLM's synthetic warmup is exempt from the stale-cursor contract; nothing else is.

The contract is the one thing that catches a prefix served from somebody
else's state, so an exemption that widened by accident would be worse than
the crash it removes. Both directions are asserted, and the negative one is
the load-bearing half.
"""

import numpy as np

from atom.model_ops.attentions.deepseek_v41 import cache as v41_cache


class _Cache:
    _report_stale_state = v41_cache.PagedAttentionCache._report_stale_state


def _check(ids, starts, cursors):
    arr = np.asarray([[c, 0] for c in cursors], dtype=np.int64)
    _Cache()._report_stale_state(arr, np.asarray(starts, dtype=np.int64), ids)


def test_a_real_request_with_a_stale_cursor_still_raises():
    try:
        _check(["chatcmpl-abc"], [2], [0])
    except ValueError as exc:
        assert "needs state at 2" in str(exc)
    else:
        raise AssertionError("the contract did not fire for a real request")


def test_vllms_warmup_request_is_let_through():
    before = v41_cache.warmup_exemptions()
    _check(["_warmup_0_"], [2], [0])
    assert v41_cache.warmup_exemptions() == before + 1


def test_a_warmup_row_does_not_cover_a_real_one_beside_it():
    """The exemption is per-row, not per-batch."""
    try:
        _check(["_warmup_0_", "chatcmpl-abc"], [2, 2], [0, 0])
    except ValueError as exc:
        assert "chatcmpl-abc" in str(exc)
    else:
        raise AssertionError("a warmup row silenced a real one")


def test_a_name_that_merely_mentions_warmup_is_not_exempt():
    assert not v41_cache._is_vllm_warmup_request("chatcmpl-_warmup_0_-tail")
    assert not v41_cache._is_vllm_warmup_request("warmup_0_")
    assert v41_cache._is_vllm_warmup_request("_warmup_12_")
