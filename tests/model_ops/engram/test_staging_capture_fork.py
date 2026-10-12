# SPDX-License-Identifier: MIT
"""`EngramStaging` forks a side stream, and `_forks` is what decides.

`EngramStaging.start` opens a side stream whose joins are per layer and lazy --
`consume` waits on a layer's event where that layer needs its rows -- so the
fork spans the whole forward.  vLLM's breakable CUDA graph ends a capture
segment at every attention break, and `capture_end()` refuses a segment that
still has work on a forked stream:

    HIP error: capturing stream has unjoined work (hipErrorStreamCaptureUnjoined)

The fix for that is NOT to stop forking -- the vLLM path keeps the fork and
issues the two orderings inside eager breaks (`VllmEngramStaging`), the way
upstream's own V4.1 does.  `_forks` stays as the seam for a frontend that can
do neither, and because `consume` and `join` must not order against a stream
that was never forked.

Both directions are asserted.  Without the negative control a staging that had
simply stopped forking altogether would pass, and the overlap this buys would
be gone with nothing to say so.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from atom.model_ops.engram.device import staging as staging_mod


def _call_start(forks: bool):
    """Run `start` with no layers, reporting which stream it issued on."""
    compute = object()
    side = SimpleNamespace(waited=[])
    side.wait_stream = side.waited.append

    self = SimpleNamespace(
        host=SimpleNamespace(device="cuda:0", layer_ids=[]),
        stream=side,
        _snapshot=lambda width: None,
        _advance_cursor=lambda width: None,
        _forks=lambda: forks,
    )
    # The stream decision moved into `_prefetch` so a frontend can wrap
    # exactly it in a capture break; bind the real one so this still goes in
    # through `start`, the entry the engine actually calls.
    self._prefetch = lambda width: staging_mod.EngramStaging._prefetch(self, width)
    issued = []

    def fake_stream(stream):
        issued.append(stream)
        return nullcontext()

    with (
        patch.object(torch.cuda, "current_stream", lambda device=None: compute),
        patch.object(torch.cuda, "stream", fake_stream),
    ):
        staging_mod.EngramStaging.start(self, 4)
    assert len(issued) == 1
    return compute, side, issued[0], self


def test_no_fork_when_the_seam_says_not_to():
    compute, side, issued, self_ = _call_start(forks=False)
    assert issued is compute, "staging issued off the compute stream"
    assert side.waited == [], "staging forked while the capture can break"
    # `consume` and `join` read this; ordering against a stream that was never
    # forked is refused under a breakable capture, not merely redundant.
    assert self_.forked is False


def test_forks_when_the_seam_says_to():
    compute, side, issued, self_ = _call_start(forks=True)
    assert issued is side, "staging stopped overlapping the prefetch"
    assert side.waited == [compute], "the fork did not order on its parent"
    assert self_.forked is True


def test_core_forks_by_default():
    """ATOM's own engine records one graph per forward and never breaks, so
    the default has to be the overlapping one -- a frontend that needs the
    other says so by overriding."""
    assert staging_mod.EngramStaging._forks(object()) is True


def test_the_staging_class_is_the_core_one_until_a_frontend_replaces_it():
    assert staging_mod.engram_staging_cls() is staging_mod.EngramStaging
    sentinel = type("Sentinel", (staging_mod.EngramStaging,), {})
    try:
        staging_mod.set_engram_staging_cls(sentinel)
        assert staging_mod.engram_staging_cls() is sentinel
    finally:
        staging_mod.set_engram_staging_cls(None)
    assert staging_mod.engram_staging_cls() is staging_mod.EngramStaging


def test_the_vllm_staging_keeps_the_fork_and_moves_the_waits():
    """The whole point of the port: overlap is kept, the orderings move.

    Upstream's V4.1 decorates `_start_prefetch` and `_finish_prefetch` rather
    than issuing the gather on the compute stream, and this follows it. If the
    subclass ever went back to answering `_forks` False, the gather would
    serialize against the layers again with nothing to say so.
    """
    from atom.plugin.vllm.v41_engram_staging import VllmEngramStaging

    assert VllmEngramStaging._forks(object()) is True
    assert issubclass(VllmEngramStaging, staging_mod.EngramStaging)
