# SPDX-License-Identifier: MIT
"""The staging prefetch must not fork while the frontend may break a capture.

`EngramStaging.start` opens a side stream whose joins are per layer and lazy --
`consume` waits on a layer's event where that layer needs its rows -- so the
fork spans the whole forward.  vLLM's breakable CUDA graph ends a capture
segment at every attention break, and `capture_end()` refuses a segment that
still has work on a forked stream:

    HIP error: capturing stream has unjoined work (hipErrorStreamCaptureUnjoined)

Both directions are asserted.  Without the negative control a staging that had
simply stopped forking altogether would pass, and the overlap this buys on
every other path would be gone with nothing to say so.
"""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import torch

from atom.model_ops.engram.device import staging as staging_mod


def _call_start(breaks: bool):
    """Run `start` with no layers, reporting which stream it issued on."""
    compute = object()
    side = SimpleNamespace(waited=[])
    side.wait_stream = side.waited.append

    self = SimpleNamespace(
        host=SimpleNamespace(device="cuda:0", layer_ids=[]),
        stream=side,
        _snapshot=lambda width: None,
        _advance_cursor=lambda width: None,
    )
    issued = []

    def fake_stream(stream):
        issued.append(stream)
        return nullcontext()

    with (
        patch.object(staging_mod, "capture_breaks_mid_forward", lambda: breaks),
        patch.object(torch.cuda, "current_stream", lambda device=None: compute),
        patch.object(torch.cuda, "stream", fake_stream),
    ):
        staging_mod.EngramStaging.start(self, 4)
    assert len(issued) == 1
    return compute, side, issued[0]


def test_no_fork_when_the_capture_can_break():
    compute, side, issued = _call_start(breaks=True)
    assert issued is compute, "staging issued off the compute stream"
    assert side.waited == [], "staging forked while the capture can break"


def test_still_forks_when_the_capture_cannot_break():
    compute, side, issued = _call_start(breaks=False)
    assert issued is side, "staging stopped overlapping the prefetch"
    assert side.waited == [compute], "the fork did not order on its parent"
