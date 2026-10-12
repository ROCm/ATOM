# SPDX-License-Identifier: MIT
"""Keep Engram's prefetch overlapped under a capture that breaks mid-forward.

Core forks a side stream for the Engram gather and joins it per layer, lazily
-- `consume` waits on that layer's event where the layer needs its rows. The
fork therefore spans the forward, which collides with vLLM's breakable CUDA
graph twice:

    hipErrorStreamCaptureUnjoined    a segment ended with work still on the
                                     forked stream
    hipErrorStreamCaptureIsolation   a captured stream waited on an event
                                     recorded off-capture

The answer is not to stop forking. vLLM's own DeepSeek-V4.1 keeps its side
stream and puts the orderings in eager breaks instead -- `_start_prefetch` and
`_finish_prefetch` in `vllm/models/deepseek_v41/nvidia/engram.py` are both
decorated, under the comment "Eager boundaries let the lookup span piecewise
graph segments." A break ends the current segment, runs the call eagerly on
the capture stream and starts a fresh one, so the wait is issued where no
segment owns it and the lookup keeps running across the ones that follow.

This path needs only the two waits decorated. The fork itself is already
outside the forward here: the metadata builder stages the step before the
model is called (`_v41_stage_outside_forward`), so `start` never runs inside a
captured segment. What was left crossing the boundary is the per-layer wait in
`consume`, which the layers do run inside one.

An earlier version of this file stood the fork down instead, issuing the
gather on the compute stream. That was correct and cost the overlap between a
largely-UVA gather and the layers consuming it -- a cost upstream does not
pay, and one this had no measurement for.
"""

import logging

from atom.model_ops.engram.device.staging import (
    EngramStaging,
    set_engram_staging_cls,
)

logger = logging.getLogger("atom")


def _eager_break(fn):
    """vLLM's capture break, or `fn` unchanged where there is none."""
    try:
        from vllm.compilation.breakable_cudagraph import eager_break_during_capture
    except ImportError:
        return fn
    return eager_break_during_capture(fn)


class VllmEngramStaging(EngramStaging):
    """`EngramStaging` whose stream orderings sit in eager breaks."""

    # Both write nothing and return nothing -- they only order streams -- so
    # the decorator's in-place-output requirement is met trivially, and a
    # replay re-issuing the same wait against the same long-lived event is
    # what the ordering needs anyway.
    # The fork too, not only the waits. An earlier version decorated the
    # waits alone, on the reasoning that this path always forks outside the
    # forward -- the metadata builder stages the step before the model is
    # called. It does not always: when the proxy pool is not bound yet, which
    # is the case for the dummy batches a capture runs, `v41_stage_step`
    # stages from inside the forward, so the fork opens inside a segment and
    # the segment ends with it open. Measured, that reasoning cost a boot:
    # 8 x hipErrorStreamCaptureUnjoined. Upstream decorates its own
    # `_start_prefetch` for the same reason.
    _prefetch = _eager_break(EngramStaging._prefetch)
    _await_layer = _eager_break(EngramStaging._await_layer)
    _await_fork = _eager_break(EngramStaging._await_fork)


def install_vllm_engram_staging() -> None:
    """Build `VllmEngramStaging` from here on. Idempotent."""
    set_engram_staging_cls(VllmEngramStaging)
