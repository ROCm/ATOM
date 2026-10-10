# SPDX-License-Identifier: MIT
"""What vLLM's breakable CUDA graph means for code that forks a stream.

Lives here, not in `atom/utils/forward_context.py`, because every line of it
is a statement about vLLM: the class it reads, the error it exists to avoid
and the window it describes are all that frontend's. ATOM core asks the
question through `atom.plugin`, the way `model_ops.moe` and
`model_ops.base_attention` already reach a frontend's own code.
"""


def capture_breaks_mid_forward() -> bool:
    """Whether the active frontend may end a capture segment mid-forward.

    vLLM's breakable CUDA graph splits one forward into several graphs, ending
    a segment at each eager break. A side stream forked earlier in that forward
    and joined later is outstanding at those breaks, and `capture_end()`
    refuses it:

        HIP error: capturing stream has unjoined work (hipErrorStreamCaptureUnjoined)

    So a fork whose join is not in the same segment must not be opened while
    this is true. False everywhere else, including under a FULL graph, which
    records the forward whole and never breaks it.
    """
    try:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    except ImportError:
        return False
    return BreakableCUDAGraphCapture.is_active()


def breakable_capture_enabled() -> bool:
    """Whether this run may capture graphs around a model's step work at all.

    Asked before a capture starts, where `capture_breaks_mid_forward` asks
    whether one is recording right now. A builder reads it to decide whether
    it has to keep one dummy cache alive across buckets, so it has to answer
    for the whole run rather than for this instant.
    """
    try:
        from vllm.compilation.breakable_cudagraph import is_breakable_cudagraph_enabled
    except ImportError:
        return False
    return bool(is_breakable_cudagraph_enabled())
