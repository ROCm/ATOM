# SPDX-License-Identifier: MIT
"""Mark V4.1's attention as a capture break, and give it a buffer to fill.

Both halves are this frontend's, and neither changes anything for ATOM's own
engine: core allocates nothing, copies nothing and hands the attention rows
back by return value, exactly as it did before any of this.

`eager_break_during_capture` is vLLM's: it ends a capture segment at the
decorated call and resumes a new one after it. ATOM's own engine records one
graph per forward and never breaks, so the decorator is the identity there and
the model carries an `import vllm` for nobody. Applying it from here keeps the
model free of that import and leaves native untouched.

Why V4.1 needs the break at all: attention's kernels are launched with
per-step host values -- the batch's longest KV extent among them -- which a
capture freezes at whatever length it recorded while every later decode step
grows past it. Ending the segment around attention is what lets the rest of
the forward be captured while attention itself runs eagerly each step.

The region is wide on purpose. Upstream keeps a narrow one for model runner V2
and a wide one for V1, because V1's piecewise capture produces garbage with
the narrow region (vllm #51430): the attention input preparation stays inside
the graph and the replay reuses it. Wide is correct under both; narrow is an
optimisation under one, and picking by runner would be a config-dependent
choice whose wrong side is silent.

Applied to the class, before any model is built. `Attention._attend_block` is
looked up on the instance at call time, so patching the attribute reaches
every layer of every V4.1 model this process goes on to construct.
"""

import logging

logger = logging.getLogger("atom")

_MARKER = "_atom_v41_eager_break_applied"


def _attn_out_buffer(self, hidden):
    """The buffer the broken-out attention region writes its rows into.

    A region ended by a capture break runs eagerly on every replay, so a fresh
    tensor returned from it would hand back a different address each time
    while the captured segments after it still point at the first. Allocating
    here -- in what stays inside the graph -- gives the capture one address to
    record and every replay the same one to find.

    Every leading dimension of `hidden`, not just the first: ATOM hands this
    layer `[batch, tokens, dim]` where upstream's native V4.1 gets
    `[tokens, dim]`, so `hidden.shape[0]` is the batch and sizing from it
    yields one row per sequence instead of one per token. The rest of this
    path flattens the leading pair, which turns that into a kernel `out=`
    mismatch rather than a wrong answer -- loud, but only because the kernel
    checks.
    """
    import torch

    return torch.empty(
        (*hidden.shape[:-1], self.heads, self.head_dim),
        dtype=hidden.dtype,
        device=hidden.device,
    )


def apply_v41_eager_break() -> bool:
    """Wrap `Attention._attend_block` in vLLM's capture break. Idempotent."""
    try:
        from vllm.compilation.breakable_cudagraph import eager_break_during_capture
    except ImportError:
        return False

    from atom.models.deepseek_v41.attention import Attention

    original = Attention._attend_block
    if getattr(original, _MARKER, False):
        return False

    wrapped = eager_break_during_capture(original)
    if wrapped is original:
        # The decorator reads `VLLM_USE_BREAKABLE_CUDAGRAPH` when it is
        # applied, and hands the function straight back when breaks are off.
        # Marking it then would brand the undecorated function as done and
        # make a later call -- with the env set -- skip the work.
        return False
    setattr(wrapped, _MARKER, True)
    Attention._attend_block = wrapped
    # The break and the buffer go together: the decorator's contract is that
    # the region writes through a caller-owned tensor, and core hands out none
    # until a frontend says it needs one.
    Attention._attn_out_buffer = _attn_out_buffer
    logger.info(
        "ATOM plugin: DeepSeek-V4.1 attention runs in an eager break, so a "
        "capture does not freeze its per-step launch bounds."
    )
    return True
