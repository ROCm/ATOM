# SPDX-License-Identifier: MIT
"""ATOM's four V4.1 classes, as this frontend needs them.

The metadata builder, the CSA2 cache, attention and the Engram staging each
need one thing here that ATOM's own engine does not, and each gets it the same
way: an override on a subclass, or an attribute set on the class before any
model is built. Nothing in this file changes what native computes -- core
allocates its own pool, frees its own scratch, forks its own stream and hands
attention's rows back by return value, exactly as it did before any of this.

They are collected here because they are the same kind of thing and have the
same reason to exist, which in every case is a property of vLLM rather than of
V4.1: that vLLM owns the KV allocation, that its capture can break a forward
into segments, that it captures synthetic batches, and that a broken-out
region must write through a caller-owned buffer because a fresh tensor changes
address on every replay.
"""

import logging

import torch

from atom.model_ops.attentions.deepseek_v41.backend import DeepseekV41MetadataBuilder
from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.engram.device.staging import EngramStaging, set_engram_staging_cls
from atom.plugin.vllm.vllm_runtime import breakable_capture_enabled

logger = logging.getLogger("atom")


# ---------------------------------------------------------------------------
# The metadata builder: one dummy cache a capture can hold on to
# ---------------------------------------------------------------------------
# Core allocates a dummy batch's scratch cache per call and frees it with the
# call, which is right for ATOM's own engine: its capture entry fabricates a
# batch with `is_dummy_run=False` against the serving cache, so no recorded
# kernel ever holds a scratch address, and its only dummy batches come from
# `warmup_model`, outside any capture.
#
# This path does capture dummy batches. `v41_stage_step` falls back to a
# fabricated dummy batch whenever a step arrives without a snapshot, without a
# slot allocator, or before the proxy pool is bound -- which includes vLLM's
# memory-profiling forward -- and under breakable capture a graph can be
# recording while that happens. A scratch cache freed when the capture returns
# leaves the recorded kernels pointing at freed memory, read back on the first
# replay as an illegal access reported asynchronously somewhere else entirely.
#
# So while capture is possible, hand out one cache, allocated once at the
# capture ceiling and kept alive on the builder. Every bucket's dummy batch
# fits inside it -- its page, slot and token counts are caps and the batch
# indexes from zero -- so one allocation serves them all rather than one per
# bucket (the STATE side alone is ~5 MiB per slot).


class VllmDeepseekV41MetadataBuilder(DeepseekV41MetadataBuilder):
    """The builder ATOM ships, with this path's two answers."""

    # The deferred stale-state verdict is wrong under concurrency here, and
    # wrong in the direction that takes the engine down -- it refuses correct
    # state. `DeepseekV41MetadataBuilder._state_probe_blocking` carries the
    # measurement; this is where this path says it wants the blocking one.
    _state_probe_blocking_default = True

    def _dummy_cache(self, pages, slots, running_tokens):
        if not breakable_capture_enabled():
            return super()._dummy_cache(pages, slots, running_tokens)
        cached = getattr(self, "_capture_dummy_cache", None)
        if cached is None:
            ceiling_pages = max(
                pages, -(-self.max_num_batched_tokens // self.block_size)
            )
            cached = PagedAttentionCache(
                self.geometry,
                ceiling_pages,
                max(slots, self.max_bs),
                self.device,
                max_tokens=max(running_tokens, self.max_num_batched_tokens),
                workspace=self.score_workspace,
            )
            self._capture_dummy_cache = cached
        return cached


# ---------------------------------------------------------------------------
# The cache: pages out of vLLM's own KV allocation
# ---------------------------------------------------------------------------
# ATOM's `PagedAttentionCache` allocates its pool itself. On this path the
# bytes already exist: vLLM's `KVCacheManager` allocated a slab for ATOM's
# proxy attention layer, and allocating a second pool beside it would
# double-count the memory vLLM has already budgeted. So the cache is handed
# that slab instead, through the one seam core leaves for it.
#
# Why the whole slab and not the paged part: V4.1 buys its pool in two
# currencies, and `V41PoolGeometry.window` states a slot's STATE ring as an
# offset past the *absolute* end of the paged region, which the `_indices`
# kernels dereference against a single base. PAGE and STATE therefore live in
# one address space, and vLLM -- which only knows how to buy the first
# currency -- has to be made to buy a slab wide enough for the second behind
# it. `deepseek_v41_state_reserve_patch` is what does that: it withholds the
# tail's worth of block ids from the scheduler while leaving the tensor's size
# alone.


class VllmBackedPagedAttentionCache(PagedAttentionCache):
    """`PagedAttentionCache` over a slab somebody else owns."""

    def __init__(self, *args, backing: torch.Tensor, **kwargs):
        # Set before `super().__init__`, which calls `_allocate_backing` while
        # it runs. An attribute assigned after that call would not exist yet
        # when the override reads it.
        self._external_backing = backing
        super().__init__(*args, **kwargs)

    def _allocate_backing(self, size, device):
        """The slab, trimmed and zeroed to core's contract.

        The arena has to behave exactly like the `torch.zeros` it replaces:
        flat uint8, contiguous, on this cache's device, zeroed, and exactly
        `size` elements. The trim is not an optimization -- every region is a
        `view` with its own strides, and a longer tail does not divide. What
        it drops is the rounding slack of a STATE tail bought in whole pages,
        under one page of a pool that holds tens of thousands.
        """
        backing = self._external_backing
        if not isinstance(backing, torch.Tensor):
            raise TypeError("An external CSA2 backing must be a tensor")
        if backing.dtype != torch.uint8 or backing.dim() != 1:
            raise ValueError("An external CSA2 backing must be flat uint8")
        if not backing.is_contiguous():
            raise ValueError("An external CSA2 backing must be contiguous")
        # Where `torch.zeros(device=device)` would have put it -- an index-less
        # `cuda` means the current device, which is what vLLM hands down in
        # `device_config.device`, and comparing that to a tensor's
        # always-indexed `.device` would never match.
        resolved = torch.empty(0, device=device).device
        if backing.device != resolved:
            raise ValueError(
                f"An external CSA2 backing on {backing.device} cannot back a "
                f"cache on {resolved}"
            )
        if backing.numel() < size:
            raise ValueError(
                f"An external CSA2 backing of {backing.numel()} bytes is short "
                f"of the {size} this geometry needs"
            )
        arena = backing[:size]
        # Core's own allocation is zeroed; the cursor reset and every cold read
        # assume it, and a reused slab is not.
        arena.zero_()
        return arena


# ---------------------------------------------------------------------------
# Attention: a capture break, and a buffer for it to fill
# ---------------------------------------------------------------------------
# `eager_break_during_capture` is vLLM's: it ends a capture segment at the
# decorated call and resumes a new one after it. ATOM's own engine records one
# graph per forward and never breaks, so the decorator is the identity there
# and the model would carry an `import vllm` for nobody. Applying it from here
# keeps the model free of that import and leaves native untouched.
#
# Why V4.1 needs the break at all: attention's kernels are launched with
# per-step host values -- the batch's longest KV extent among them -- which a
# capture freezes at whatever length it recorded while every later decode step
# grows past it. Ending the segment around attention is what lets the rest of
# the forward be captured while attention itself runs eagerly each step.
#
# The region is wide on purpose. Upstream keeps a narrow one for model runner
# V2 and a wide one for V1, because V1's piecewise capture produces garbage
# with the narrow region (vllm #51430): the attention input preparation stays
# inside the graph and the replay reuses it. Wide is correct under both;
# narrow is an optimisation under one, and picking by runner would be a
# config-dependent choice whose wrong side is silent.
#
# Applied to the class, before any model is built. `Attention._attend_block`
# is looked up on the instance at call time, so patching the attribute reaches
# every layer of every V4.1 model this process goes on to construct.

_EAGER_BREAK_MARKER = "_atom_v41_eager_break_applied"


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
    if getattr(original, _EAGER_BREAK_MARKER, False):
        return False

    def live_step_attend_block(
        self, attn_out, hidden, hidden_scale, cache, step, rope, queries=None
    ):
        """`_attend_block`, reading the step the cache holds right now.

        The decorator binds this call's arguments once, when the graph is
        recorded, and reruns no host code to rebind them -- so the `step` a
        replay receives is the one the capture saw: a warmup batch's lengths
        and cursors, for the life of that graph. The cache outlives every
        step and `begin_step` leaves the current one on it, so reading it
        here -- inside the region, which does rerun -- is what makes a replay
        see its own step.

        Here and not in core, because core has callers that mean the step
        they pass: bounded replay re-enters attention with a tail Step while
        the cache still holds the whole forward's, and substituting there
        gave the tail's rows the full forward's Step. This path refuses
        bounded replay (`model_wrapper`), so the two cannot collide.
        """
        return original(
            self,
            attn_out,
            hidden,
            hidden_scale,
            cache,
            getattr(cache, "current_step", None) or step,
            rope,
            queries,
        )

    wrapped = eager_break_during_capture(live_step_attend_block)
    if wrapped is live_step_attend_block:
        # The decorator reads `VLLM_USE_BREAKABLE_CUDAGRAPH` when it is
        # applied, and hands the function straight back when breaks are off.
        # Marking it then would brand the undecorated function as done and
        # make a later call -- with the env set -- skip the work. Core's own
        # `_attend_block` stays installed, which is right: with no break
        # there is no frozen argument to work around.
        return False
    setattr(wrapped, _EAGER_BREAK_MARKER, True)
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


# ---------------------------------------------------------------------------
# Engram staging: keep the prefetch overlapped across capture segments
# ---------------------------------------------------------------------------
# Core forks a side stream for the Engram gather and joins it per layer,
# lazily -- `consume` waits on that layer's event where the layer needs its
# rows. The fork therefore spans the forward, which collides with vLLM's
# breakable CUDA graph twice:
#
#     hipErrorStreamCaptureUnjoined    a segment ended with work still on the
#                                      forked stream
#     hipErrorStreamCaptureIsolation   a captured stream waited on an event
#                                      recorded off-capture
#
# The answer is not to stop forking. vLLM's own DeepSeek-V4.1 keeps its side
# stream and puts the orderings in eager breaks instead -- `_start_prefetch`
# and `_finish_prefetch` in `vllm/models/deepseek_v41/nvidia/engram.py` are
# both decorated, under the comment "Eager boundaries let the lookup span
# piecewise graph segments." A break ends the current segment, runs the call
# eagerly on the capture stream and starts a fresh one, so the wait is issued
# where no segment owns it and the lookup keeps running across the ones that
# follow.
#
# An earlier version stood the fork down instead, issuing the gather on the
# compute stream. That was correct and cost the overlap between a largely-UVA
# gather and the layers consuming it -- a cost upstream does not pay, and one
# this had no measurement for.


def _eager_break(fn):
    """vLLM's capture break, or `fn` unchanged where there is none."""
    try:
        from vllm.compilation.breakable_cudagraph import eager_break_during_capture
    except ImportError:
        return fn
    return eager_break_during_capture(fn)


class VllmEngramStaging(EngramStaging):
    """`EngramStaging` whose stream orderings sit in eager breaks."""

    # All three write nothing and return nothing -- they only order streams --
    # so the decorator's in-place-output requirement is met trivially, and a
    # replay re-issuing the same wait against the same long-lived event is
    # what the ordering needs anyway.
    #
    # The fork too, not only the waits. An earlier version decorated the waits
    # alone, on the reasoning that this path always forks outside the forward
    # -- the metadata builder stages the step before the model is called. It
    # does not always: when the proxy pool is not bound yet, which is the case
    # for the dummy batches a capture runs, `v41_stage_step` stages from
    # inside the forward, so the fork opens inside a segment and the segment
    # ends with it open. Measured, that reasoning cost a boot:
    # 8 x hipErrorStreamCaptureUnjoined. Upstream decorates its own
    # `_start_prefetch` for the same reason.
    _prefetch = _eager_break(EngramStaging._prefetch)
    _await_layer = _eager_break(EngramStaging._await_layer)
    _await_fork = _eager_break(EngramStaging._await_fork)


def install_vllm_engram_staging() -> None:
    """Build `VllmEngramStaging` from here on. Idempotent."""
    set_engram_staging_cls(VllmEngramStaging)
