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

import numpy as np
import torch

from atom.model_ops.attentions.deepseek_v41.backend import (
    DeepseekV41MetadataBuilder,
)
from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan
from atom.model_ops.attentions.deepseek_v41.packed_attention import (
    packed_decode,
    packed_prefill,
)
from atom.model_ops.engram.device.hashing import engram_snapshot_indices
from atom.model_ops.engram.device.staging import EngramStaging
from atom.model_ops.engram.device.uva import uva_gather_into
from atom.model_ops.v4_kernels import sparse_attn_v4_paged_2src
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

    def _capture_dummy_cache(self, pages, slots, running_tokens):
        """One scratch cache, kept alive for as long as a capture may run.

        Core allocates a dummy batch's scratch per call and frees it with the
        call, which is right for ATOM's own engine: its capture entry
        fabricates a batch with ``is_dummy_run=False`` against the serving
        cache, so no recorded kernel ever holds a scratch address.

        This path does capture dummy batches -- `v41_stage_step` falls back
        to a fabricated one whenever a step arrives before the proxy pool is
        bound, which includes vLLM's memory-profiling forward -- and a
        scratch freed when the capture returns leaves the recorded kernels
        pointing at freed memory, read back on the first replay as an illegal
        access reported asynchronously somewhere else entirely.

        Allocated once at the capture ceiling and kept on the builder. Every
        bucket's dummy batch fits inside it -- its page, slot and token counts
        are caps and the batch indexes from zero -- so one allocation serves
        them all rather than one per bucket (the STATE side alone is ~5 MiB
        per slot).
        """
        if not breakable_capture_enabled():
            return PagedAttentionCache(
                self.geometry,
                pages,
                slots,
                self.device,
                max_tokens=running_tokens,
                workspace=self.score_workspace,
            )
        cached = getattr(self, "_capture_dummy", None)
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
            self._capture_dummy = cached
        return cached

    def _prepare(
        self,
        batch,
        running_bs,
        running_tokens,
        *,
        max_q_len=None,
        tentative=False,
        is_prefill=False,
        start_positions=None,
        query_prefix_ready=False,
        engram_live=True,
    ):
        """``engram_live`` False: Engram's forward kernels run on no token (a
        capture: they record on serving's buffers, touching none)."""
        # Once graphs bind the serving pool, an idle DP rank must publish
        # padding into that pool's metadata. A fresh scratch cache would refill
        # different indptr addresses while replay still reads the captured ones.
        if batch.is_dummy_run and self.cache is not None:
            return self._prepare_idle(
                batch,
                running_bs,
                running_tokens,
                max_q_len=max_q_len,
                tentative=tentative,
                is_prefill=is_prefill,
                query_prefix_ready=query_prefix_ready,
            )
        spans, rows, offset, next_page = [], [], 0, 0
        slots = batch.state_slots_committed
        if not batch.is_dummy_run and len(slots) != batch.total_seqs_num:
            raise ValueError("CSA2 requires a STATE slot for every scheduled request")
        for i, (request_id, length, end) in enumerate(
            zip(batch.req_ids, batch.num_scheduled_tokens, batch.context_lens)
        ):
            length, end = int(length), int(end)
            if length == 0:
                continue
            if batch.is_dummy_run:
                # Warmup uses private scratch. A dummy rank may never mutate
                # a live slot or PAGE, even when its fabricated block ID is 0.
                position = 0
                count = -(-length // self.block_size)
                blocks = np.arange(next_page, next_page + count, dtype=np.int32)
                next_page += count
                slot = len(spans)
            else:
                position = (
                    end - length if start_positions is None else int(start_positions[i])
                )
                blocks = batch.block_tables[i]
                slot = slots[i]
            spans.append(RequestSpan(request_id, position, offset, length, slot))
            rows.append(blocks)
            offset += length
        if offset != batch.total_tokens_num or running_tokens < offset:
            raise ValueError("CSA2 batch token spans disagree with the runner")
        # Each span's batch index: zero-token scheduler rows have none.
        scheduled = np.flatnonzero(batch.num_scheduled_tokens)
        resumes, floors, cached_until, histories = (
            (None,) * 4
            if batch.is_dummy_run or not batch.replaying
            else self._replay_layout(batch.replaying, scheduled, spans)
        )
        cache = (
            self._capture_dummy_cache(
                max(next_page, 1), max(len(spans), 1), running_tokens
            )
            if batch.is_dummy_run
            else self.cache
        )
        if cache is None:
            raise RuntimeError("CSA2 cache must be allocated before serving")
        compacted = len(spans) != len(batch.req_ids)
        return self._prepare_step(
            batch,
            spans,
            rows,
            cache,
            running_bs,
            running_tokens,
            max_q_len=max_q_len,
            tentative=tentative,
            is_prefill=is_prefill,
            query_prefix_ready=query_prefix_ready and not compacted,
            query_prefix_republish_reason=(
                "compact zero-token scheduler rows for CSA2 after input assembly"
                if query_prefix_ready and compacted
                else None
            ),
            engram_live=engram_live,
            scheduled=scheduled,
            replay_layout=(resumes, floors, cached_until, histories),
        )


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

    def prepare_state(self, step, *, histories=True):
        """Always render this step's stale-slot verdict now, never later.

        `histories=False` lets core ship the cursor rows without waiting and
        judge them on the next call, which takes one blocking D2H (~58 us) off
        each decode step. That deferred verdict is wrong under concurrency on
        this path, and wrong in the direction that costs most: it refuses
        correct state and takes the engine down with it. Measured,
        FULL_AND_PIECEWISE, 64 concurrent GSM8K requests, no KV connector:

            needs state at 952, found 951

        while the state-row dump for that same request, printed beside it,
        reads `computed 953, cursor 953` -- self-consistent. The verdict is
        rendered against a claim from one step and a `_probe` buffer that
        later steps have already reused, so the two describe different
        moments.

        A guard, not a fix: the register slip in the deferred probe is still
        not understood, and a check that fails closed on correct state is
        worse than the microseconds it saves. Native keeps the saving, since
        that is where the fault was never seen.

        `ATOM_V41_BLOCKING_STATE_PROBE=0` opts back out.
        """
        import os

        override = os.environ.get("ATOM_V41_BLOCKING_STATE_PROBE")
        if override is not None and override != "":
            blocking = override not in ("0", "false", "False")
        else:
            blocking = True
        return super().prepare_state(step, histories=histories or blocking)

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
# Attention: a capture break over a copy of the forward
# ---------------------------------------------------------------------------
# `eager_break_during_capture` ends a capture segment at the decorated call
# and resumes a new one after it. V4.1 needs that around attention because
# its kernels are launched with per-step host values -- the batch's longest
# KV extent among them -- which a capture freezes at whatever length it
# recorded while every later decode step grows past it.
#
# The decorator also requires the broken-out region to write through a
# caller-owned buffer: a fresh tensor hands back a different address on every
# replay while the captured segments after it still point at the first.
#
# Both of those are facts about this frontend, so the region lives here as a
# copy of `Attention.forward` rather than as a split in the shared model.
# What that copy must track is `atom/models/deepseek_v41/attention.py` --
# `tests/plugin/test_vllm_deepseek_v41_attention_copy.py` fails when core's
# version moves, which is the only warning available for a copy whose drift
# is arithmetic rather than a crash.
#
# The region is wide on purpose. Upstream keeps a narrow one for model runner
# V2 and a wide one for V1, because V1's piecewise capture produces garbage
# with the narrow region (vllm #51430): the attention input preparation stays
# inside the graph and the replay reuses it. Wide is correct under both.

_EAGER_BREAK_MARKER = "_atom_v41_eager_break_applied"


def _attn_out_buffer(self, hidden):
    """The buffer the broken-out region writes its rows into.

    Every leading dimension of `hidden`, not just the first: ATOM hands this
    layer `[batch, tokens, dim]` where upstream's native V4.1 gets
    `[tokens, dim]`, so `hidden.shape[0]` is the batch and sizing from it
    yields one row per sequence instead of one per token.
    """
    return torch.empty(
        (*hidden.shape[:-1], self.heads, self.head_dim),
        dtype=hidden.dtype,
        device=hidden.device,
    )


def _attend_block(self, attn_out, hidden, hidden_scale, cache, _step, rope, queries):
    """Everything from the compressor fork to the attention rows, in place.

    Reads the step from the cache, not from its argument: the decorator binds
    this call's arguments when the graph is recorded and reruns no host code
    to rebind them, so an argument would hold the capture's step on every
    replay. The cache outlives every step and `begin_step` leaves the current
    one on it. `_step` is the fallback for the window before a pool is bound.
    """
    step = getattr(cache, "current_step", None) or _step
    compressed = self._fork_compress(hidden, cache, step, rope)
    q_lora, kv_pre = self.project_qkv(hidden, hidden_scale)
    qr, qr_scale, kv_normed = self.qk_norm(q_lora, kv_pre)
    selecting = self._fork_select(
        hidden, qr, qr_scale, cache, step, rope, compressed=compressed
    )
    if queries is not None:
        qr, qr_scale = queries.take(qr, qr_scale)
    query = self.wq_b(qr, x_scale=qr_scale).unflatten(-1, (self.heads, self.head_dim))
    # One launch for both rotations, the KV row's FP8 bytes and, where a
    # decode allows it, the window write. `window_kv` is what is left for
    # `write_window` below, which is nothing when that fold happened.
    if queries is None:
        kv, window_kv = cache.rope_quant_window(
            self.spec.layer_id, query, kv_normed, rope, step
        )
    else:
        kv, window_kv = cache.rope_quant_window(
            self.spec.layer_id, None, kv_normed, rope, step
        )
        cache.rotate_query(query, rope, queries.positions)
    # A forked compressor is already inside this join, having been waited
    # at the scorer; where the scorer itself was not forked it waited on
    # the main stream instead and there is nothing left to join here.
    if selecting is not None:
        selecting.wait_stream(self.index_stream)
    prefix, prefix_indptr, second, second_indptr = (
        cache.attention_indices(self.spec, step)
        if queries is None
        else queries.indices(cache, self.spec)
    )
    flat_query = query.flatten(0, 1)
    if step.decode:
        cache.write_window(self.spec.layer_id, window_kv, step)
        if cache.packed:
            output = packed_decode(
                flat_query,
                cache.pool,
                prefix,
                prefix_indptr,
                self.attn_sink,
                self.softmax_scale,
            )
        else:
            # The window, this step's rows included, is read through the
            # ring: the second source, beside the selection in the pool.
            output = sparse_attn_v4_paged_2src(
                flat_query,
                cache.pool,
                prefix,
                prefix_indptr,
                cache.ring_view(self.spec.layer_id),
                second,
                second_indptr,
                self.attn_sink,
                self.softmax_scale,
                out=flat_query,
            )
    else:
        prefill = packed_prefill if cache.packed else sparse_attn_v4_paged_2src
        output = prefill(
            flat_query,
            cache.pool,
            prefix,
            prefix_indptr,
            kv.flatten(0, 1),
            second,
            second_indptr,
            self.attn_sink,
            self.softmax_scale,
            out=flat_query,
        )
        # Preserve the prior ring until every query has consumed its prefix.
        cache.write_window(self.spec.layer_id, window_kv, step)
    flat_out = attn_out.flatten(0, 1)
    # `out=` is a request, not a guarantee: these kernels are shared with
    # V4 and may answer in a buffer of their own, and the decode pair is
    # not offered one at all. A broken-out region has to write through the
    # caller's buffer to be one, so an answer that landed elsewhere is
    # copied in rather than ignored.
    rows = output.view_as(query).flatten(0, 1)
    if rows.data_ptr() != flat_out.data_ptr():
        flat_out.copy_(rows)


def _forward(self, hidden, hidden_scale, cache, step, rope, queries=None):
    """`Attention.forward` with its attention half inside a capture break."""
    attn_out = _attn_out_buffer(self, hidden)
    type(self)._atom_v41_attend_block(
        self, attn_out, hidden, hidden_scale, cache, step, rope, queries
    )
    return self._project_out(
        attn_out,
        rope,
        cache.rope_positions(step) if queries is None else queries.positions,
    )


def apply_v41_eager_break() -> bool:
    """Install the broken-out forward on `Attention`. Idempotent."""
    try:
        from vllm.compilation.breakable_cudagraph import eager_break_during_capture
    except ImportError:
        return False

    from atom.models.deepseek_v41.attention import Attention

    if getattr(Attention, _EAGER_BREAK_MARKER, False):
        return False
    wrapped = eager_break_during_capture(_attend_block)
    if wrapped is _attend_block:
        # The decorator reads `VLLM_USE_BREAKABLE_CUDAGRAPH` when it is
        # applied and hands the function straight back when breaks are off.
        # Core's own forward stays installed, which is right: with no break
        # there is no frozen argument to work around and no buffer to fill.
        return False
    Attention._atom_v41_attend_block = wrapped
    Attention.forward = _forward
    setattr(Attention, _EAGER_BREAK_MARKER, True)
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
    """`EngramStaging` with its stream orderings inside eager breaks.

    The three methods below are copies of core's, split at the points a
    capture break has to sit. Copies rather than seams in core, because every
    one of these boundaries is a fact about vLLM's breakable capture and about
    nothing else -- the flow they reproduce is the same flow, and a core that
    carried hooks for it would carry them for one caller.

    What a capture break buys here: a segment that ends with work still on a
    forked stream is refused (`hipErrorStreamCaptureUnjoined`), and a captured
    stream waiting on an event recorded off-capture is refused too
    (`hipErrorStreamCaptureIsolation`). Issuing the fork and the two waits
    inside breaks puts each where no segment owns it.

    The fork is kept, not stood down. The overlap between a largely-UVA gather
    and the layers consuming it is the whole point of the side stream, and
    upstream's own V4.1 keeps it for the same reason; standing it down also
    works and costs exactly that.
    """

    # `_prefetch` is broken out as well as the waits. An earlier version broke
    # only the waits, on the reasoning that this path always forks outside the
    # forward -- the metadata builder stages the step before the model is
    # called. It does not always: when the proxy pool is not bound yet, which
    # is the case for the dummy batches a capture runs, `v41_stage_step`
    # stages from inside the forward, so the fork opens inside a segment and
    # the segment ends with it open. Measured, that reasoning cost a boot:
    # 8 x hipErrorStreamCaptureUnjoined. Upstream decorates its own
    # `_start_prefetch` for the same reason.
    def start(self, width):
        self._snapshot(width)
        # Past the fork on purpose: the side stream reads the snapshot, not
        # the cursor, so the cursor may be advanced before the gather runs.
        self._advance_cursor(width)
        self._prefetch(width)

    def consume(self, layer, width):
        host = self.host
        self._await_layer(self.done[layer])
        if host._tp_group is not None and self.collective is None:
            out = self.flat[layer][:width]
            out = host._tp_group.all_gather(out, use_custom=True, dim=1)
            out = out[:, : host.embed_width]
            host.buffers[layer].gpu[:width].copy_(out)

    def join(self):
        # Close the fork even if a partial forward did not consume every layer.
        self._await_fork()


def _prefetch(self, width):
    """Fork, and issue every layer's lookup on the stream that results.

    The fork and the lookups belong to one segment, and `_snapshot` /
    `_advance_cursor` must stay outside it -- they write state a break would
    otherwise re-run on every replay, which would advance the cursor twice.
    """
    host = self.host
    compute = torch.cuda.current_stream(host.device)
    # Get the parent BEFORE entering the guard: waiting on ourselves would
    # neither order the inputs nor join the graph capture.
    self.stream.wait_stream(compute)
    with torch.cuda.stream(self.stream):
        for layer in host.layer_ids:
            ids = engram_snapshot_indices(
                self.uva.hash_tables,
                layer,
                self.snapshot[:width],
                self.uva.row_ids[:width],
            )
            table = host.prefetcher._tables[layer]
            uva_gather_into(
                table,
                ids,
                self.flat[layer][:width].view(width, host.local_heads, table.head_dim),
                head_start=host.head_start,
                local_heads=host.local_heads,
                total_heads=host.total_heads,
            )
            if self.collective is not None:
                out = self.gathered[layer][:width]
                self.collective.all_gather_unreg(
                    self.flat[layer][:width], out=out, dim=1
                )
                if self.gathered[layer] is not host.buffers[layer].gpu:
                    host.buffers[layer].gpu[:width].copy_(out[:, : host.embed_width])
            elif host._tp_group is None:
                host.buffers[layer].gpu[:width].copy_(self.flat[layer][:width])
            self.done[layer].record(self.stream)


def _await_layer(self, event) -> None:
    """Order the compute stream behind one layer's prefetch."""
    torch.cuda.current_stream(self.host.device).wait_event(event)


def _await_fork(self) -> None:
    """Order the compute stream behind every outstanding lookup."""
    torch.cuda.current_stream(self.host.device).wait_stream(self.stream)


VllmEngramStaging._prefetch = _eager_break(_prefetch)
VllmEngramStaging._await_layer = _eager_break(_await_layer)
VllmEngramStaging._await_fork = _eager_break(_await_fork)


_STAGING_MARKER = "_atom_v41_vllm_staging"


def install_vllm_engram_staging() -> None:
    """Put this path's staging on `EngramStaging` itself. Idempotent.

    Installed on the class rather than chosen through a registry, so core
    keeps no hook for picking one: the Engram runtime builds `EngramStaging`
    the way it always did, and this process has replaced the three methods
    whose boundaries vLLM's capture cares about.
    """
    if getattr(EngramStaging, _STAGING_MARKER, False):
        return
    for name in (
        "start",
        "consume",
        "join",
        "_prefetch",
        "_await_layer",
        "_await_fork",
    ):
        setattr(EngramStaging, name, getattr(VllmEngramStaging, name))
    setattr(EngramStaging, _STAGING_MARKER, True)
