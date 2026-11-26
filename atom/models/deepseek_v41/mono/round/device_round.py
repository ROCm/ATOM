# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Device-side round advance on the steps the V4.1 mono hi hook serves (``ATOM_DSV41_MONO_HI``; on top of the lean host
prep, ``lean_host``).

On a DSpark decode the host waits for the previous step's verdict (the rejected-draft counts, copied to the host) before it
builds the next step's metadata: the next verify starts where the verdict leaves the request (``starts = context_lens -
(spec + 1) - rejected``). With lean host prep (``lean_host``) that is the only verdict-dependent input a served step still
reads from the host. Everything else it reads is device-resident or the same every step for one request: the ids
(``fill_deferred_decode_ids`` from the previous sampled ids and the draft's), the slot, the batch ids, the query starts, the
block table (the scheduler allocates the whole draft span ahead), Engram's cursor (committed on the device).

On a device-round step the next step's positions are advanced on the device instead: ``positions[i] = start + 1 + accepted + i``
(``_advance_kernel``), where ``start`` is the previous served step's first position (the hook's ``pos``, written by its
begin-step bridge) and ``accepted`` is the previous verdict's accepted-draft count, read off its anchor row (a device
tensor: the anchors the runner committed, ``DeviceRound.committed``; request b's anchor is its row b x rows +
accepted). The builder tells the runner the step consumes the device verdict (``consumes``,
``AttentionMetadataBuilder.consumes_device_verdict``): the runner builds the step's metadata without waiting, as if no
draft had been rejected (zero counts: a start no lower than the true one, inside the block table's span), the builder
publishes it, and the advance kernel overwrites the positions on the same stream before any reader runs (Engram staging
and the target graph come after it). The runner collects the status after the target graph is launched, where the
scheduler's output needs it. The stale-state probe (``cache.prepare_state``), which compares the restored cursor with
the host's start, is skipped on these steps; ``cache.prepare_device_round_state`` checks on the host instead that the
step continues the previous one's requests.

A step is a device-round step only when every link holds: the lean predicate (a served one-request verify at a served width,
every layer on the hi path), the previous forward was a served step of the same request that produced a verdict, the verify
width is the hook's (the full DSpark span), the on-device Engram path is in use, no image rows. The decision is host-side and
TP-uniform (the scheduler's batch and this rank's own history of identical forwards). Any other step takes the lean path.

Discards: a request that ends inside an accepted block (EOS, ``max_tokens``, a stop string), is aborted or preempted is
handled by the scheduler as today: with deferred outputs the scheduler has already scheduled the next step before it sees the
previous one's tokens, so the next step runs (and is dropped) on the host path too. The device round changes when the host
learns the verdict, not which steps run nor where they write: its positions are the host's formula on the device's copy of
the same verdict.

**The first decode step after a prefill**: the step right after its requests' final prefill chunk waits, on the
host, for the prefill's status (the rejected / bonus counts the next step's metadata reads), which is ready only when
the whole prefill has run: the host then prepares the step while the GPU idles (~0.85 ms at ~10K, every turn of a
conversation). A prefill verifies nothing, so that status is known: 0 rejected, 0 bonus. Such a step (a served verify
of the requests the output-producing prefill forward before it prefilled) consumes no verdict either: the runner builds
its metadata from zeros without waiting and takes the owed wait after the target launch, like a device-round step; its
positions are the host's (nothing to advance).

Called at the builder's decode-round points (``round.HiRound``); such a step's routing decision is
``RoundService.DEVICE`` (``AttentionMetaData.served_by_round``), on which the builder takes the cache's device-round
state (no probe). Nothing changes with the hook off, or on any other step.
"""

from __future__ import annotations

import logging

import torch
import triton
import triton.language as tl

from atom.models.deepseek_v41.mono.hi_gate import log_once
from atom.utils.forward_context import RoundService

logger = logging.getLogger("atom")


@triton.jit
def _advance_kernel(positions, start, anchors, rows, BLOCK: tl.constexpr):
    """Program b: positions[b rows + i] = start[b] + 1 + accepted + i, i < rows, where accepted = anchors[b] - b rows
    (request b of the step: the next verify's anchor is the previous one's bonus token, ``accepted`` rows past its
    anchor; the previous verify laid request b out at row b x rows, its anchor that row + its accepted drafts).
    """
    b = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    s = (
        tl.load(start + b).to(tl.int64)
        + 1
        + tl.load(anchors + b).to(tl.int64)
        - b * rows
    )
    tl.store(
        positions + b * rows + i, (s + i).to(positions.dtype.element_ty), mask=i < rows
    )


def advance(
    positions: torch.Tensor,
    start: torch.Tensor,
    anchors: torch.Tensor,
    rows: int,
    requests: int = 1,
) -> None:
    """``_advance_kernel`` on the current stream: ``positions`` [>= requests x rows] (int32 / int64), ``start`` int32
    [>= requests] (each request's previous first position), ``anchors`` int32 [requests] (the previous verdict's anchor
    rows: the rejection sampler's verdict row 2); ``rows`` a request."""
    _advance_kernel[(requests,)](
        positions, start, anchors, rows, BLOCK=triton.next_power_of_2(max(rows, 2))
    )


def device_round_step(state, batch, max_seqlen_q, hook) -> bool:
    """Whether this forward may skip the verdict wait (module docstring): host-side, from the scheduler's batch and the
    runner's own previous forward (identical on every rank). B requests (a batch width of the hook): the same
    requests, in the same order, as the served step that left the verdict."""
    return (
        _served_verify(batch, max_seqlen_q, hook)
        and state.ready is not None
        and state.ready == tuple(batch.req_ids[: batch.total_seqs_num])
    )


def first_step(state, batch, max_seqlen_q, hook) -> bool:
    """Whether this forward is the first decode step after a prefill and may skip the prefill's status wait (module
    docstring): a served verify right after an output-producing forward
    that prefilled only requests of this step (the status it waits for is that prefill's: zero for each of them; the
    step's other requests take their counts from the scheduler, as they would after the wait). Host-side, TP-uniform.
    """
    return (
        _served_verify(batch, max_seqlen_q, hook)
        and state.prefilled is not None
        and set(state.prefilled) <= set(batch.req_ids[: batch.total_seqs_num])
    )


def _served_verify(batch, max_seqlen_q, hook) -> bool:
    """A decode verify the hook serves on the lean path: its full DSpark span at a served width, no image rows, no
    replayed rows."""
    if hook is None or not hook.lean_host or not hook.runs_every_layer:
        return False
    n = batch.total_seqs_num
    if batch.is_dummy_run or batch.total_seqs_num_prefill > 0 or not 1 <= n:
        return False
    width = n * max_seqlen_q
    if width not in hook.served_widths or batch.num_spec_step + 1 != max_seqlen_q:
        return False
    if n > 1 and getattr(hook, "split", {}).get(width) != (n, max_seqlen_q):
        return False
    data = getattr(batch, "multimodal_data", None)
    return not (data or getattr(batch, "replaying", None))


class _State:
    """The device round's host-side state across forwards."""

    def __init__(self):
        self.on = False  # this forward is a device-round step
        self.cur_req = None  # this forward's requests (a tuple), if it may be served
        self.cur_served = False  # this forward's step was a lean served one
        self.ready = (
            None  # the requests whose served step left a verdict + start on the device
        )
        self.anchors = None  # that verdict's anchor rows (device int32 [requests])
        self.advance_from = (
            None  # the anchors this forward advances from (a device-round step)
        )
        self.first = False  # this forward is a first step after a prefill that skips the prefill's status wait
        self.prefilled = None  # the requests of the previous forward, if it was an output-producing prefill of them all
        self.steps = 0


class DeviceRound:
    """The device round at the builder's decode-round points (``HiRound``: ``consumes``, ``laid_out``,
    ``committed``), on its host-side state (``_State``)."""

    def __init__(self, hook) -> None:
        self.hook, self.state = hook, _State()

    def consumes(self, batch, max_seqlen_q) -> bool:
        """Before the step's input ids, every forward (``consumes_device_verdict``): whether this forward is a
        device-round step or a first step after a prefill (either consumes no host-side status: the runner defers
        its wait), and the requests it may serve."""
        state, hook = self.state, self.hook
        state.on = device_round_step(state, batch, max_seqlen_q, hook)
        state.first = not state.on and first_step(state, batch, max_seqlen_q, hook)
        n, pre = batch.total_seqs_num, getattr(batch, "total_seqs_num_prefill", 0)
        produced = getattr(batch, "produces_output", None)
        state.prefilled = (
            tuple(batch.req_ids[:n])
            if n >= 1
            and pre == n
            and not batch.is_dummy_run
            and (produced is None or produced())
            else None
        )
        # the previous forward's verdict feeds this step only; whatever this forward is, it leaves its own (or none)
        state.advance_from = state.anchors if state.on else None
        state.ready = state.anchors = None
        state.cur_req = (
            tuple(batch.req_ids[:n])
            if 1 <= n <= getattr(hook, "bmax", 1) and not batch.is_dummy_run
            else None
        )
        state.cur_served = False
        return state.on or state.first

    def laid_out(self, metadata) -> None:
        """After the builder's layout: a device-round step's (``RoundService.DEVICE``) positions advanced on the
        device. The step's routing decision is taken: no later layout of this forward is a device-round one.
        """
        state, hook, step = self.state, self.hook, metadata.step
        running_bs, running_tokens = int(step.slots.shape[0]), step.running_tokens
        state.cur_served = (
            metadata.served_by_round >= RoundService.LEAN
            and state.cur_req is not None
            and running_bs == len(state.cur_req)
        )
        on, state.on, state.first = state.on, False, False
        if metadata.served_by_round is not RoundService.DEVICE:
            return
        if not on or state.advance_from is None:
            raise RuntimeError(
                "V4.1 mono: a device-round step without the verdict it advances from"
            )
        # after the step's publication on this stream: every reader of the positions runs after this
        advance(
            step.positions,
            hook.pos,
            state.advance_from,
            running_tokens // running_bs,
            running_bs,
        )
        state.steps += 1

    def committed(self, metadata, anchors) -> None:
        """A served verify's commit at the verdict's ``anchors`` (the runner's, whichever verdict it took): the
        verdict a served step of its requests leaves for the next step's advance."""
        state = self.state
        if (
            state.cur_served
            and state.cur_req is not None
            and anchors is not None
            and tuple(anchors.shape) == (len(state.cur_req),)
        ):
            state.ready, state.anchors = state.cur_req, anchors


def install(metadata_builder, hook) -> DeviceRound | None:
    """The device round for ``hook`` (``HiRound``'s), or None: it needs the lean host prep and the on-device Engram
    path."""
    if hook is None:
        return None
    if not hook.lean_host:
        log_once(
            logging.WARNING,
            "V4.1 mono hi: the device round needs the lean host prep: off",
        )
        return None
    engram = getattr(metadata_builder, "engram", None)
    if engram is not None and engram.host.overlap is None:
        log_once(
            logging.WARNING,
            "V4.1 mono hi: the device round needs the on-device Engram path: off",
        )
        return None
    log_once(
        logging.INFO,
        "V4.1 mono hi: device round advance on served steps; the first step after a prefill without its "
        "wait",
    )
    return DeviceRound(hook)
