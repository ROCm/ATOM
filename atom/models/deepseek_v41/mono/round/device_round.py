# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Device-side round advance on the steps the V4.1 mono hi hook serves (``ATOM_DSV41_MONO_HI``; on top of the lean host
prep, ``lean_host``).

On a DSpark decode the host waits for the previous step's verdict (the rejected-draft counts, copied to the host) before it
builds the next step's metadata: the next verify starts where the verdict leaves the request (``starts = context_lens -
(spec + 1) - rejected``). With lean host prep (``lean_host``) that is the only verdict-dependent input a served step still
reads from the host. Everything else it reads is device-resident or the same every step for one request: the ids
(``_fill_deferred_decode_ids`` from the previous sampled ids and the draft's), the slot, the batch ids, the query starts, the
block table (the scheduler allocates the whole draft span ahead), Engram's cursor (committed on the device).

On a device-round step the next step's positions are advanced on the device instead: ``positions[i] = start + 1 + accepted + i``
(``_advance_kernel``), where ``start`` is the previous served step's first position (the hook's ``pos``, written by its
begin-step bridge) and ``accepted`` is the previous verdict's accepted-draft count (row 0 of the rejection sampler's verdict, a
device tensor). The host builds the step's metadata without waiting, as if no draft had been rejected (the status it
hands ``prepare_input_ids`` is zeros: a start no lower than the true one, inside the block table's span), publishes it, and the advance kernel overwrites the positions on the same
stream before any reader runs (Engram staging and the target graph come after it). The host collects the verdict after the
target graph is launched, where the scheduler's output needs it. The stale-state probe (``cache.prepare_state``), which
compares the restored cursor with the host's start, is skipped on these steps.

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
of the same requests, in the same order, as the output-producing prefill forward before it) builds its metadata from
zeros without waiting and takes the owed wait after the target launch, like a device-round step; its positions are
the host's (nothing to advance).

``prologue``: a device-round step's packed publications and ids fill are recorded and launched with its advance as
one kernel.

Called at the runner's and the builder's decode-round hooks (``round.HiRound``); such a step's routing decision is
``RoundService.DEVICE`` (``AttentionMetaData.served_by_round``), on which the builder takes the cache's device-round
state (no probe). Nothing changes with the hook off, or on any other step.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import triton
import triton.language as tl

from atom.utils.forward_context import RoundService

logger = logging.getLogger("atom")


@triton.jit
def _advance_kernel(
    positions, start, verdict, verdict_stride, rows, BLOCK: tl.constexpr
):
    """Program b: positions[b rows + i] = start[b] + 1 + verdict[0, b] + i, i < rows (request b of the step: the
    next verify's anchor is the previous one's bonus token, ``accepted`` rows past its anchor).
    """
    b = tl.program_id(0)
    i = tl.arange(0, BLOCK)
    s = (
        tl.load(start + b).to(tl.int64)
        + 1
        + tl.load(verdict + b * verdict_stride).to(tl.int64)
    )
    tl.store(
        positions + b * rows + i, (s + i).to(positions.dtype.element_ty), mask=i < rows
    )


def advance(
    positions: torch.Tensor,
    start: torch.Tensor,
    verdict: torch.Tensor,
    rows: int,
    requests: int = 1,
) -> None:
    """``_advance_kernel`` on the current stream: ``positions`` [>= requests x rows] (int32 / int64), ``start`` int32
    [>= requests] (each request's previous first position), ``verdict`` int32 [3, requests] (the rejection sampler's:
    accepted, rejected, anchor, a column a request); ``rows`` a request."""
    _advance_kernel[(requests,)](positions, start, verdict, verdict.stride(1), rows,
                                 BLOCK=triton.next_power_of_2(max(rows, 2)))  # fmt: skip


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
        self.owed = (
            False  # its verdict wait is still owed (taken after the target launch)
        )
        self.cur_req = None  # this forward's requests (a tuple), if it may be served
        self.cur_served = False  # this forward's step was a lean served one
        self.ready = (
            None  # the request whose served step left a verdict + start on the device
        )
        self.verdict = None  # that verdict (device int32 [3, 1])
        self.advance_from = (
            None  # the verdict this forward advances from (a device-round step)
        )
        self.first = False  # this forward is a first step after a prefill that skips the prefill's status wait
        self.prefilled = None  # the requests of the previous forward, if it was an output-producing prefill of them all
        self.prefilled_by = None  # this forward's previous one's (``prefilled`` before this forward replaced it)
        self.steps = 0


class DeviceRound:
    """The device round at the runner's and the builder's hooks (``HiRound``: ``before_input_ids``, ``spec_status``,
    ``laid_out``, ``after_forward``, ``note_verdict``), on its host-side state (``_State``) and the prologue's recorder
    (``pro``)."""

    def __init__(self, hook, pro) -> None:
        self.hook, self.pro, self.state = hook, pro, _State()

    def before_input_ids(self, batch, max_seqlen_q) -> None:
        """``prepare_input_ids``, first: whether this forward is a device-round step or a first step after a prefill,
        and the requests it may serve."""
        state, hook = self.state, self.hook
        state.on = device_round_step(state, batch, max_seqlen_q, hook)
        state.first = not state.on and first_step(state, batch, max_seqlen_q, hook)
        # the prefill forward whose status this step owes (when ``first``)
        state.prefilled_by = state.prefilled
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
        state.advance_from = state.verdict if state.on else None
        state.ready = state.verdict = None
        state.cur_req = (
            tuple(batch.req_ids[:n])
            if 1 <= n <= getattr(hook, "bmax", 1) and not batch.is_dummy_run
            else None
        )
        state.cur_served = False
        # a device-round step's pre-target launches recorded until its advance; a decode step's publications read
        # their pinned arenas directly
        self.pro.arm(state.on, decode=n >= 1 and pre == 0)

    def spec_status(self):
        """``prepare_input_ids``' status: zeros on a device-round step (the step is built as if no draft had been
        rejected: module docstring) and the prefill's zeros on a first step (each wait owed: ``after_forward``), else
        None (the queue's).
        """
        state = self.state
        if state.on:
            state.owed = True  # taken after the target launch (``after_forward``), in the same queue order
            # one a request of the served step that left the verdict (this step's requests, in the same order)
            zeros = np.zeros(len(state.cur_req), dtype=np.int32)
            return zeros, zeros.copy()
        if state.first:
            state.owed = (
                True  # the same, for the prefill's status: known (module docstring)
            )
            # one a request of the prefill forward
            zeros = np.zeros(len(state.prefilled_by), dtype=np.int32)
            return zeros, zeros.copy()
        return None

    def laid_out(self, metadata) -> None:
        """After the builder's layout: a device-round step's (``RoundService.DEVICE``) positions advanced on the
        device (with the prologue's recorded launches as one kernel)."""
        state, hook, step = self.state, self.hook, metadata.step
        running_bs, running_tokens = int(step.slots.shape[0]), step.width
        state.cur_served = (
            metadata.served_by_round >= RoundService.LEAN
            and state.cur_req is not None
            and running_bs == len(state.cur_req)
        )
        if metadata.served_by_round is not RoundService.DEVICE:
            return
        if state.advance_from is None:
            raise RuntimeError(
                "V4.1 mono: a device-round step without the verdict it advances from"
            )
        # after the step's publication on this stream: every reader of the positions runs after this
        args = (
            step.positions,
            hook.pos,
            state.advance_from,
            running_tokens // running_bs,
            running_bs,
        )
        if self.pro.armed:  # with the recorded publications and ids fill, as one kernel
            self.pro.advance(*args)
        else:
            advance(*args)
        state.steps += 1

    def note_verdict(self, out) -> None:
        """A verify's (tokens, verdict, ..): the verdict a served step of its requests leaves for the next step's
        advance."""
        state = self.state
        if (
            state.cur_served
            and state.cur_req is not None
            and out[1].shape == (3, len(state.cur_req))
        ):
            state.ready, state.verdict = state.cur_req, out[1]

    def after_forward(self, processor) -> None:
        """After the target's launch: the owed verdict wait (what the scheduler's output reads, ``postprocess``)."""
        state = self.state
        if state.owed:
            state.owed = False
            processor.prev_rejected_num, processor.prev_bonus_num = (
                processor.recv_spec_status()
            )
        state.on = state.first = False


def install(metadata_builder, hook, runner) -> DeviceRound | None:
    """The device round for ``hook`` (``HiRound``'s) with its prologue recorder on ``runner``'s publications, or None:
    it needs the lean host prep and the on-device Engram path."""
    if hook is None:
        return None
    if not hook.lean_host:
        logger.warning("V4.1 mono hi: the device round needs the lean host prep: off")
        return None
    engram = getattr(metadata_builder, "engram", None)
    if engram is not None and engram.host.overlap is None:
        logger.warning(
            "V4.1 mono hi: the device round needs the on-device Engram path: off"
        )
        return None
    from atom.models.deepseek_v41.mono.round import prologue

    pro = prologue.install(runner)
    logger.info("V4.1 mono hi: device round advance on served steps; the first step after a prefill without its "
                "wait")  # fmt: skip
    return DeviceRound(hook, pro)
