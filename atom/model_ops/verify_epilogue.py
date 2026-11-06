# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A speculative verify's work after sampling in one launch
(ATOM_MTP_FUSED_VERIFY_EPILOGUE).

Takes the sampler's token ids (one a verify row, then one bonus a request) and
does what the eager chain after sampling did, in one kernel:

* the rejection (``rejection_greedy_sample_kernel``): a draft stands while it
  equals the sampled target token; the output rows ``[bs, num_spec_steps + 1]``
  and the verdict ``[3, bs]`` (accepted, rejected, anchor row);
* the presence marks (`PresencePenaltyStep.mark`): every emitted token set in
  its request's mask;
* the next anchor ids, ``out[b, accepted[b]]`` (the runner's ``torch.gather``);
* with ATOM_MTP_DRAFT_PRESENCE_PENALTY: the draft penalty's per-step reset
  (`DraftPresencePenalty.begin`, five launches);
* on a uniform, graphed verify under ATOM_MTP_FUSED_DRAFT_GLUE: step 0's staged
  inputs and the anchor scatter into the target's ids
  (`eagle_fused_glue.stage_verify_step0`);
* with ATOM_MTP_VERIFY_HOST_HANDOFF: the output rows and the
  ``[accepted, rejected]`` status written straight into pinned host slots, each
  then flagged at system scope (`output_transfer.HostHandoffRing`), so the
  host reads them without an event, a side-stream copy and its wake-up.

The same values, bit for bit, as the chain it replaces. Program 0 is every
request's verdict, one wave: the rejection, the hand-off, the marks, the next
ids, the staged ids and anchors and the scatter; the programs after it copy
the staged hidden states and positions, which no verdict decides. Program 0
reads the drafts from the target's ids and then scatters the next ids into
them, which can overwrite a rejected draft: one wave, and the scattered values
come from those loads, so the reads are done first; no other program reads
the target's ids.

The sampler's tokens are the input and its contract is unchanged. As the
sampler's epilogue later, this would run in the select kernel's last arriving
program (an atomic arrival count, no waiting), once every row is written.
"""

from typing import NamedTuple

import torch
import triton
import triton.language as tl

# Hidden-state rows a copy program moves.
_COPY_ROWS = 2


class VerifyHandoff(NamedTuple):
    """What the epilogue wrote for the drafter's `propose`, which skips it:
    step 0's staged buffers (or None), and whether the draft presence penalty
    was reset for the step."""

    staged0: dict | None
    presence_begun: bool


@triton.jit(
    do_not_specialize=[
        "mask_stride",
        "dp_n",
        "dp_rows",
        "dp_ctx_len",
        "hidden_stride",
        "staged_hidden_stride",
        "num_spec_steps",
        "scheduled_bs",
        "running_bs",
        "num_tokens",
        "tok_seq",
        "status_seq",
    ],
    do_not_specialize_on_alignment=[
        "target_ptr",
        "bonus_ptr",
        "cu_ptr",
        "tli_ptr",
        "bli_ptr",
        "input_ids_ptr",
        "out_ptr",
        "verdict_ptr",
        "next_ptr",
        "slot_ptr",
        "step_slots_ptr",
        "step_pen_ptr",
        "ids_ptr",
        "positions_ptr",
        "staged_ids_ptr",
        "staged_pos_ptr",
        "staged_anchors_ptr",
        "host_tok_ptr",
        "host_status_ptr",
        "tok_flag_ptr",
        "status_flag_ptr",
    ],
)
def _verify_epilogue_kernel(
    # the sampler's ids: one a verify row, one bonus a request
    target_ptr,
    bonus_ptr,
    # SpecDecodeMetadata; draft r is input_ids[tli[r] + 1]
    cu_ptr,
    tli_ptr,
    bli_ptr,
    input_ids_ptr,
    # results: [bs, num_spec_steps + 1], [3, bs], [bs]
    out_ptr,
    verdict_ptr,
    next_ptr,
    # presence marks (MARK)
    slot_ptr,
    mask_ptr,
    mask_stride,
    vocab,
    # DraftPresencePenalty.begin (BEGIN): its buffers and the step's rows
    dp_slots_ptr,
    dp_pen_ptr,
    dp_pos_ptr,
    dp_ctx_ptr,
    step_slots_ptr,
    step_pen_ptr,
    dp_n,
    dp_rows,
    dp_ctx_len,
    # step 0's staging (STAGE): the target's ids (input_ids + 1), positions,
    # hidden states, and step 0's buffers at running_bs
    ids_ptr,
    positions_ptr,
    hidden_ptr,
    hidden_stride,
    staged_ids_ptr,
    staged_pos_ptr,
    staged_hidden_ptr,
    staged_hidden_stride,
    staged_anchors_ptr,
    # the host hand-off (HOST): pinned rows and their flags
    host_tok_ptr,
    host_status_ptr,
    tok_flag_ptr,
    status_flag_ptr,
    tok_seq,
    status_seq,
    num_spec_steps,
    scheduled_bs,
    running_bs,
    num_tokens,
    R_P2: tl.constexpr,
    S_P2: tl.constexpr,
    MARK: tl.constexpr,
    BEGIN: tl.constexpr,
    BEGIN_STEP: tl.constexpr,
    DP_BLOCK: tl.constexpr,
    STAGE: tl.constexpr,
    HOST: tl.constexpr,
    UNIFORM: tl.constexpr,
    Q: tl.constexpr,
    Q_P2: tl.constexpr,
    H: tl.constexpr,
    COPY_ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid == 0:
        b = tl.arange(0, R_P2)
        real = b < scheduled_bs
        p = tl.arange(0, S_P2)[None, :]
        if UNIFORM:
            # every request verifies num_spec_steps drafts on its own
            # num_spec_steps + 1 rows: the indices the metadata holds
            start = b * num_spec_steps
            nd = tl.where(real, num_spec_steps, 0)
            verify = real[:, None] & (p < nd[:, None])
            rows = b[:, None] * (num_spec_steps + 1) + p
        else:
            start = tl.load(cu_ptr + b - 1, mask=real & (b > 0), other=0)
            nd = tl.load(cu_ptr + b, mask=real, other=0) - start
            verify = real[:, None] & (p < nd[:, None])
            rows = tl.load(tli_ptr + start[:, None] + p, mask=verify, other=0)
        draft = tl.load(input_ids_ptr + rows + 1, mask=verify, other=-1).to(tl.int32)
        target = tl.load(target_ptr + start[:, None] + p, mask=verify, other=-1)
        target = target.to(tl.int32)
        # the first mismatch is the correction; none: every draft stands
        accepted = tl.min(tl.where(verify & (draft != target), p, nd[:, None]), axis=1)
        rejected = accepted < nd
        bonus = tl.load(bonus_ptr + b, mask=real, other=-1).to(tl.int32)
        tok = tl.where(verify & (p <= accepted[:, None]), target, -1)
        tok = tl.where(p == nd[:, None], tl.where(rejected, -1, bonus)[:, None], tok)
        in_row = real[:, None] & (p <= num_spec_steps)
        flat = b[:, None] * (num_spec_steps + 1) + p
        tl.store(out_ptr + flat, tok, mask=in_row)
        rejected_count = num_spec_steps - accepted
        if HOST:
            # first, so the host waits for as little as possible
            tl.store(host_tok_ptr + flat, tok, mask=in_row)
            tl.store(host_status_ptr + b, accepted, mask=real)
            tl.store(host_status_ptr + scheduled_bs + b, rejected_count, mask=real)
            tl.debug_barrier()
            tl.atomic_xchg(tok_flag_ptr, tok_seq, sem="release", scope="sys")
            tl.atomic_xchg(status_flag_ptr, status_seq, sem="release", scope="sys")
        if UNIFORM:
            anchor = b * (num_spec_steps + 1) + accepted
        else:
            anchor = tl.load(bli_ptr + b, mask=real, other=0) - nd + accepted
        tl.store(verdict_ptr + b, accepted, mask=real)
        tl.store(verdict_ptr + scheduled_bs + b, rejected_count, mask=real)
        tl.store(verdict_ptr + 2 * scheduled_bs + b, anchor, mask=real)
        nxt = tl.sum(tl.where(p == accepted[:, None], tok, 0), axis=1)
        tl.store(next_ptr + b, nxt, mask=real)
        if MARK:
            slot = tl.load(slot_ptr + b, mask=real, other=0).to(tl.int64)
            tl.store(
                mask_ptr + slot[:, None] * mask_stride + tok,
                tl.full([R_P2, S_P2], 1, tl.uint8),
                mask=in_row & (tok >= 0) & (tok < vocab),
            )
        if STAGE:
            # Request b's rows are [b * Q, (b + 1) * Q); a pad request (b past
            # scheduled_bs) repeats the last token row and the last anchor.
            # The anchor row stages the next id, which the target's ids then
            # keep, as `input_ids.scatter_` left them.
            last = scheduled_bs - 1
            last_anchor = tl.sum(tl.where(b == last, anchor, 0), axis=0)
            last_next = tl.sum(tl.where(b == last, nxt, 0), axis=0)
            run = b < running_bs
            tl.store(
                staged_anchors_ptr + b,
                tl.where(real, anchor, last_anchor).to(tl.int64),
                mask=run,
            )
            k = tl.arange(0, Q_P2)[None, :]
            r = b[:, None] * Q + k
            src = tl.minimum(r, num_tokens - 1)
            is_anchor = src == tl.where(real, anchor, last_anchor)[:, None]
            live = run[:, None] & (k < Q)
            ids = tl.load(ids_ptr + src, mask=live & ~is_anchor)
            ids = tl.where(is_anchor, tl.where(real, nxt, last_next)[:, None], ids)
            tl.store(staged_ids_ptr + r, ids, mask=live)
            tl.store(ids_ptr + anchor, nxt.to(ids.dtype), mask=real)
        if BEGIN:
            # DraftPresencePenalty.begin: no penalty, no drafts, then the
            # step's slots and penalties on its first dp_n rows
            d = tl.arange(0, DP_BLOCK)
            dlive = d < dp_rows
            if BEGIN_STEP:
                dreal = d < dp_n
                pen = tl.load(step_pen_ptr + d, mask=dreal, other=0)
                tl.store(dp_pen_ptr + d, pen, mask=dlive)
                tl.store(
                    dp_slots_ptr + d,
                    tl.load(step_slots_ptr + d, mask=dreal),
                    mask=dreal,
                )
            else:
                tl.store(dp_pen_ptr + d, tl.zeros([DP_BLOCK], tl.int32), mask=dlive)
            tl.store(dp_pos_ptr + d, tl.zeros([DP_BLOCK], tl.int32), mask=dlive)
            for base in range(0, dp_ctx_len, DP_BLOCK):
                c = base + d
                tl.store(
                    dp_ctx_ptr + c,
                    tl.full([DP_BLOCK], -1, tl.int32),
                    mask=c < dp_ctx_len,
                )
    elif STAGE:
        # step 0's hidden states and positions + 1: flat row r stages token
        # row min(r, num_tokens - 1)
        row = (pid - 1) * COPY_ROWS + tl.arange(0, COPY_ROWS)
        row_live = row < running_bs * Q
        row_src = tl.minimum(row, num_tokens - 1)
        pos = tl.load(positions_ptr + row_src, mask=row_live)
        tl.store(staged_pos_ptr + row, pos + 1, mask=row_live)
        cols = tl.arange(0, H)
        x = tl.load(
            hidden_ptr + row_src.to(tl.int64)[:, None] * hidden_stride + cols[None, :],
            mask=row_live[:, None],
        )
        tl.store(
            staged_hidden_ptr
            + row.to(tl.int64)[:, None] * staged_hidden_stride
            + cols[None, :],
            x,
            mask=row_live[:, None],
        )


def _rows_block(max_bs: int) -> int:
    # Program 0's request lanes: the deployment's largest running batch, so
    # one compiled variant serves every batch.
    return max(16, triton.next_power_of_2(max_bs))


def verify_epilogue(
    target_token_ids: torch.Tensor,
    bonus_token_ids: torch.Tensor,
    metadata,
    *,
    max_bs: int,
    presence_step=None,
    draft_presence=None,
    draft_presence_bs: int = 0,
    stage: dict | None = None,
    host=None,
    uniform: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(output ids [bs, S + 1], verdict [3, bs], next anchor ids [bs])``, all
    int32, as `rejection_sample` (sampled target ids), ``presence_step.mark``
    and the anchor gather would make them; with ``draft_presence`` also its
    ``begin(presence_step, draft_presence_bs)``; with ``stage`` also
    `stage_verify_step0` of those next ids and anchors.

    ``metadata`` is the step's `SpecDecodeMetadata`; ``max_bs`` the largest
    running batch of the deployment. ``stage`` holds the target's ids
    (``input_ids``, the stream after ``metadata.input_ids``), first-axis
    ``positions``, ``hidden_states``, step 0's buffers (``staged``),
    ``scheduled_bs``, ``running_bs`` and ``query_width``, on a uniform verify
    of ``scheduled_bs * query_width`` rows. ``host`` is a pair of
    `output_transfer.HostSlot` (the output rows', the ``[accepted, rejected]``
    status's) the kernel fills and flags. ``uniform``: every request verifies
    ``S`` drafts on its own ``S + 1`` rows, in order (the metadata's indices
    are then not read).
    """
    cu = metadata.cu_num_draft_tokens
    bs = cu.numel()
    S = metadata.num_spec_steps
    dev = cu.device
    assert 1 <= bs <= max_bs and bonus_token_ids.numel() == bs
    assert target_token_ids.numel() == metadata.target_logits_indices.numel()
    assert target_token_ids.is_contiguous() and bonus_token_ids.is_contiguous()
    assert not uniform or target_token_ids.numel() == bs * S
    # One allocation for the three results; the views keep it alive.
    buf = torch.empty(bs * (S + 5), dtype=torch.int32, device=dev)
    out = buf[: bs * (S + 1)].view(bs, S + 1)
    verdict = buf[bs * (S + 1) : bs * (S + 4)].view(3, bs)
    nxt = buf[bs * (S + 4) :]

    unused = cu  # the pointer passed for what a variant does not read
    if presence_step is not None:
        state = presence_step.state
        assert presence_step.slots.numel() >= bs
        mark = (
            presence_step.slots,
            state.mask,
            state.mask.stride(0),
            state.mask.shape[1],
        )
    else:
        mark = (unused, unused, 0, 0)
    begin_step = False
    dp_block = 16
    if draft_presence is not None:
        dp = draft_presence
        if presence_step is not None:
            begin_step = True
            dp_n = min(int(draft_presence_bs), presence_step.slots.numel(), dp.max_rows)
            step_bufs = (presence_step.slots, presence_step.penalties.view(torch.int32))
        else:
            dp_n = 0
            step_bufs = (unused, unused)
        begin = (
            dp.slots,
            dp.penalties.view(torch.int32),
            dp.pos,
            dp.ctx,
            *step_bufs,
            dp_n,
            dp.max_rows,
            dp.ctx.numel(),
        )
        dp_block = max(16, triton.next_power_of_2(dp.max_rows))
    else:
        begin = (unused,) * 6 + (0, 0, 0)

    grid = 1
    if stage is not None:
        q = stage["query_width"]
        sbs, rbs = stage["scheduled_bs"], stage["running_bs"]
        ids = stage["input_ids"]
        positions = stage["positions"]
        hidden = stage["hidden_states"]
        staged = stage["staged"]
        num_tokens = ids.shape[0]
        assert sbs == bs and num_tokens == bs * q and bs <= rbs <= max_bs
        num_rows = rbs * q
        s_ids = staged["input_ids"].view(-1)
        s_pos = staged["positions"].view(-1)
        s_hid = staged["hidden_states"]
        s_hid = s_hid.view(-1, s_hid.shape[-1])
        s_lti = staged["last_token_indices"]
        h = hidden.shape[-1]
        assert s_hid.shape[0] == num_rows and s_hid.stride(-1) == 1
        assert hidden.stride(-1) == 1 and positions.stride(-1) == 1
        assert s_ids.numel() == num_rows and s_pos.numel() == num_rows
        assert s_lti.numel() == rbs and h == triton.next_power_of_2(h)
        staging = (
            ids,
            positions,
            hidden,
            hidden.stride(0),
            s_ids,
            s_pos,
            s_hid,
            s_hid.stride(0),
            s_lti,
        )
        grid += triton.cdiv(num_rows, _COPY_ROWS)
    else:
        q, rbs, num_tokens, h = 1, bs, 0, 16
        staging = (unused,) * 3 + (0,) + (unused,) * 3 + (0, unused)

    if host is not None:
        tok_slot, status_slot = host
        assert (
            tok_slot.view.numel() == bs * (S + 1) and status_slot.view.numel() == 2 * bs
        )
        handoff = (
            tok_slot.view,
            status_slot.view,
            tok_slot.flag,
            status_slot.flag,
            tok_slot.seq,
            status_slot.seq,
        )
    else:
        handoff = (unused,) * 4 + (0, 0)

    _verify_epilogue_kernel[(grid,)](
        target_token_ids,
        bonus_token_ids,
        cu,
        metadata.target_logits_indices,
        metadata.bonus_logits_indices,
        metadata.input_ids,
        out,
        verdict,
        nxt,
        *mark,
        *begin,
        *staging,
        *handoff,
        S,
        bs,
        rbs,
        num_tokens,
        R_P2=_rows_block(max_bs),
        S_P2=triton.next_power_of_2(S + 1),
        MARK=presence_step is not None,
        BEGIN=draft_presence is not None,
        BEGIN_STEP=begin_step,
        DP_BLOCK=dp_block,
        STAGE=stage is not None,
        HOST=host is not None,
        UNIFORM=uniform,
        Q=q,
        Q_P2=triton.next_power_of_2(q),
        H=h,
        COPY_ROWS=_COPY_ROWS,
        num_warps=1,
    )
    return out, verdict, nxt


def warm_verify_epilogue_kernels(
    *,
    max_bs: int,
    num_spec_steps: int,
    vocab: int,
    hidden_size: int,
    hidden_dtype: torch.dtype,
    ids_dtype: torch.dtype,
    device,
    draft_presence=None,
    host_capacity: tuple[int, int] | None = None,
) -> None:
    """Compile every variant serving launches (with and without a presence
    step, with and without step 0's staging) on scratch tensors of the served
    dtypes, so the first verify after startup compiles nothing. ``draft_presence``
    and ``host_capacity`` (the token and status rings' slot capacities) are the
    deployment's; only scratch is written."""
    import itertools
    from types import SimpleNamespace as NS

    from atom.model_engine.output_transfer import HostHandoffRing

    S, q = num_spec_steps, num_spec_steps + 1
    i32 = {"dtype": torch.int32, "device": device}
    i64 = {"dtype": torch.int64, "device": device}
    ids = torch.zeros(q + 2, dtype=ids_dtype, device=device)
    meta = NS(
        input_ids=ids[:q],
        num_spec_steps=S,
        cu_num_draft_tokens=torch.full((1,), S, **i32),
        target_logits_indices=torch.arange(S, **i64),
        bonus_logits_indices=torch.full((1,), S, **i64),
    )
    sampled = torch.zeros(S + 1, **i32)
    mask = torch.zeros(1, vocab, dtype=torch.uint8, device=device)
    step = NS(
        state=NS(mask=mask),
        slots=torch.zeros(1, **i32),
        penalties=torch.zeros(1, dtype=torch.float32, device=device),
    )
    dp = None
    if draft_presence is not None:
        rows = draft_presence.max_rows
        dp = NS(
            max_rows=rows,
            slots=torch.zeros(rows, **i32),
            penalties=torch.zeros(rows, dtype=torch.float32, device=device),
            pos=torch.zeros(rows, **i32),
            ctx=torch.zeros(draft_presence.ctx.numel(), **i32),
        )
    rings = None
    if host_capacity is not None:
        rings = tuple(HostHandoffRing(2, cap) for cap in host_capacity)
    for with_step, with_stage, uniform in itertools.product((False, True), repeat=3):
        stage = None
        if with_stage:
            stage = {
                "input_ids": ids[1 : q + 1],
                "positions": torch.zeros(q, **i64),
                "hidden_states": torch.zeros(
                    q, hidden_size, dtype=hidden_dtype, device=device
                ),
                "staged": {
                    "input_ids": torch.zeros(1, q, dtype=ids_dtype, device=device),
                    "positions": torch.zeros(1, q, **i64),
                    "hidden_states": torch.zeros(
                        1, q, hidden_size, dtype=hidden_dtype, device=device
                    ),
                    "last_token_indices": torch.zeros(1, **i64),
                },
                "scheduled_bs": 1,
                "running_bs": 1,
                "query_width": q,
            }
        host = None
        if rings is not None:
            host = (rings[0].claim((1, q)), rings[1].claim((2, 1)))
        verify_epilogue(
            sampled[:S],
            sampled[S:],
            meta,
            max_bs=max_bs,
            presence_step=step if with_step else None,
            draft_presence=dp,
            draft_presence_bs=1,
            stage=stage,
            host=host,
            uniform=uniform,
        )
        if host is not None:
            for slot in host:
                slot.synchronize()
                slot.release()
    torch.cuda.synchronize(device)
