# SPDX-License-Identifier: MIT
"""The eager glue of `EagleProposer.propose` between its draft passes, fused
(``ATOM_MTP_FUSED_DRAFT_GLUE``).

Without it a decode step's three MTP passes are separated by 57 small eager
launches: staging the verify stream into step 0's buffers, cloning each
replay's outputs out of the graph pool, exporting the draft ids, rewriting the
target's metadata to one row per sequence (`_enter_decode_metadata`),
advancing it a token (`_advance_decode_metadata` and the GDN builder's
`prepare_mtp_decode`), and staging the next pass's inputs. Here they are:

* `stage_verify_step0` -- one launch: the anchor scatter into the target's
  input ids, ``positions + 1`` and the pad-to-``running_bs`` staging of ids,
  positions, hidden states and anchors into step 0's buffers.
* `advance_draft_pass` -- two launches a mid-step: one kernel for every
  per-row update (context lens, positions, ``kv_indptr``, write slots,
  ``cu_seqlens_q``, the next pass's ids and hidden states staged with their
  pad rows, the draft id exported), then the GDN builder's own
  ``kv_indices`` generation, which reads the new ``kv_indptr``.

Integer metadata comes out bit for bit what the eager chain writes, and the
copies are copies, so the passes see the same inputs. The pass outputs are
read straight out of the graph pool (`DraftGraph.run(borrow=True)`): the
kernel consumes them before the next replay, on the same stream.

Kept out of the model and the builder: these are the proposer's own buffers,
and only the GDN hybrid's builder (`supports_fused_mtp_glue`) is matched.
"""

import torch
import triton
import triton.language as tl

from atom.utils.block_convert import kv_indices_generate_triton

# Rows of the hidden-state copy a program moves; the metadata is program 0's.
_COPY_ROWS_PER_PROGRAM = 4


# Runtime counts and the id / index / metadata pointers (views at any offset)
# are not specialized on: one compiled variant serves every batch, so nothing
# compiles mid-serve (`warm_fused_glue_kernels` compiles them at startup).
@triton.jit(
    do_not_specialize=["num_tokens", "num_rows", "scheduled_bs", "running_bs"],
    do_not_specialize_on_alignment=[
        "input_ids_ptr",
        "next_ids_ptr",
        "anchors_ptr",
        "positions_ptr",
        "staged_ids_ptr",
        "staged_positions_ptr",
        "staged_anchors_ptr",
    ],
)
def _stage_verify_step0_kernel(
    input_ids_ptr,
    next_ids_ptr,
    anchors_ptr,
    positions_ptr,
    hidden_ptr,
    hidden_stride,
    staged_ids_ptr,
    staged_positions_ptr,
    staged_hidden_ptr,
    staged_hidden_stride,
    staged_anchors_ptr,
    num_tokens,
    num_rows,
    scheduled_bs,
    running_bs,
    QUERY_WIDTH: tl.constexpr,
    H: tl.constexpr,
    ROWS: tl.constexpr,
    BLOCK_BS: tl.constexpr,
):
    """Step 0's staged inputs on a uniform verify batch.

    ``num_tokens = scheduled_bs * QUERY_WIDTH`` real rows, sequence ``b`` on
    rows ``[b * W, (b + 1) * W)`` and its anchor (the accepted row, which the
    rejection verdict puts inside its own segment) one of them. Flat row ``r``
    stages token row ``min(r, num_tokens - 1)``, the pad rule of
    `DraftGraph.stage`; its id is the anchor's next token on the anchor row.
    """
    pid = tl.program_id(0)
    if pid == 0:
        b = tl.arange(0, BLOCK_BS)
        real = b < scheduled_bs
        anchor = tl.load(anchors_ptr + tl.minimum(b, scheduled_bs - 1))
        tl.store(staged_anchors_ptr + b, anchor.to(tl.int64), mask=b < running_bs)
        # The target's input ids carry the scattered anchors afterwards, as
        # `input_ids.scatter_` left them. No program below reads an anchor row.
        nxt = tl.load(next_ids_ptr + b, mask=real)
        tl.store(input_ids_ptr + anchor, nxt, mask=real)
    rows = pid * ROWS + tl.arange(0, ROWS)
    live = rows < num_rows
    src = tl.minimum(rows, num_tokens - 1)
    seq = src // QUERY_WIDTH
    is_anchor = src == tl.load(anchors_ptr + seq, mask=live, other=-1)
    tok = tl.load(input_ids_ptr + src, mask=live & ~is_anchor)
    tok = tl.where(is_anchor, tl.load(next_ids_ptr + seq, mask=live & is_anchor), tok)
    tl.store(staged_ids_ptr + rows, tok, mask=live)
    pos = tl.load(positions_ptr + src, mask=live)
    tl.store(staged_positions_ptr + rows, pos + 1, mask=live)
    cols = tl.arange(0, H)
    for k in tl.static_range(ROWS):
        r = pid * ROWS + k
        if r < num_rows:
            s = tl.minimum(r, num_tokens - 1).to(tl.int64)
            x = tl.load(hidden_ptr + s * hidden_stride + cols)
            tl.store(
                staged_hidden_ptr + r.to(tl.int64) * staged_hidden_stride + cols, x
            )


@triton.jit(
    do_not_specialize=["draft_col", "scheduled_bs", "running_bs", "num_slot_rows"],
    do_not_specialize_on_alignment=[
        "src_ids_ptr",
        "dst_ids_ptr",
        "dst_positions_ptr",
        "draft_ptr",
        "context_lens_ptr",
        "num_reject_ptr",
        "kv_indptr_ptr",
        "block_tables_ptr",
        "slot_mapping_ptr",
        "cu_seqlens_q_ptr",
        "row_ids_ptr",
        "positions_src_ptr",
        "anchors_ptr",
    ],
)
def _advance_draft_pass_kernel(
    # this pass's outputs, read in place (graph pool or eager)
    src_ids_ptr,
    src_hidden_ptr,
    src_hidden_stride,
    # the next pass's staged inputs
    dst_ids_ptr,
    dst_hidden_ptr,
    dst_hidden_stride,
    dst_positions_ptr,
    # `[scheduled_bs, mtp_k]` draft ids, column `draft_col`
    draft_ptr,
    draft_stride,
    draft_col,
    # target metadata, one row per sequence
    context_lens_ptr,
    num_reject_ptr,
    kv_indptr_ptr,
    block_tables_ptr,
    block_tables_stride,
    slot_mapping_ptr,
    cu_seqlens_q_ptr,
    row_ids_ptr,
    # FIRST: positions of the step-0 stream and each sequence's anchor row
    positions_src_ptr,
    anchors_ptr,
    scheduled_bs,
    running_bs,
    num_slot_rows,
    KV_PAGE: tl.constexpr,
    SLOT_PAGE: tl.constexpr,
    H: tl.constexpr,
    FIRST: tl.constexpr,
    HAS_REJECT: tl.constexpr,
    BLOCK: tl.constexpr,
    ROWS: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid == 0:
        r = tl.arange(0, BLOCK)
        live = r < running_bs
        real = r < scheduled_bs
        if FIRST:
            # _enter_decode_metadata: one query per row.
            tl.store(
                cu_seqlens_q_ptr + r,
                tl.load(row_ids_ptr + r, mask=r <= running_bs),
                mask=r <= running_bs,
            )
            # index_select(positions, anchors), staged with its pad rows, + 1.
            anchor = tl.load(anchors_ptr + tl.minimum(r, scheduled_bs - 1), mask=live)
            pos = tl.load(positions_src_ptr + anchor, mask=live)
        else:
            pos = tl.load(dst_positions_ptr + r, mask=live)
        tl.store(dst_positions_ptr + r, pos + 1, mask=live)
        # _advance_decode_metadata's `context_lens += 1`, then
        # prepare_mtp_decode drops the rejected suffix of the real rows.
        ctx = tl.load(context_lens_ptr + r, mask=live, other=0) + 1
        if HAS_REJECT:
            ctx = ctx - tl.load(num_reject_ptr + r, mask=real, other=0).to(ctx.dtype)
        tl.store(context_lens_ptr + r, ctx, mask=live)
        # kv_indptr: 0, then the running page count of the real rows (pages of
        # the attention builder's block, as `prepare_mtp_decode` counts them).
        counts = tl.where(real, (ctx + KV_PAGE - 1) // KV_PAGE, 0)
        tl.store(
            kv_indptr_ptr + 1 + r,
            tl.cumsum(counts, axis=0).to(kv_indptr_ptr.dtype.element_ty),
            mask=live,
        )
        tl.store(kv_indptr_ptr + r, 0, mask=r == 0)
        # draft_kv_write_slots: the newest token's slot (in the runner's
        # blocks, which `block_tables` holds); -1 from num_slot_rows on.
        last = tl.maximum(ctx.to(tl.int64) - 1, 0)
        col = last // SLOT_PAGE
        page = tl.load(
            block_tables_ptr + r.to(tl.int64) * block_tables_stride + col,
            mask=r < num_slot_rows,
            other=0,
        ).to(tl.int64)
        slot = tl.where(
            r < num_slot_rows, page * SLOT_PAGE + (last - col * SLOT_PAGE), -1
        )
        tl.store(slot_mapping_ptr + r, slot, mask=live)
    else:
        rows = (pid - 1) * ROWS + tl.arange(0, ROWS)
        row_live = rows < running_bs
        src = tl.minimum(rows, scheduled_bs - 1)
        ids = tl.load(src_ids_ptr + src, mask=row_live)
        tl.store(dst_ids_ptr + rows, ids, mask=row_live)
        tl.store(
            draft_ptr + rows.to(tl.int64) * draft_stride + draft_col,
            ids,
            mask=rows < scheduled_bs,
        )
        cols = tl.arange(0, H)
        for k in tl.static_range(ROWS):
            row = (pid - 1) * ROWS + k
            if row < running_bs:
                s = tl.minimum(row, scheduled_bs - 1).to(tl.int64)
                x = tl.load(src_hidden_ptr + s * src_hidden_stride + cols)
                tl.store(
                    dst_hidden_ptr + row.to(tl.int64) * dst_hidden_stride + cols, x
                )


def _block(max_bs: int) -> int:
    # Program 0's lanes: every row of the deployment's largest batch (and the
    # `+ 1` of an indptr), so one compiled variant serves every batch.
    return max(16, triton.next_power_of_2(max_bs + 1))


def stage_verify_step0(
    input_ids: torch.Tensor,
    next_token_ids: torch.Tensor,
    anchors: torch.Tensor,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    staged: dict[str, torch.Tensor],
    *,
    scheduled_bs: int,
    running_bs: int,
    query_width: int,
    max_bs: int,
) -> dict[str, torch.Tensor]:
    """`EagleProposer` step 0's inputs on a uniform verify batch, one launch.

    The same writes as ``input_ids.scatter_(0, anchors, next_token_ids)``,
    ``positions + 1`` and `EagleProposer._stage_step0`. ``positions`` is the
    first (or only) axis, ``staged`` step 0's buffers at ``running_bs``;
    ``max_bs`` the largest batch the deployment runs. Returns ``staged``.
    """
    num_tokens = input_ids.shape[0]
    assert num_tokens == scheduled_bs * query_width, (num_tokens, scheduled_bs)
    assert 1 <= scheduled_bs <= running_bs <= max_bs
    s_ids = staged["input_ids"].view(-1)
    s_pos = staged["positions"].view(-1)
    s_hid = staged["hidden_states"]
    s_hid = s_hid.view(-1, s_hid.shape[-1])
    s_lti = staged["last_token_indices"]
    num_rows = running_bs * query_width
    hidden = hidden_states.shape[-1]
    assert s_hid.shape[0] == num_rows and s_hid.stride(-1) == 1
    assert hidden_states.stride(-1) == 1 and positions.stride(-1) == 1
    _stage_verify_step0_kernel[(triton.cdiv(num_rows, _COPY_ROWS_PER_PROGRAM),)](
        input_ids,
        next_token_ids,
        anchors,
        positions,
        hidden_states,
        hidden_states.stride(0),
        s_ids,
        s_pos,
        s_hid,
        s_hid.stride(0),
        s_lti,
        num_tokens,
        num_rows,
        scheduled_bs,
        running_bs,
        QUERY_WIDTH=query_width,
        H=hidden,
        ROWS=_COPY_ROWS_PER_PROGRAM,
        BLOCK_BS=_block(max_bs),
        num_warps=4,
    )
    return staged


def _launch_advance(
    *,
    src_ids,
    src_hidden,
    dst_ids,
    dst_hidden,
    dst_positions,
    draft_ids,
    draft_col,
    context_lens,
    num_reject_tokens,
    kv_indptr,
    block_tables,
    slot_mapping,
    cu_seqlens_q,
    row_ids,
    positions_src,
    anchors,
    scheduled_bs,
    running_bs,
    num_slot_rows,
    kv_page,
    slot_page,
):
    first = positions_src is not None
    assert 1 <= scheduled_bs <= running_bs < context_lens.shape[0] + 1
    assert src_hidden.stride(-1) == 1 and dst_hidden.stride(-1) == 1
    assert draft_ids.shape[0] == scheduled_bs and draft_ids.dim() == 2
    grid = (1 + triton.cdiv(running_bs, _COPY_ROWS_PER_PROGRAM),)
    _advance_draft_pass_kernel[grid](
        src_ids,
        src_hidden,
        src_hidden.stride(0),
        dst_ids,
        dst_hidden,
        dst_hidden.stride(0),
        dst_positions,
        draft_ids,
        draft_ids.stride(0),
        draft_col,
        context_lens,
        num_reject_tokens if num_reject_tokens is not None else context_lens,
        kv_indptr,
        block_tables,
        block_tables.stride(0),
        slot_mapping,
        cu_seqlens_q,
        row_ids,
        positions_src if first else dst_positions,
        anchors if first else dst_positions,
        scheduled_bs,
        running_bs,
        num_slot_rows,
        KV_PAGE=kv_page,
        SLOT_PAGE=slot_page,
        H=dst_hidden.shape[-1],
        FIRST=first,
        HAS_REJECT=num_reject_tokens is not None,
        BLOCK=_block(context_lens.shape[0]),
        ROWS=_COPY_ROWS_PER_PROGRAM,
        num_warps=4,
    )


def advance_draft_pass(
    builder,
    *,
    src_ids: torch.Tensor,
    src_hidden: torch.Tensor,
    dst_ids: torch.Tensor,
    dst_hidden: torch.Tensor,
    dst_positions: torch.Tensor,
    draft_ids: torch.Tensor,
    draft_col: int,
    scheduled_bs: int,
    running_bs: int,
    max_seqlen_k: int,
    num_slot_rows: int,
    num_reject_tokens: torch.Tensor | None = None,
    positions_src: torch.Tensor | None = None,
    anchors: torch.Tensor | None = None,
) -> dict:
    """Advance the one-row-per-sequence metadata a drafted token and stage the
    next pass, as `_enter_decode_metadata` (``positions_src`` given: the first
    advance after step 0) / `_advance_decode_metadata` with the GDN builder's
    `prepare_mtp_decode`, then the proposer's pad-row slot fill and
    `_stage_step_inputs`, would. Returns ``prepare_mtp_decode``'s workinfos.

    ``src_*`` are this pass's outputs (at least ``scheduled_bs`` rows),
    ``dst_*`` the next pass's buffers at ``running_bs``; ``draft_ids`` is the
    ``[scheduled_bs, mtp_k]`` result, column ``draft_col`` this pass's.
    ``num_slot_rows`` is ``scheduled_bs``, or 0 on a dummy run.
    """
    var = builder.model_runner.forward_vars
    kv_indptr = var["kv_indptr"].gpu[: running_bs + 1]
    block_tables = var["block_tables"].gpu
    slot_mapping = var["slot_mapping"].gpu[:running_bs]
    _launch_advance(
        src_ids=src_ids,
        src_hidden=src_hidden,
        dst_ids=dst_ids,
        dst_hidden=dst_hidden,
        dst_positions=dst_positions,
        draft_ids=draft_ids,
        draft_col=draft_col,
        context_lens=var["context_lens"].gpu,
        num_reject_tokens=num_reject_tokens,
        kv_indptr=kv_indptr,
        block_tables=block_tables,
        slot_mapping=slot_mapping,
        cu_seqlens_q=var["cu_seqlens_q"].gpu,
        row_ids=builder.row_ids,
        positions_src=positions_src,
        anchors=anchors,
        scheduled_bs=scheduled_bs,
        running_bs=running_bs,
        num_slot_rows=num_slot_rows,
        kv_page=builder.block_size,
        slot_page=builder.model_runner.block_size,
    )
    # The CSR the step's page list is laid out by is the one just written.
    kv_indices_generate_triton(
        block_tables[:running_bs],
        var["kv_indices"].gpu,
        kv_indptr,
        builder.block_ratio,
        max_seqlen_k,
    )
    return builder.finish_mtp_decode(running_bs, slot_mapping)


def warm_fused_glue_kernels(
    builder,
    *,
    hidden_size: int,
    hidden_dtype: torch.dtype,
    ids_dtype: torch.dtype,
    query_width: int,
    max_bs: int,
) -> None:
    """Compile every variant the glue launches (step 0's staging, the first and
    the later advances) on scratch tensors of the served dtypes, so the first
    decode step after startup compiles nothing. Writes only the scratch."""
    var = builder.model_runner.forward_vars
    dev = var["context_lens"].gpu.device
    i32 = {"dtype": torch.int32, "device": dev}
    i64 = {"dtype": torch.int64, "device": dev}
    rows = query_width
    ids = torch.zeros(rows + 1, dtype=ids_dtype, device=dev)
    hidden = torch.zeros(rows, hidden_size, dtype=hidden_dtype, device=dev)
    staged = {
        "input_ids": torch.zeros(1, rows, dtype=ids_dtype, device=dev),
        "positions": torch.zeros(1, rows, **i64),
        "hidden_states": torch.zeros(
            1, rows, hidden_size, dtype=hidden_dtype, device=dev
        ),
        "last_token_indices": torch.zeros(1, **i64),
    }
    anchors = torch.full((1,), rows - 1, **i32)
    stage_verify_step0(
        ids[1:],
        torch.zeros(1, dtype=ids_dtype, device=dev),
        anchors,
        torch.zeros(rows, **i64),
        hidden,
        staged,
        scheduled_bs=1,
        running_bs=1,
        query_width=query_width,
        max_bs=max_bs,
    )
    context_lens = torch.ones(var["context_lens"].gpu.shape[0], **i32)
    for first in (True, False):
        _launch_advance(
            src_ids=ids[:1],
            src_hidden=hidden[:1],
            dst_ids=torch.zeros(1, dtype=ids_dtype, device=dev),
            dst_hidden=torch.zeros(1, hidden_size, dtype=hidden_dtype, device=dev),
            dst_positions=torch.zeros(1, **i64),
            draft_ids=torch.zeros(1, 2, dtype=ids_dtype, device=dev),
            draft_col=0,
            context_lens=context_lens,
            num_reject_tokens=torch.zeros(1, **i32) if first else None,
            kv_indptr=torch.zeros(2, **i32),
            # Read only; the real table, so its row stride specializes alike.
            block_tables=var["block_tables"].gpu,
            slot_mapping=torch.zeros(1, **i64),
            cu_seqlens_q=torch.zeros(2, **i32),
            row_ids=builder.row_ids,
            positions_src=torch.zeros(rows, **i64) if first else None,
            anchors=anchors if first else None,
            scheduled_bs=1,
            running_bs=1,
            num_slot_rows=1,
            kv_page=builder.block_size,
            slot_page=builder.model_runner.block_size,
        )
