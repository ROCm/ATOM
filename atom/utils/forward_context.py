# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
import threading
from collections.abc import Container
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import torch

from atom.config import Config, KVCacheTensor, ParallelConfig

# A module, not its names: `region_graph` reads this one back at call time.
from atom.utils import region_graph

if TYPE_CHECKING:
    from atom.model_ops.indexer import IndexRows


class AttnState(Enum):
    """Attention dispatch state — controls which kv-indices buffers are built
    and which forward branch fires.

    Backends that distinguish only "decode vs prefill" can treat any
    ``PREFILL_*`` value as prefill. Backends with chunked-prefill awareness
    (e.g. V4) further distinguish ``PREFILL_NATIVE`` from ``PREFILL_PREFIX``.

    - ``DECODE``: 1+K tokens/seq uniformly (decode + spec). Per-token decode
      kv-indices buffers are valid; prefill prefix buffers may be stale.
    - ``PREFILL_NATIVE``: fresh prefill — every seq starts at position 0 in
      this fwd. No prior-chunk KV history to read; the prefix region is
      empty per token.
    - ``PREFILL_PREFIX``: chunked prefill — at least one seq has
      ``chunk_start > 0`` and therefore reads its prior chunk's KV from
      the paged history (e.g. V4 SWA ring via ``kv_indices_prefix_swa``).
    """

    DECODE = "decode"
    PREFILL_NATIVE = "prefill_native"
    PREFILL_PREFIX = "prefill_prefix"


@dataclass(frozen=True)
class DPMetadata:
    """The sizes the MoE DP collectives read off every rank's running tokens
    -- the rows each rank's forward runs."""

    running_tokens_per_dp_rank: np.ndarray

    @property
    def max_tokens_across_dp(self) -> int:
        return int(self.running_tokens_per_dp_rank.max())

    @staticmethod
    def gather_running_tokens(
        parallel_config: ParallelConfig, running_tokens: int
    ) -> np.ndarray:
        """Every DP rank's running tokens, from each rank's own, by one CPU
        all_reduce that a group of one skips. For a caller that knows only its
        own count; one that knows every rank's builds the table itself."""
        table = np.zeros(parallel_config.data_parallel_size, dtype=np.int32)
        table[parallel_config.data_parallel_rank] = running_tokens
        if parallel_config.data_parallel_size > 1:
            import torch.distributed as dist
            from aiter.dist.parallel_state import get_dp_group

            dist.all_reduce(torch.from_numpy(table), group=get_dp_group().cpu_group)
        return table

    def get_sizes_across_dp(self) -> list[int]:
        return self.running_tokens_per_dp_rank.tolist()


@dataclass
class SpecDecodeMetadata:
    # The step's input ids. Draft r is input_ids[target_logits_indices[r] + 1]:
    # logits row t verifies the token that follows it. Read in place by the
    # rejection sampler, on the forward stream the next step publishes its
    # input ids on, so they cannot be overwritten under it.
    input_ids: torch.Tensor
    num_spec_steps: int
    num_draft_tokens_np: np.ndarray
    cu_num_draft_tokens: torch.Tensor
    target_logits_indices: torch.Tensor
    bonus_logits_indices: torch.Tensor


_uncaptured_decode_shape_warned = False


def _warn_uncaptured_decode_shape(running_bs: int, max_seqlen_q: int) -> None:
    global _uncaptured_decode_shape_warned
    if _uncaptured_decode_shape_warned:
        return
    _uncaptured_decode_shape_warned = True
    _logger.warning(
        "No CUDA graph recorded at (bs=%d, q=%d): decode steps at an uncaptured "
        "shape run eagerly. Expected only on a P/D producer that keeps a drafter.",
        running_bs,
        max_seqlen_q,
    )


@dataclass(frozen=True)
class ForwardMode:
    """One step's shape, settled once, in the only two units ATOM has.

    ``running_bs`` sizes everything per-sequence (attention above all);
    ``running_tokens`` everything per-row (MoE above all). ``scheduled_*`` is
    this rank's own, and what a result slices back to.

    They are agreed across DP to different extents, and the difference is the
    point. ``running_bs`` is agreed on EVERY step -- it is the reduction on a
    ladder every rank shares -- because a captured graph holds its collective
    at that batch and a rank arriving with another one hangs the group.
    ``running_tokens`` is agreed exactly when ``running_tokens_are_unified``:
    off it, MoE takes the variable-length gather, whose contract is that the
    rows it was handed are the rows THIS rank scheduled, so a group height
    there would claim rows nobody allocated. An over-wide per-sequence array
    costs a few sentinel rows; an over-wide height is a contract break.

    Nothing downstream recomputes a width. ``running_bs * q`` is NOT
    ``running_tokens`` on a ragged step, and every site that rederived one from
    the other is a site where the two came to disagree.

    ``decide`` owns the whole chain -- the collective, adopting its reductions
    onto the batch, the ladder, both widths -- because the steps are not
    separable: the reduced query length rewrites the batch, and everything else
    is read off it afterwards.
    """

    use_cudagraph: bool
    is_prefill: bool
    scheduled_bs: int
    scheduled_tokens: int
    running_bs: int
    running_tokens: int
    running_tokens_are_unified: bool
    # Query rows per sequence this step. `num_spec_step + 1` unless a DSpark
    # shrink cut it, then DP-MAX'd -- so it is the group's, like the widths.
    max_seqlen_q: int
    # TBO's cross-DP AND -- on one rank its own answer, no peer vetoing it.
    tbo_collective_active: bool
    # What else the one per-step exchange returned (`DPSyncResult`): every
    # rank's row, a group of one's included -- per-rank token counts, the
    # DSpark shape max. Not shape.
    sync: Any
    # The step runs a recorded token-graph bucket (`running_tokens`), so its
    # model's `TokenGraph` replays (`atom/utils/region_graph.py`).
    token_graph: bool = False
    # The bucket it replays is a prefill one, on every rank of the group: a
    # rank decoding beside a prefilling peer replays the prefill recording too
    # (`_pad_to_token_graph`, `region_graph._phase`).
    prefill_token_graph: bool = False
    # The DP group replays request graphs as one this step: a decode's whole
    # forward (`use_cudagraph`), a prefill's decoder (`DecoderGraph`).
    req_graphs: bool = False
    # The decode batch this step's TBO micro-batches are cut from
    # (`decode_ubatch_request_ranges`), and so the one every backend lays its
    # per-ubatch metadata out for: the padded bucket when the step replays a
    # TBO graph, which recorded that split; the scheduled requests on an eager
    # TBO split. None when the step does not split a decode.
    ubatch_split_bs: int | None = None

    @classmethod
    def decide(
        cls,
        *,
        batch,
        dp_size: int,
        dp_group,
        enforce_eager: bool,
        step_needs_eager: bool,
        capture_sizes: list[int],
        is_block_drafter: bool,
        tbo_on: bool,
        local_tbo: tuple[bool, bool, int, int],
        max_seqlen_q: int,
        running_tokens_per_dp_rank: np.ndarray,
        graph_shapes: Container[tuple[int, int]] | None = None,
        tbo_decode_graphs: bool = False,
    ) -> "ForwardMode":
        """Run the step's DP collective and settle its shape from the result.

        ``tbo_decode_graphs``: every decode bucket that splits
        (`decode_bucket_splits`) was recorded as a TBO graph, so a decode the
        group splits replays like any other.

        ``running_tokens_per_dp_rank`` is the caller's [dp_size] buffer, which
        every step settles in place from the group's scheduled tokens.

        Any new force-eager condition belongs here, not in a caller-side check.
        `step_needs_eager` is this rank's attention backend refusing a graph for
        the step (`AttentionMetadataBuilder.step_needs_eager`); the DP group
        agrees on it.
        """
        # Lazy: `atom.utils.tbo`'s package init reaches back into this module.
        from atom.utils.tbo.ubatch_splitting import decode_bucket_splits
        from atom.utils.tbo.ubatching import sync_dp_metadata

        is_prefill = batch.total_tokens_num_prefill > 0
        scheduled_tokens = batch.total_tokens_num
        # The whole batch, never a per-kind count: `schedule` returns its
        # prefill batch before it can add a decode row, so the two are equal
        # today, and this is the one that survives a mixed batch.
        scheduled_bs = batch.total_seqs_num
        meets_min, can_split, ub0, ub1 = local_tbo
        # Every rank's row of the step, a group of one's included (its own
        # row, with nothing exchanged).
        sync = sync_dp_metadata(
            dp_group=dp_group,
            dp_size=dp_size,
            scheduled_tokens=scheduled_tokens,
            scheduled_bs=scheduled_bs,
            is_prefill=is_prefill,
            tbo_on=tbo_on,
            needs_eager=step_needs_eager,
            final_requests=batch.final_prefill_requests(),
            local_meets_min_tokens=meets_min,
            local_can_split=can_split,
            local_ub_tokens=(ub0, ub1),
            # Only a block drafter needs the group to agree on it; every other
            # flavor keeps its own and the extra wire row is not sent.
            max_seqlen_q=max_seqlen_q if is_block_drafter else None,
        )
        # The group's query length, taken BEFORE the rows are read off it: a
        # rank still on its local q settles on a different `running_tokens`
        # than its peers.
        if sync.max_seqlen_q_across_dp is not None:
            max_seqlen_q = sync.max_seqlen_q_across_dp
        unified_bs = sync.max_bs_across_dp
        tbo_collective_active = sync.tbo_collective_active
        # One meaning and only one: every rank runs one height. Every rank
        # decoding, the group max pads them level and `scheduled_tokens`
        # reduces to a decode count on every rank; a prefilling peer makes the
        # step ragged, and each rank runs its own count through the
        # variable-length gather. A group of one has one height whatever it runs.
        unified = dp_size == 1 or not sync.any_rank_has_prefill
        # A graph replays its collectives: the group replays or runs eagerly as
        # one.
        step_needs_eager = sync.any_rank_needs_eager

        # ---- the batch: one rule, every step ---------------------------
        # `unified_bs` is the reduction's own answer and the ladder is a rule
        # every rank shares, so this is DP-agreed by construction -- not by a
        # case analysis a future branch could get wrong. That is the whole
        # property: a graph key, a draft pass and an attention plan all read
        # this one field and cannot end up holding two widths.
        #
        # One lookup answers both of the step's questions. `capture_sizes` is
        # ascending int32 (`ModelRunner.capture_sizes_np`, sorted where it is
        # built and never mutated), so the insertion point IS the smallest size
        # that holds the batch, and running off the end IS "no recording can".
        rung = int(np.searchsorted(capture_sizes, unified_bs))
        on_ladder = rung < len(capture_sizes)
        running_bs = int(capture_sizes[rung]) if on_ladder else unified_bs

        # ---- dispatch: a different question, asked separately ----------
        # Whether the TARGET replays. It no longer decides how wide anything
        # is; when the batch answered four ways depending on this, `running_bs`
        # was per-rank on two of them and every consumer had to know which.
        #
        # The ladder answers the batch question alone, but a recording is keyed
        # by its request bucket (`request_bucket`). Only a P/D producer passes
        # `graph_shapes`: it keeps its drafter, so capture holds the
        # draft_num=mtp_k buckets, while the scheduler never speculates for it
        # and every decode it runs has no draft. That holds on every rank, its DP dummy included, and
        # `running_bs` is the group's, so the lookup is DP-agreed: the group
        # replays or runs eagerly as one.
        # A step the group splits into TBO microbatches has no recording of its
        # own, and `tbo_collective_active` is the group's: it runs eagerly as
        # one -- unless it is a decode and the buckets were recorded split
        # (`tbo_decode_graphs`), where the bucket's recording is the split.
        tbo_eager = tbo_collective_active and (is_prefill or not tbo_decode_graphs)
        graphable = not (enforce_eager or step_needs_eager or tbo_eager)
        # A recording holds every rank's rows in its collectives, so the group
        # replays request graphs as one, on a step every rank runs alike: all
        # decoding (`unified`) or all prefilling.
        req_graphs = graphable and (unified or all(sync.prefill_across_dp))
        replayable = req_graphs and not is_prefill and on_ladder
        shape_captured = (
            graph_shapes is None
            or request_bucket(running_bs, max_seqlen_q) in graph_shapes
        )
        if replayable and not shape_captured:
            _warn_uncaptured_decode_shape(running_bs, max_seqlen_q)
        use_cudagraph = replayable and shape_captured
        # A replayed bucket that splits was recorded split at the bucket, on
        # every rank alike, whatever each rank scheduled into it; an eager
        # split cuts this rank's scheduled requests.
        if use_cudagraph and tbo_decode_graphs and decode_bucket_splits(running_bs):
            ubatch_split_bs = running_bs
        elif tbo_collective_active and not is_prefill:
            ubatch_split_bs = scheduled_bs
        else:
            ubatch_split_bs = None

        running_tokens, decode_recorded = cls._running_tokens(
            is_prefill=is_prefill,
            unified=unified,
            running_bs=running_bs,
            scheduled_tokens=scheduled_tokens,
            scheduled_tokens_per_dp_rank=sync.scheduled_tokens_per_dp_rank,
            is_block_drafter=is_block_drafter,
            max_seqlen_q=max_seqlen_q,
        )
        token_graph = cls._pad_to_token_graph(
            graphable, sync, running_tokens_per_dp_rank
        )
        if token_graph:
            running_tokens = token_graph
        if unified:
            # Every rank runs the group's height.
            running_tokens_per_dp_rank.fill(running_tokens)
        return cls(
            use_cudagraph=use_cudagraph,
            is_prefill=is_prefill,
            scheduled_bs=scheduled_bs,
            scheduled_tokens=scheduled_tokens,
            running_bs=running_bs,
            running_tokens=running_tokens,
            running_tokens_are_unified=unified,
            max_seqlen_q=max_seqlen_q,
            tbo_collective_active=tbo_collective_active,
            sync=sync,
            token_graph=bool(token_graph) or (decode_recorded and graphable),
            prefill_token_graph=bool(token_graph),
            req_graphs=req_graphs,
            ubatch_split_bs=ubatch_split_bs,
        )

    @staticmethod
    def _pad_to_token_graph(graphable, sync, running_tokens_per_dp_rank) -> int:
        """Settle every rank's running tokens into ``running_tokens_per_dp_rank``
        -- its scheduled ones, or on a step some rank prefills every rank's
        raised to the prefill bucket holding the group's largest
        (`TokenGraph.bucket`) -- and return that bucket, 0 for none.

        One bucket for every rank, a decoding one and an idle rank's dummy step
        included: a recording's collectives are sized by its bucket and hold
        the peer buffers the group recorded it with, so a rank replaying
        another bucket, or running eagerly beside a replay, hangs the group. A
        decoding rank replays the prefill recording on its decode rows
        (`ForwardMode.prefill_token_graph`); its attention, an eager region,
        still runs the decode it scheduled.
        """
        running = running_tokens_per_dp_rank
        np.copyto(running, sync.scheduled_tokens_per_dp_rank)
        if not graphable or not any(sync.prefill_across_dp):
            return 0
        group = region_graph.TokenGraph.bucket(int(running.max()), "prefill")
        if group is None:
            return 0
        running.fill(group)
        return group

    @staticmethod
    def _running_tokens(
        *,
        is_prefill: bool,
        unified: bool,
        running_bs: int,
        scheduled_tokens: int,
        scheduled_tokens_per_dp_rank,
        is_block_drafter: bool,
        max_seqlen_q: int,
    ) -> tuple[int, bool]:
        """``(running_tokens, recorded)``: the decode's width, and whether a
        decode token graph is recorded at it (`TokenGraph.captured_sizes`).

        `pad_for_all_gather` asserts the first equals the rows the tensor really
        has, so it is not "the biggest count anyone reported": it is what THIS
        forward will produce.
        """
        if not unified or is_prefill:
            # Nothing padded the tensor -- the varlen gather sends each rank its
            # own count, and a prompt is ragged, so no product recovers a
            # height -- and this rank's own count is the whole answer.
            return scheduled_tokens, False

        # The batch was rounded onto the ladder AFTER the sync, so this runs rows
        # no rank reported; group-agreed all the same, both factors having come
        # out of the reduction.
        q = max_seqlen_q
        running_tokens = running_bs * q
        captured = region_graph.TokenGraph.captured_sizes("decode")
        if not captured:
            return running_tokens, False
        if not is_block_drafter:
            # Rectangular layout: one row per (seq, query), which every per-token
            # buffer spans and `slot_mapping[i * q + j]` addresses through. Anything
            # under the product would leave the tail of that rectangle unwritten.
            return running_tokens, running_tokens in captured

        # Marker-driven layout: the rows are one flat run, so the width is the
        # smallest recorded one that holds it, not a product -- q-divisible
        # because the per-seq structures tile the run at that stride. Every rank
        # is decoding here, so `scheduled_tokens` IS a decode count on each of
        # them and row 0's reduction is the group's run.
        run = int(scheduled_tokens_per_dp_rank.max())
        recorded = region_graph.TokenGraph.bucket(run, "decode", multiple_of=q)
        if recorded is None:
            # Nothing recorded holds the run, so forward it eagerly rather than
            # claim a graph nobody made at this width.
            return max(run, running_tokens), False
        return recorded, True

    def assert_shape_contract(
        self,
        input_ids: "torch.Tensor",
        attn_metadata: "AttentionMetaData",
    ) -> None:
        """Tie what the builder published to what this step settled.

        Covers the PADDED step, which is the only one that can be wrong: a
        width is either both units' or neither's, and every way they have come
        apart -- a builder on a stale batch, a per-seq array widened while the
        per-token ones were not -- shows up as one of the equalities below. The
        version that skipped whenever ``running_bs != scheduled_bs`` exempted
        exactly that case, so a mismatch reached the model and surfaced as lost
        accuracy rather than as a failure here.

        Shapes only, never values: reading a device tensor's contents on this
        path is a D2H sync per step.

        Prefill is still out. Its rows are the prompt's, so no product of a
        batch and a query length describes them, and its per-seq tail is padded
        for a following draft pass rather than for anything this step runs.
        """
        if self.is_prefill or input_ids is None or attn_metadata is None:
            return

        def _rows(name):
            t = getattr(attn_metadata, name, None)
            return None if t is None else int(t.shape[0])

        # `input_ids` is the scheduled prefix at the runner boundary. Uniform
        # decode exposes `running_tokens` to the model in both eager and graph
        # execution; a mixed prefill/decode step keeps its local token count.
        assert input_ids.shape[0] == self.scheduled_tokens, (
            f"input_ids length {input_ids.shape[0]} != scheduled_tokens="
            f"{self.scheduled_tokens} ({self})"
        )
        assert self.scheduled_tokens <= self.running_tokens, (
            f"running_tokens={self.running_tokens} is below the rows this rank "
            f"scheduled ({self.scheduled_tokens}) ({self})"
        )
        # Per-token: sized by `running_tokens`, the height everything per-row runs.
        slot_rows = _rows("slot_mapping")
        assert slot_rows in (None, self.running_tokens), (
            f"slot_mapping length {slot_rows} != running_tokens="
            f"{self.running_tokens} ({self}); the builder used another width"
        )
        # Per-sequence: sized by `running_bs`. `cu_seqlens_q` carries the extra
        # boundary entry; the state slots are V4's and absent elsewhere.
        cu_rows = _rows("cu_seqlens_q")
        assert cu_rows in (None, self.running_bs + 1), (
            f"cu_seqlens_q length {cu_rows} != running_bs+1="
            f"{self.running_bs + 1} ({self})"
        )
        slot_out_rows = _rows("state_slot_out")
        assert slot_out_rows in (None, self.running_bs), (
            f"state_slot_out length {slot_out_rows} != running_bs="
            f"{self.running_bs} ({self})"
        )
        # No rectangle assertion: `_running_tokens` deliberately returns a
        # height ABOVE `running_bs * max_seqlen_q` when nothing recorded holds
        # the run.


def running_tokens_from_bs(bs: int, *, is_prefill: bool, attn_metadata) -> int:
    """``running_tokens`` for a BRIDGE that only knows a request count.

    Native ATOM never calls this -- it reads `ForwardMode.running_tokens`. A
    plugin bridge has no ForwardMode, so it reconstructs the rectangle here and
    nowhere else.

    Prefill returns the count verbatim, which is NOT a height: prompt lengths
    are ragged, so no multiplier recovers one, and a number below the real
    height just leaves the batch unpadded -- correct there, since a prefill
    all_gather is variable-length.
    """
    if is_prefill or attn_metadata is None:
        return bs
    return bs * int(attn_metadata.max_seqlen_q)


@dataclass
class Context:
    # This context is used to store the basic context of the forward.
    positions: torch.Tensor
    is_prefill: bool = False
    is_dummy_run: bool = False
    # What this rank was handed. Duplicated from `forward_mode` because a
    # capture context has none; `scheduled_tokens` counts this rank's real rows.
    scheduled_bs: int = 0
    scheduled_tokens: int = 0
    # The step's DP-unified padded shape. `running_bs` counts SEQUENCES (graph
    # identity, the draft's pad width), `running_tokens` the hidden_states rows
    # MoE pads to. Both stored, because the ratio is not always max_seqlen_q --
    # a DSpark ragged step runs a packed width no rectangular bs*q recovers.
    running_bs: int = 0
    running_tokens: int = 0
    is_draft: bool = False
    # Every rank of the group is decoding, so `running_tokens` above is the
    # group's number and not this rank's own. That is what picks MoE's
    # fixed-size all_gather over the variable-length one, and so what makes a
    # captured graph usable at all. One meaning, deliberately: it once also
    # stood for "dp-attention is off", and readers needing the strong claim --
    # `_can_use_dp_sharded_head` above all -- silently got the weak one.
    running_tokens_are_unified: bool = True
    # `running_tokens` for every rank, unreduced, as Python ints
    # (`set_forward_context`'s table); this rank's entry IS `running_tokens`.
    # Consumers reduce it, because they want different reductions: padding
    # each rank up to the group is a MAX, an all2all's receive width holds what
    # every rank sent and is a SUM. None on a TBO ubatch whose counts were not
    # exchanged.
    running_tokens_across_dp: tuple[int, ...] | None = None
    # The step's whole shape decision. Set by `prepare_model` via
    # `ForwardMode.decide`; None only on a capture context, which declares its
    # shape rather than reading one.
    forward_mode: ForwardMode | None = None
    # Optional flat token ids for the current forward. Read by callbacks
    # invoked inside Dynamo-opaque custom ops (e.g. V4 MoE hash routing)
    # that need the token ids but cannot receive them as a function arg
    # (the op signature is fixed by the consumer's plugin contract).
    input_ids: torch.Tensor | None = None
    # Row offset of this (micro-)batch's tokens on the full forward's token
    # axis. 0 except inside a TBO micro-batch. Anything writing into a
    # whole-forward buffer from inside the model must add it, or the ubatches
    # land on each other's rows. Read per-thread (the context is thread-local).
    ubatch_token_offset: int = 0
    # This pass's rows are this rank's round-robin 1/pcp shard of the tokens.
    # Set where the split is made; FusedMoE reads it to gather its routed rows
    # across the PCP group (`ATOM_PCP_MOE_MERGE`).
    pcp_split: bool = False
    # Whether this step's TARGET forward row-split, fixed by ModelRunner right
    # after it. Unlike `pcp_split` it survives drafting, which clears that per
    # pass; the drafter's aux gather and draft prefill split read this one.
    target_pcp_split: bool = False
    # `scheduled_tokens` on the device (`[1]` int32), an entry of the step's
    # `forward_vars["batch_info"]` read in place, so a replay reads its own
    # step whatever shape it was recorded at. A draft re-points it with
    # `scheduled_tokens` (`Drafter._publish_draft_shape`).
    scheduled_tokens_gpu: torch.Tensor | None = None

    # Optional speculative-decoding inputs staged alongside the target
    # forward's other host-to-device copies. Keeping these on the per-forward
    # context avoids launching pinned-buffer H2Ds later from postprocess.
    draft_anchor_overrides: torch.Tensor | None = None

    def __init__(
        self,
        positions: torch.Tensor,
        is_prefill: bool = False,
        is_dummy_run: bool = False,
        scheduled_bs: int = 0,
        scheduled_tokens: int = 0,
        running_bs: int = 0,
        running_tokens: int = 0,
        is_draft: bool = False,
        running_tokens_are_unified: bool = True,
        running_tokens_across_dp: tuple[int, ...] | None = None,
        forward_mode: ForwardMode | None = None,
        input_ids: torch.Tensor | None = None,
        ubatch_token_offset: int = 0,
        draft_anchor_overrides: torch.Tensor | None = None,
        pcp_split: bool = False,
        target_pcp_split: bool = False,
        scheduled_tokens_gpu: torch.Tensor | None = None,
    ):
        self.positions = positions
        self.is_prefill = is_prefill
        self.is_dummy_run = is_dummy_run
        self.scheduled_bs = scheduled_bs
        self.scheduled_tokens = scheduled_tokens
        self.running_bs = running_bs
        self.running_tokens = running_tokens
        self.is_draft = is_draft
        self.running_tokens_are_unified = running_tokens_are_unified
        self.running_tokens_across_dp = running_tokens_across_dp
        self.forward_mode = forward_mode
        self.input_ids = input_ids
        self.ubatch_token_offset = ubatch_token_offset
        self.draft_anchor_overrides = draft_anchor_overrides
        self.pcp_split = pcp_split
        self.target_pcp_split = target_pcp_split
        self.scheduled_tokens_gpu = scheduled_tokens_gpu


@dataclass
class AttentionMetaData:
    """Attention metadata for prefill and decode batched together."""

    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    min_seqlen_q: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    dropout_p: float = 0.0
    # True for standard causal attention; False only for DSpark's bidirectional
    # draft block. The MLA asm decode kernel selects a different .co by this flag.
    causal: bool = True

    state: AttnState = AttnState.PREFILL_NATIVE
    """One of `DECODE / PREFILL_NATIVE / PREFILL_PREFIX` — controls which
    kv-indices buffers downstream forward branches read. Default is
    `PREFILL_NATIVE`; every `prepare_*` path overrides explicitly.
    Backends that don't need the NATIVE/PREFIX distinction can treat
    `any PREFILL_*` as prefill. See ``AttnState`` for full semantics."""

    kv_indptr: torch.Tensor | None = None
    kv_indices: torch.Tensor | None = None
    qo_indptr: torch.Tensor | None = None
    kv_last_page_lens: torch.Tensor | None = None
    cu_seqlen_ks: torch.Tensor | None = None
    cu_seqlen_ke: torch.Tensor | None = None
    sparse_kv_indptr: torch.Tensor | None = None
    # Last-page lens for sparse (DSA) attention: all 1s, one per query token in
    # prefill/MTP-verify and per seq in decode. Separate from kv_last_page_lens
    # (the dense per-seq buffer) so the two never clobber each other.
    sparse_kv_last_page_lens: torch.Tensor | None = None
    # Per-rank KV length of each query token's causal window, for the sparse
    # DSA + DCP indexer: MTP verify gives each draft position its own window.
    # The padded running-width buffer is shared by eager, graph and TBO paths.
    dcp_local_context_lens: torch.Tensor | None = None
    # Whether this prefill runs sparse (DSA) MLA, decided once a step by the
    # attention builder: the indexer, MLA and PCP all follow it.
    run_sparse: bool = False
    # The step's index query rows as `score_topk` scores them, published once
    # a step by the attention builder for every indexer layer.
    index_rows: "IndexRows | None" = None
    # The same rows for a pooled indexer's MTP layer, whose positions run one
    # behind the step's, so its rows close their pools a token later.
    mtp_index_rows: "IndexRows | None" = None
    # Block-table row per query token, for an indexer that reads its keys a
    # query row at a time (DCP decode, a pooled indexer's verify). Aliases
    # block_tables at one query per sequence.
    # Every producer that rewrites block_tables must rewrite this too: a stale
    # one has the right length and the wrong rows.
    token_block_tables: torch.Tensor | None = None

    work_meta_data: torch.Tensor | None = None
    work_indptr: torch.Tensor | None = None
    work_info_set: torch.Tensor | None = None
    reduce_indptr: torch.Tensor | None = None
    reduce_final_map: torch.Tensor | None = None
    reduce_partial_map: torch.Tensor | None = None

    # for prefix cache
    has_cached: bool = False
    total_kv: int | None = None
    num_cached_tokens: torch.Tensor | None = None
    seq_starts: torch.Tensor | None = None

    def __init__(
        self,
        cu_seqlens_q: torch.Tensor | None = None,
        cu_seqlens_k: torch.Tensor | None = None,
        max_seqlen_q: int = 0,
        max_seqlen_k: int = 0,
        min_seqlen_q: int = 0,
        slot_mapping: torch.Tensor | None = None,
        context_lens: torch.Tensor | None = None,
        block_tables: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        causal: bool = True,
        state: AttnState = AttnState.PREFILL_NATIVE,
        kv_indptr: torch.Tensor | None = None,
        kv_indices: torch.Tensor | None = None,
        qo_indptr: torch.Tensor | None = None,
        kv_last_page_lens: torch.Tensor | None = None,
        cu_seqlen_ks: torch.Tensor | None = None,
        cu_seqlen_ke: torch.Tensor | None = None,
        sparse_kv_indptr: torch.Tensor | None = None,
        sparse_kv_last_page_lens: torch.Tensor | None = None,
        dcp_local_context_lens: torch.Tensor | None = None,
        work_meta_data: torch.Tensor | None = None,
        work_indptr: torch.Tensor | None = None,
        work_info_set: torch.Tensor | None = None,
        reduce_indptr: torch.Tensor | None = None,
        reduce_final_map: torch.Tensor | None = None,
        reduce_partial_map: torch.Tensor | None = None,
        sparse_cu_seqlens_q: torch.Tensor | None = None,
        batch_id_per_q_token: torch.Tensor | None = None,
        batch_id_per_k_token: torch.Tensor | None = None,
        has_cached: bool = False,
        total_kv: int | None = None,
        num_cached_tokens: torch.Tensor | None = None,
        seq_starts: torch.Tensor | None = None,
    ):
        self.has_cached = has_cached
        self.total_kv = total_kv
        self.num_cached_tokens = num_cached_tokens
        self.seq_starts = seq_starts
        self.cu_seqlens_q = cu_seqlens_q
        self.cu_seqlens_k = cu_seqlens_k
        self.max_seqlen_q = max_seqlen_q
        self.max_seqlen_k = max_seqlen_k
        self.min_seqlen_q = min_seqlen_q
        self.slot_mapping = slot_mapping
        self.context_lens = context_lens
        self.block_tables = block_tables
        self.dropout_p = dropout_p
        self.causal = causal
        self.state = state
        self.kv_indptr = kv_indptr
        self.kv_indices = kv_indices
        self.qo_indptr = qo_indptr
        self.kv_last_page_lens = kv_last_page_lens
        self.cu_seqlen_ks = cu_seqlen_ks
        self.cu_seqlen_ke = cu_seqlen_ke
        self.sparse_kv_indptr = sparse_kv_indptr
        self.sparse_kv_last_page_lens = sparse_kv_last_page_lens
        self.dcp_local_context_lens = dcp_local_context_lens
        self.work_meta_data = work_meta_data
        self.work_indptr = work_indptr
        self.work_info_set = work_info_set
        self.reduce_indptr = reduce_indptr
        self.reduce_final_map = reduce_final_map
        self.reduce_partial_map = reduce_partial_map
        self.sparse_cu_seqlens_q = sparse_cu_seqlens_q
        self.batch_id_per_q_token = batch_id_per_q_token
        # [total_kv] int32 — the KV-axis twin, over the gathered cached+new KV.
        # Layer-invariant, so the prefill builder makes it once per fwd.
        self.batch_id_per_k_token = batch_id_per_k_token


def get_published_dcp_local_context_lens(
    attn_metadata: AttentionMetaData, num_rows: int
) -> torch.Tensor | None:
    """Return the valid prefix of a published sparse-DCP local-length buffer.

    Metadata is allocated at the engine's padded running width while TR_HYBRID
    eager execution may score only the scheduled rows. The real requests are
    always the prefix and the producer zero-fills the padded tail.
    """
    local_ctx = attn_metadata.dcp_local_context_lens
    if local_ctx is None or local_ctx.shape[0] < num_rows:
        return None
    if local_ctx.shape[0] == num_rows:
        return local_ctx
    return local_ctx[:num_rows]


@dataclass
class GDNAttentionMetadata:
    num_prefills: int
    num_prefill_tokens: int
    num_decodes: int
    num_decode_tokens: int
    num_spec_decodes: int
    num_spec_decode_tokens: int
    num_actual_tokens: int

    has_initial_state: torch.Tensor | None = None

    spec_query_start_loc: torch.Tensor | None = None  # shape: [num_spec_decodes + 1,]
    non_spec_query_start_loc: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes + 1,]
    )

    spec_state_indices_tensor: torch.Tensor | None = None  # shape: [batch, num_spec]
    non_spec_state_indices_tensor: torch.Tensor | None = (
        None  # shape: [batch - num_spec_decodes,]
    )
    spec_sequence_masks: torch.Tensor | None = None  # shape: [batch,]
    spec_token_indx: torch.Tensor | None = None
    non_spec_token_indx: torch.Tensor | None = None

    num_accepted_tokens: torch.Tensor | None = None  # shape: [batch,]

    # The following attributes are for triton implementation of causal_conv1d
    nums_dict: dict | None = None
    batch_ptr: torch.Tensor | None = None
    token_chunk_offset_ptr: torch.Tensor | None = None


@dataclass(frozen=True)
class DPStage:
    """MoE layers a forward runs on other rows than its step: every DP rank's
    token count for them and the `DPMetadata` the variable-length gather
    reads while they run (`ForwardContext.use_dp_stage`)."""

    tokens_across_dp: tuple[int, ...]
    local_tokens: int
    dp_metadata: DPMetadata

    @classmethod
    def make(cls, parallel_config, tokens_across_dp) -> "DPStage":
        tokens = tuple(int(n) for n in tokens_across_dp)
        return cls(
            tokens,
            tokens[parallel_config.data_parallel_rank],
            DPMetadata(np.array(tokens, dtype=np.int32)),
        )


def request_bucket(running_bs: int, max_seqlen_q: int) -> tuple[int, int]:
    """A request bucket (`ReqGraph`): the requests a step runs, and the draft
    tokens each verifies -- its query rows less the one it decodes."""
    return running_bs, max_seqlen_q - 1


@dataclass
class ForwardContext:
    # copy from vllm_config.compilation_config.static_forward_context
    no_compile_layers: dict[int, Any] = field(default_factory=dict)

    attn_metadata: Union["AttentionMetaData", dict[str, "AttentionMetaData"]] | None = (
        None
    )

    kv_cache_data: dict[str, KVCacheTensor] = None

    context: Context | None = None

    dp_metadata: DPMetadata | None = None

    spec_decode_metadata: SpecDecodeMetadata | None = None

    ubatch_slices: list[Any] | None = None

    # Cross-DP MAX of per-ubatch token counts, reduced in the step's one packed
    # all_gather (``ForwardMode.decide``) and propagated here so
    # ``UBatchWrapper._compute_ub_graph_bs`` no longer needs its own
    # per-ubatch all_reduce. Shape: tuple of length N == len(ubatch_slices).
    # None when TBO is not active this step.
    ub_max_tokens_across_dp: tuple | None = None
    # The same counts UNREDUCED: per ubatch, every rank's. The MAX above is one
    # reduction of it and an all2all's receive width wants the SUM, so consumers
    # take their own. Same None conditions as the MAX.
    ub_tokens_across_dp: tuple | None = None

    # Cached current_stream() captured at set_forward_context() time, so
    # downstream code (V4 attention / MoE / metadata builder) doesn't have
    # to query torch.cuda.current_stream() repeatedly during a forward —
    # multiple call sites caching independent Stream handles was widening
    # the hipStream handle pool and complicating reasoning about which
    # logical stream each wait_stream() refers to. CG capture / TBO
    # threads each call set_forward_context() inside their own stream
    # context, so the cached value is correct for the captured graph or
    # active thread.
    main_stream: torch.cuda.Stream | None = None

    # True only while the model forward runs inside a CUDAGraph capture
    # block (model_runner.capture_model loop). Components that gate
    # multi-stream side-launches (V4 main Compressor on alt_stream,
    # indexer.compressor on indexer_stream) check this flag: side-stream
    # work is safe to emit inside a captured graph (graph records the
    # fork-join edges and replay re-uses the same stream layout) but
    # racy in eager mode where launches accumulate across layers and
    # deadlock the hipStream queue. Replay does not re-execute Python
    # forward, so it ignores the flag entirely.
    in_hipgraph: bool = False

    # TR graphs (`atom/utils/region_graph.py`): "record" while the runner
    # records a token bucket, "record_req" while it records a prefill's request
    # bucket (`ModelRunner._record_prefill_req_graphs`), "replay" while a step
    # replays a token bucket, None everywhere else -- a module with no
    # recording for its rows then runs as it always did.
    tr_graph: str | None = None

    def __post_init__(self):
        if not hasattr(self, "no_compile_layers") or self.no_compile_layers is None:
            self.no_compile_layers = {}
        if self.attn_metadata is None:
            self.attn_metadata = {}

    @contextmanager
    def use_dp_stage(self, stage: DPStage | None):
        """Size the DP MoE collectives for ``stage`` while this block runs.

        The variable-length gather reads every rank's count from
        ``dp_metadata`` and this rank's own from ``context.scheduled_tokens``;
        both are switched to the stage's and restored on exit. A no-op for
        None, so a caller outside DP runs unchanged.
        """
        if stage is None:
            yield
            return
        context = self.context
        outer = (
            self.dp_metadata,
            context.scheduled_tokens,
            context.running_tokens_across_dp,
        )
        self.dp_metadata = stage.dp_metadata
        context.scheduled_tokens = stage.local_tokens
        context.running_tokens_across_dp = stage.tokens_across_dp
        try:
            yield
        finally:
            (
                self.dp_metadata,
                context.scheduled_tokens,
                context.running_tokens_across_dp,
            ) = outer

    @contextmanager
    def use_pass_rows(
        self, running_tokens: int, scheduled_tokens_gpu: torch.Tensor | None
    ):
        """Run this block as a pass of its own over other rows than its step's
        -- a decoder over the step's tails: ``running_tokens`` rows, the
        leading ``scheduled_tokens_gpu`` of them real
        (`pass_scheduled_tokens_gpu`). Both are restored on exit."""
        context = self.context
        outer = context.running_tokens, context.scheduled_tokens_gpu
        context.running_tokens = running_tokens
        context.scheduled_tokens_gpu = scheduled_tokens_gpu
        try:
            yield
        finally:
            context.running_tokens, context.scheduled_tokens_gpu = outer


_forward_context: ForwardContext | None = ForwardContext()
_forward_kv_cache_context: ForwardContext | None = ForwardContext()

# Cached once at module import — CUDA availability does not change at
# runtime, so we don't pay torch.cuda.is_available() per set_forward_context().
_CUDA_AVAILABLE: bool = torch.cuda.is_available()


# Thread-local storage for TBO dual-thread execution

_forward_context_local = threading.local()


def get_forward_context() -> ForwardContext:
    """Get the current forward context."""
    # Check thread-local first (used by TBO threads)
    ctx = getattr(_forward_context_local, "ctx", None)
    if ctx is not None:
        return ctx

    assert _forward_context is not None, (
        "Forward context is not set. "
        "Please use `set_forward_context` to set the forward context."
    )
    return _forward_context


def pass_scheduled_tokens_gpu(rows: int) -> torch.Tensor | None:
    """`Context.scheduled_tokens_gpu` for a consumer of `rows` rows, or None.
    Only the pass's whole rows (`running_tokens`) are counted by it; rows of
    another pass declare it (`ForwardContext.use_pass_rows`, as a decoder over
    the step's tails does), and any other subset -- a PCP shard -- gets None
    and runs every row, as the MoE sort reads the count as its row count and
    would run past a subset's rows. None too outside an ATOM forward."""
    context = get_forward_context().context
    if context is None or rows != context.running_tokens:
        return None
    return context.scheduled_tokens_gpu


@contextmanager
def side_stream(stream: torch.cuda.Stream | None):
    """Issue the block beside the main stream instead of on it.

    Yields `(issuing, joining)`: the stream the block runs on, and the one to
    wait on it afterwards, `None` when nothing forked.

    Forks only inside the capture loop, the one window where a side stream is
    both safe and useful: eager launches pile up across layers with nothing to
    drain them, and a replay runs no Python, inheriting the recorded layout.
    `in_hipgraph` covers that loop's warmup forward too, which matters because
    AITER caches a kernel's scratch per `(device, stream)` and rejects a first
    allocation made during capture.

    The gate lives here rather than at each caller so that two branches of one
    layer cannot drift into disagreeing about when forking is allowed. A caller
    whose feature is switched off hands `None` and takes the same path, so the
    switch needs no second branch anywhere.
    """
    context = get_forward_context()
    if stream is None or not context.in_hipgraph:
        yield context.main_stream, None
        return
    stream.wait_stream(context.main_stream)
    with torch.cuda.stream(stream):
        yield stream, context.main_stream


def plugin_graph_is_piecewise() -> bool:
    """Whether the plugin frontend is running the active forward under its
    own PIECEWISE graph. In plugin mode graph capture/replay is owned by the
    frontend, so its forward context is authoritative; native ATOM has no
    such graph. An unavailable context is not one. Replay does not execute
    this Python dispatcher."""
    from atom.plugin import is_vllm

    if not is_vllm():
        return False
    try:
        from vllm.forward_context import (
            get_forward_context as get_vllm_forward_context,
        )
        from vllm.forward_context import (
            is_forward_context_available,
        )

        if not is_forward_context_available():
            return False
        mode = get_vllm_forward_context().cudagraph_runtime_mode
    except (ImportError, AttributeError, AssertionError):
        return False
    return (mode if isinstance(mode, str) else getattr(mode, "name", None)) == (
        "PIECEWISE"
    )


def set_forward_context(
    attn_metadata: AttentionMetaData,
    atom_config: Config,
    context: Context,
    running_tokens_per_dp_rank: np.ndarray,
    spec_decode_metadata: SpecDecodeMetadata | None = None,
    ubatch_slices: list[Any] | None = None,
    in_hipgraph: bool = False,
    ub_max_tokens_across_dp: tuple | None = None,
    ub_tokens_across_dp: tuple | None = None,
) -> None:
    """``running_tokens_per_dp_rank``: the rows every DP rank's forward runs,
    this rank's included (`DPMetadata.gather_running_tokens` when only that
    one is known)."""
    global _forward_context
    dp_metadata = (
        DPMetadata(running_tokens_per_dp_rank)
        if atom_config.parallel_config.data_parallel_size > 1
        else None
    )

    # Mirrored here rather than at each caller: the two going out of step is how
    # an all2all comes to bound its receive buffer by one rank's count.
    context.running_tokens_across_dp = tuple(running_tokens_per_dp_rank.tolist())

    forward_mode = context.forward_mode
    _forward_context = ForwardContext(
        attn_metadata=attn_metadata,
        no_compile_layers=atom_config.compilation_config.static_forward_context,
        kv_cache_data=_forward_kv_cache_context.kv_cache_data,
        context=context,
        dp_metadata=dp_metadata,
        spec_decode_metadata=spec_decode_metadata,
        ubatch_slices=ubatch_slices,
        ub_max_tokens_across_dp=ub_max_tokens_across_dp,
        ub_tokens_across_dp=ub_tokens_across_dp,
        main_stream=(torch.cuda.current_stream() if _CUDA_AVAILABLE else None),
        in_hipgraph=in_hipgraph,
        tr_graph=(
            "replay" if forward_mode is not None and forward_mode.token_graph else None
        ),
    )  # _forward_context.attn_metadata = attn_metadata
    # _forward_context.no_compile_layers = atom_config.compilation_config.static_forward_context
    # _forward_context = ForwardContext(no_compile_layers=atom_config.compilation_config.static_forward_context, attn_metadata=attn_metadata)


def reset_forward_context() -> None:
    global _forward_context
    _forward_context = ForwardContext()


# ---------------------------------------------------------------------------
# KV Connector global instances (lazy initialization)
# ---------------------------------------------------------------------------

_logger = logging.getLogger("atom")

_global_kvconnector: Any | None = None
_global_kvconnector_scheduler: Any | None = None


def get_kvconnector(role: str = "worker", config: Config | None = None) -> Any:
    """Get or lazily initialize the global KV connector instance.

    The connector is role-dependent:
      - ``"worker"``: Returns a :class:`KVConnectorBase` (worker-side, per TP rank).
      - ``"scheduler"``: Returns a :class:`KVConnectorSchedulerBase` (scheduler-side).

    The concrete backend is selected by :class:`KVConnectorFactory` based on
    ``config.kv_transfer_config["kv_connector"]`` (default: ``"moriio"``).

    Args:
        role: Either ``"worker"`` or ``"scheduler"``.
        config: Engine config; required on first call to trigger initialization.

    Returns:
        The KV connector instance, or ``None`` if KV transfer is not configured.
    """
    global _global_kvconnector, _global_kvconnector_scheduler

    if not (hasattr(config, "kv_transfer_config") and config.kv_transfer_config):
        return _global_kvconnector

    if role == "worker":
        from aiter.dist.parallel_state import get_tp_group

        try:
            tp_rank = get_tp_group().rank_in_group
        except Exception:  # noqa: BLE001 -- logged; any failure means "no group"
            _logger.warning(
                "get_tp_group() failed (dist not initialized?), returning None"
            )
            return None

        if _global_kvconnector is None:
            from atom.kv_transfer.disaggregation import KVConnectorFactory

            _global_kvconnector = KVConnectorFactory.create_connector(
                config, role="worker"
            )
            _logger.debug("Initialized global KVConnector at tp_rank %d", tp_rank)

    elif role == "scheduler":
        from atom.kv_transfer.disaggregation import KVConnectorFactory

        _global_kvconnector_scheduler = KVConnectorFactory.create_connector(
            config, role="scheduler"
        )
        _logger.debug("Initialized global KVConnectorScheduler")
        return _global_kvconnector_scheduler

    else:
        raise ValueError(f"Unknown KV connector role: {role!r}")

    return _global_kvconnector


def set_kv_cache_data(
    kv_cache_data: dict[int, KVCacheTensor],
    config: Config | None = None,
    transfer_tensors: Any = None,
    num_blocks: int | None = None,
) -> None:
    """Register KV cache data globally and with the KV connector if enabled.

    ``num_blocks`` is the scheduler-visible KV block count (the ID space used
    by request block tables). The offload connector needs it to byte-slice
    MLA's token-major latent cache, where tensor.shape[0] is the page-size-1
    physical row count rather than the scheduler block count.
    """
    if hasattr(config, "kv_transfer_config") and config.kv_transfer_config:
        connector = get_kvconnector(config=config)
        if connector is not None:
            connector.register_kv_caches(
                kv_cache_data, transfer_tensors, num_blocks=num_blocks
            )

    _forward_kv_cache_context.kv_cache_data = kv_cache_data
