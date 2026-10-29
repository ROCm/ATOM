# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import dataclasses
import itertools
import logging
from dataclasses import dataclass

import numpy as np

# The micro-batches a TBO step splits into.
NUM_TBO_UBATCHES = 2


@dataclass
class UBatchSlice:
    """Describes which portion of a batch belongs to a micro-batch."""

    request_slice: slice  # the step's requests this micro-batch runs
    # The running tokens it runs: the micro-batches lay out end to end, none
    # overlapping.
    token_slice: slice
    # The rows of `token_slice` its requests' tokens fill, the rest padding:
    # its first rows (`cut_decode_ubatches`, `cut_prefill_ubatches`),
    # or all of them.
    scheduled_token_slice: slice | None = None
    # The requests its per-request metadata spans: its share of the step's
    # `running_bs` (`cut_decode_ubatches`), or `request_slice`'s.
    running_bs: int | None = None

    def __post_init__(self):
        if self.scheduled_token_slice is None:
            self.scheduled_token_slice = self.token_slice
        if self.running_bs is None:
            self.running_bs = self.request_slice.stop - self.request_slice.start

    @property
    def running_tokens(self) -> int:
        return self.token_slice.stop - self.token_slice.start

    @property
    def scheduled_tokens(self) -> int:
        scheduled = self.scheduled_token_slice
        return scheduled.stop - scheduled.start

    def scheduled_bs(self, step_scheduled_bs: int) -> int:
        """This micro-batch's scheduled requests: the step's first
        `step_scheduled_bs` falling in `request_slice`, the rest padding."""
        rs = self.request_slice
        return max(0, min(rs.stop, step_scheduled_bs) - rs.start)

    # Where this ubatch's own prefill/decode boundary falls, in ubatch-LOCAL
    # coordinates. `None` on both when the parent batch is not mixed.
    #
    # A mixed batch is laid out `[prefill rows | decode rows]`, and a token
    # split lands wherever it lands, so the two ubatches carry different
    # amounts of each: cut inside the prefill region and ubatch 1 gets
    # `[prefill tail | every decode row]`; cut inside the decode region and
    # ubatch 0 gets `[every prefill row | decode head]`. Neither ubatch can
    # infer its own boundary from the parent's, so it is recorded per slice.
    num_prefill_tokens: int | None = None
    num_prefill_seqs: int | None = None


def decode_bucket_splits(bs: int) -> bool:
    """Whether a decode batch of ``bs`` requests is split into micro-batches.

    One threshold for both places a decode split is decided: a step's
    `running_bs` (`ForwardMode.decide`) and a recorded request bucket
    (`ModelRunner.capture_cudagraph`, which records a bucket that splits as a
    TBO graph). Such a bucket is even (`Config.capture_sizes`), its two halves
    as many requests."""
    return bs > 2


def cut_decode_ubatches(
    running_bs: int, running_tokens: int, lengths: np.ndarray
) -> tuple[UBatchSlice, ...]:
    """A split decode's two micro-batches, cut by request as its graphs are
    recorded (`ReqGraph`). Cut once a step (`ModelRunner._cut_ubatches`, or
    the capture of a TBO graph) and handed to every consumer.

    One cut through both of the step's axes, the same on every step of the
    bucket: its `running_bs` requests in half, the first taking the first
    half of this rank's scheduled requests (``lengths``, their query tokens),
    and its `running_tokens` in half. Each half runs its own requests' packed
    tokens first, then its share of the padding
    (`decode_ubatch_token_order`)."""
    scheduled_bs = len(lengths)
    first_bs = running_bs // 2
    split = min(first_bs, scheduled_bs)
    half = running_tokens // 2
    first_tokens = int(lengths[:split].sum())
    second_tokens = int(lengths[split:].sum())
    if first_tokens > half or second_tokens > running_tokens - half:
        raise ValueError(
            f"{first_tokens} and {second_tokens} tokens of {first_bs} and "
            f"{running_bs - first_bs} requests do not fit two micro-batches of "
            f"{half} and {running_tokens - half} tokens"
        )
    return (
        UBatchSlice(
            request_slice=slice(0, split),
            token_slice=slice(0, half),
            scheduled_token_slice=slice(0, first_tokens),
            running_bs=first_bs,
        ),
        UBatchSlice(
            request_slice=slice(split, scheduled_bs),
            token_slice=slice(half, running_tokens),
            scheduled_token_slice=slice(first_tokens, first_tokens + second_tokens),
            running_bs=running_bs - first_bs,
        ),
    )


def decode_ubatch_token_order(
    slices: tuple[UBatchSlice, ...],
    scheduled_tokens: int,
    num_ubatches: int = 2,
    is_prefill: bool = False,
    num_scheduled_tokens: np.ndarray | None = None,
    max_tokens_per_ubatch: int | None = None,
    force: bool = False,
    num_prefill_seqs: int | None = None,
    num_prefill_tokens: int | None = None,
) -> list[UBatchSlice] | None:
    """Split a batch into N micro-batch slices.

    For decode: split by request count (uniform tokens per request).
    For prefill: token-balanced split so each ubatch has roughly equal
                 token count, respecting request boundaries.
    For mixed (`num_prefill_*` given): token midpoint over the whole batch,
                 with each slice recording its own prefill boundary.

    Returns None if the batch is too small to split or if the split
    would produce a ubatch exceeding max_tokens_per_ubatch.

    ``force`` bypasses the ATOM_TBO_PREFILL_MIN_TOKENS gate: the cross-DP
    decision is OR-reduced, so a rank below the min-token bar can still be
    told to split (because a peer cleared it). It MUST split anyway to keep
    the per-ubatch collectives size-aligned, or RCCL will hang.
    """
    if num_ubatches <= 1:
        return None

    # Mixed batches have their own splitter: the cut spans both segments and
    # each slice has to record where its own prefill ends. Dispatch before the
    # prefill/decode paths below, which would only see the prefill rows.
    if num_prefill_tokens is not None:
        if num_scheduled_tokens is None or num_prefill_seqs is None:
            raise RuntimeError(
                "mixed split needs per-row token counts and the prefill row "
                "count for the whole batch"
            )
        return split_mixed_token_midpoint(
            num_reqs,
            num_prefill_seqs,
            num_prefill_tokens,
            num_scheduled_tokens,
            num_ubatches,
            max_tokens_per_ubatch,
        )

    # Token-midpoint prefill split can cut *inside* a request, so it works
    # even with a single request (bs=1) — only require that there are at
    # least `num_ubatches` tokens to go around. The request-boundary path
    # below still needs num_reqs >= num_ubatches.
    token_split = num_scheduled_tokens is not None and envs.ATOM_TBO_PREFILL_TOKEN_SPLIT
    if not token_split and num_reqs < num_ubatches:
        return None

    if num_scheduled_tokens is not None:
        # Skip TBO for small prefills — unless force-split (OR-reduced cross-DP
        # decision put this under-filled rank on the TBO path; it must split to
        # stay aligned with peers).
        _min_pref = envs.ATOM_TBO_PREFILL_MIN_TOKENS
        if _min_pref > 0 and not force:
            _pref_total = int(num_scheduled_tokens[:num_reqs].sum())
            if _pref_total < _min_pref:
                return None
        # Prefill: token-balanced split
        if token_split:
            if scheduled_tokens < num_ubatches:
                return None
            return _split_prefill_token_midpoint(
                num_reqs,
                num_scheduled_tokens,
                num_ubatches,
                max_tokens_per_ubatch,
            )
        return _split_prefill_balanced(
            num_reqs,
            num_scheduled_tokens,
            num_ubatches,
            max_tokens_per_ubatch,
        )
        pad = ub.running_tokens - ub.scheduled_tokens
        token_order[start + ub.scheduled_tokens : ub.token_slice.stop] = np.arange(
            padding, padding + pad
        )
        padding += pad
    return slices[-1].token_slice.stop


def prefill_token_halves(tokens: int) -> tuple[int, int] | None:
    """A prefill's ``tokens`` cut at the midpoint: each micro-batch's count,
    or None when fewer than two would leave one empty. The one cut a rank's
    vote (`prefill_tbo_vote`) gathers and its split (`cut_prefill_ubatches`)
    runs: were they two, the micro-batches' collectives would gather sizes the
    slices do not run."""
    if tokens < 2:
        return None
    half = tokens // 2
    return half, tokens - half


def token_midpoint_cuts(total_tokens: int, num_ubatches: int) -> list[int] | None:
    """The token-midpoint split's boundaries, ``[0, c1, ..., total_tokens]``:
    ubatch i holds tokens ``[cuts[i], cuts[i + 1])``, at exact fractions of the
    total. None when a ubatch would be empty (fewer tokens than ubatches).

    The ONE statement of where a token split cuts. The DP sync MAX-reduces the
    per-ubatch sizes each rank reports before it splits (`ubatching.py`'s
    precomputes) and sizes the per-ubatch collectives with them; the slices it
    then cuts must have exactly those sizes, or the all_gathers disagree and
    the group hangs. Both sides derive them here.
    """
    cuts = [(total_tokens * i) // num_ubatches for i in range(num_ubatches + 1)]
    if any(stop <= start for start, stop in itertools.pairwise(cuts)):
        return None
    return cuts


def mixed_cut_inside_prefill(cuts: list[int], num_prefill_tokens: int) -> bool:
    """Whether a token split of a mixed ``[prefill | decode]`` batch keeps every
    ubatch but the last pure prefill: every cut but the end lands strictly
    inside the prefill tokens, so only the last ubatch straddles the boundary.

    A cut at or past the prefill tokens would leave a pure DECODE ubatch under
    a parent context that says is_prefill=True, which TBO cannot express
    (`_make_ubatch_context` branches on the parent's mode for every ubatch).
    """
    return cuts[-2] < num_prefill_tokens


def _split_prefill_token_midpoint(
    num_reqs: int,
    num_scheduled_tokens: np.ndarray,
    running_tokens: int,
) -> list[UBatchSlice] | None:
    """A prefill's two micro-batches, its ``running_tokens`` cut at the
    midpoint (`prefill_token_halves`), through a request if one straddles it,
    so a single request still splits; None when it does not cut. Tokens past
    the requests' are padding and belong to no request.

    Whether the step splits at all is the group's decision (`prefill_tbo_vote`,
    OR-reduced): a rank below the min-token bar still splits once a peer
    cleared it, to keep the per-ubatch collectives size-aligned.
    """
    halves = prefill_token_halves(running_tokens)
    if halves is None:
        return None
    toks = np.asarray(num_scheduled_tokens[:num_reqs], dtype=np.int64)
    # Exclusive-prefix cumulative token offsets: cu[i] = first token of req i.
    cu = np.zeros(num_reqs + 1, dtype=np.int64)
    np.cumsum(toks, out=cu[1:])

    cuts = token_midpoint_cuts(total_tokens, num_ubatches)
    if cuts is None:
        # Degenerate (total_tokens < num_ubatches): the caller runs unsplit.
        return None

    slices: list[UBatchSlice] = []
    for tok_start, tok_end in itertools.pairwise(cuts):
        # Requests overlapping [tok_start, tok_end):
        #   first req = the one containing tok_start
        #   last req  = the one containing tok_end - 1
        req_start = min(int(np.searchsorted(cu, tok_start, side="right") - 1), num_reqs)
        req_stop = min(int(np.searchsorted(cu, tok_end - 1, side="right")), num_reqs)
        slices.append(
            UBatchSlice(
                request_slice=slice(req_start, req_stop),
                token_slice=slice(tok_start, tok_end),
                scheduled_token_slice=slice(
                    tok_start, max(tok_start, min(tok_end, int(cu[-1])))
                ),
                # A half of nothing but padding still forwards one request.
                running_bs=max(req_stop - req_start, 1),
            )
        )

    if max_tokens_per_ubatch is not None:
        for s in slices:
            if (s.token_slice.stop - s.token_slice.start) > max_tokens_per_ubatch:
                logger.info(
                    "[TBO] token-midpoint split rejected: ubatch tokens "
                    "exceed buffer %s",
                    max_tokens_per_ubatch,
                )
                return None

    return slices


def split_mixed_token_midpoint(
    num_reqs: int,
    num_prefill_seqs: int,
    num_prefill_tokens: int,
    num_scheduled_tokens,
    num_ubatches: int = 2,
    max_tokens_per_ubatch: int | None = None,
) -> list[UBatchSlice] | None:
    """Split a mixed ``[prefill | decode]`` batch at the token midpoint.

    The prefill token split itself (`_split_prefill_token_midpoint`, cut at
    `token_midpoint_cuts` over the WHOLE batch, through a request if need be),
    refused unless `mixed_cut_inside_prefill`, and with each slice recording
    how much of itself is prefill, because that differs per ubatch and nothing
    downstream can rederive it. The prefill rows are `[0, num_prefill_seqs)` /
    `[0, num_prefill_tokens)`, so a slice's prefill part is its intersection
    with those ranges.

    `ubatching.py`'s `_precompute_mixed_token_split` reports these sizes to the
    DP sync from the same two helpers. Returns None if the batch cannot be cut
    into `num_ubatches` non-empty pieces, letting the caller run unsplit.
    """
    toks = np.asarray(num_scheduled_tokens[:num_reqs], dtype=np.int64)
    cuts = token_midpoint_cuts(int(toks.sum()), num_ubatches)
    if cuts is None or not mixed_cut_inside_prefill(cuts, num_prefill_tokens):
        return None
    slices = _split_prefill_token_midpoint(
        num_reqs, toks, num_ubatches, max_tokens_per_ubatch
    )
    if slices is None:
        return None
    return [
        dataclasses.replace(
            s,
            # `max(0, ...)` covers a slice that starts past the boundary.
            num_prefill_tokens=max(
                0, min(s.token_slice.stop, num_prefill_tokens) - s.token_slice.start
            ),
            num_prefill_seqs=max(
                0, min(s.request_slice.stop, num_prefill_seqs) - s.request_slice.start
            ),
        )
        for s in slices
    ]


def attach_tbo_cpu_lens(
    attn_metadata: AttentionMetaData, field_name: str, np_array
) -> None:
    """Publish a ready-made CPU (numpy) copy of a per-request length array.

    Called from a backend's prefill ``prepare_prefill`` AFTER all metadata is
    finalized, where the source numpy already exists (e.g.
    ``forward_vars["context_lens"].np``). ``split_attn_metadata`` then reads it
    via ``_get_tbo_cpu_lens`` (host math, zero device sync) instead of copying
    the freshly-sliced device tensors back to the host.

    NOTE: ``forward_vars[...].np`` is a *reused* pinned buffer that the next
    forward will overwrite, so callers MUST pass a ``.copy()`` (or a freshly
    allocated array) to keep it valid for this batch's lifetime.
    """
    setattr(attn_metadata, "_tbo_cpu_lens_" + field_name, np_array)


def _get_tbo_cpu_lens(attn_metadata: AttentionMetaData, field_name: str):
    """Fetch a CPU (numpy) copy of a per-request length array.

    Fast path: one was already published by ``attach_tbo_cpu_lens`` from the
    backend's prepare path (numpy was free there) — zero device sync.

    Fallback path: if none was published (a backend whose prepare_prefill
    doesn't attach one), lazily copy the device tensor to host once and cache it
    under ``_tbo_cpu_lens_<field>``; subsequent ubatch slices in the same
    forward reuse it. This is TBO-only: reached only through
    ``split_attn_metadata`` and never touched when TBO is disabled, so non-TBO
    runs pay zero extra cost.
    """
    cache_attr = "_tbo_cpu_lens_" + field_name
    if not hasattr(attn_metadata, cache_attr):
        t = getattr(attn_metadata, field_name, None)
        setattr(
            attn_metadata,
            cache_attr,
            None if t is None else t.detach().cpu().numpy(),
        )
    return getattr(attn_metadata, cache_attr)


def split_attn_metadata(
    attn_metadata: AttentionMetaData,
    ub_slice: UBatchSlice,
    running_bs: int,
) -> AttentionMetaData:
    """Split AttentionMetaData for a single micro-batch."""
    rs = ub_slice.request_slice
    ts = ub_slice.token_slice
    req_start = rs.start
    req_end = rs.stop
    ub_num_reqs = req_end - req_start

    ub_cu_seqlens_q = None
    if attn_metadata.cu_seqlens_q is not None:
        orig_cu = attn_metadata.cu_seqlens_q
        seg = orig_cu[req_start : req_end + 1]
        # clamp absolute token offsets to [ts.start, ts.stop], then re-base
        clamped = torch.clamp(seg, min=ts.start, max=ts.stop)
        ub_cu_seqlens_q = clamped - clamped[0]
        # Pad remaining entries up to running_bs + 1
        if running_bs > ub_num_reqs:
            last_val = ub_cu_seqlens_q[-1]
            pad_size = running_bs - ub_num_reqs
            padding = last_val.expand(pad_size)
            ub_cu_seqlens_q = torch.cat([ub_cu_seqlens_q, padding])

    # slot_mapping: slice by token
    ub_slot_mapping = None
    if attn_metadata.slot_mapping is not None:
        ub_slot_mapping = attn_metadata.slot_mapping[ts]
        # Pad with -1 for padded positions
        tok_count = ts.stop - ts.start
        padded_tok_count = running_bs
        if padded_tok_count > tok_count:
            pad = torch.full(
                (padded_tok_count - tok_count,),
                -1,
                dtype=attn_metadata.slot_mapping.dtype,
                device=attn_metadata.slot_mapping.device,
            )
            ub_slot_mapping = torch.cat([ub_slot_mapping, pad])

    # context_lens: slice by request
    ub_context_lens = None
    if attn_metadata.context_lens is not None:
        ub_context_lens = attn_metadata.context_lens[rs]
        if running_bs > ub_num_reqs:
            pad = torch.zeros(
                running_bs - ub_num_reqs,
                dtype=attn_metadata.context_lens.dtype,
                device=attn_metadata.context_lens.device,
            )
            ub_context_lens = torch.cat([ub_context_lens, pad])

    # block_tables: slice by request
    ub_block_tables = None
    if attn_metadata.block_tables is not None:
        ub_block_tables = attn_metadata.block_tables[rs]
        if running_bs > ub_num_reqs:
            pad = torch.zeros(
                running_bs - ub_num_reqs,
                attn_metadata.block_tables.shape[1],
                dtype=attn_metadata.block_tables.dtype,
                device=attn_metadata.block_tables.device,
            )
            ub_block_tables = torch.cat([ub_block_tables, pad])

    # kv_indptr: re-base from 0
    ub_kv_indptr = None
    if attn_metadata.kv_indptr is not None:
        orig_indptr = attn_metadata.kv_indptr
        base = orig_indptr[req_start]
        ub_kv_indptr = orig_indptr[req_start : req_end + 1] - base
        if running_bs > ub_num_reqs:
            last_val = ub_kv_indptr[-1]
            pad_size = running_bs - ub_num_reqs
            padding = last_val.expand(pad_size)
            ub_kv_indptr = torch.cat([ub_kv_indptr, padding])

    # kv_indices: shared (indexed via kv_indptr, no slicing needed)
    ub_kv_indices = attn_metadata.kv_indices

    # batch_id_per_k_token: sliced and re-based, not rebuilt -- the parent laid
    # the requests out contiguously and `cu_seqlens_k` says where this one starts.
    ub_batch_id_per_k_token = None
    if attn_metadata.batch_id_per_k_token is not None:
        cuk_cpu = _get_tbo_cpu_lens(attn_metadata, "cu_seqlens_k")
        kv = slice(int(cuk_cpu[req_start]), int(cuk_cpu[req_end]))
        ub_batch_id_per_k_token = attn_metadata.batch_id_per_k_token[kv] - req_start

    # kv_last_page_lens: slice by request
    ub_kv_last_page_lens = None
    if attn_metadata.kv_last_page_lens is not None:
        ub_kv_last_page_lens = attn_metadata.kv_last_page_lens[rs]
        if running_bs > ub_num_reqs:
            pad = torch.zeros(
                running_bs - ub_num_reqs,
                dtype=attn_metadata.kv_last_page_lens.dtype,
                device=attn_metadata.kv_last_page_lens.device,
            )
            ub_kv_last_page_lens = torch.cat([ub_kv_last_page_lens, pad])

    # Prefix-cache prefill needs these fields to gather cached KV before
    # varlen attention. Dropping them makes cu_seqlens_k describe cached+new
    # tokens while K/V only contain new tokens, which can OOB in flash-attn.
    ub_num_cached_tokens = None
    if attn_metadata.num_cached_tokens is not None:
        ub_num_cached_tokens = attn_metadata.num_cached_tokens[rs]
        if running_bs > ub_num_reqs:
            pad = torch.zeros(
                running_bs - ub_num_reqs,
                dtype=attn_metadata.num_cached_tokens.dtype,
                device=attn_metadata.num_cached_tokens.device,
            )
            ub_num_cached_tokens = torch.cat([ub_num_cached_tokens, pad])

    ub_seq_starts = None
    if attn_metadata.seq_starts is not None:
        ub_seq_starts = attn_metadata.seq_starts[rs]
        if running_bs > ub_num_reqs:
            pad = torch.zeros(
                running_bs - ub_num_reqs,
                dtype=attn_metadata.seq_starts.dtype,
                device=attn_metadata.seq_starts.device,
            )
            ub_seq_starts = torch.cat([ub_seq_starts, pad])

    # sparse_kv_indptr: slice and re-base if present (per-request dimension).
    # NOTE: In MLA prefill sparse mode, sparse_kv_indptr is per-token — that
    # case is handled by the MLA builder's build_ubatch_prefill_metadata override.
    ub_sparse_kv_indptr = None
    if attn_metadata.sparse_kv_indptr is not None:
        orig = attn_metadata.sparse_kv_indptr
        base = orig[req_start]
        ub_sparse_kv_indptr = orig[req_start : req_end + 1] - base
        if running_bs > ub_num_reqs:
            last_val = ub_sparse_kv_indptr[-1]
            pad_size = running_bs - ub_num_reqs
            padding = last_val.expand(pad_size)
            ub_sparse_kv_indptr = torch.cat([ub_sparse_kv_indptr, padding])

    # cu_seqlens_k: re-base like cu_seqlens_q (needed for prefill attention)
    ub_cu_seqlens_k = None
    if attn_metadata.cu_seqlens_k is not None:
        orig_cu_k = attn_metadata.cu_seqlens_k
        seg_k = orig_cu_k[req_start : req_end + 1]
        if not getattr(attn_metadata, "has_cached", False):
            clamped_k = torch.clamp(seg_k, min=ts.start, max=ts.stop)
            ub_cu_seqlens_k = clamped_k - clamped_k[0]
        else:
            ub_cu_seqlens_k = seg_k - seg_k[0]
        if running_bs > ub_num_reqs:
            last_val = ub_cu_seqlens_k[-1]
            pad_size = running_bs - ub_num_reqs
            padding = last_val.expand(pad_size)
            ub_cu_seqlens_k = torch.cat([ub_cu_seqlens_k, padding])

    # Scalars (max_seqlen_q/k, total_kv) are derived from per-request lengths
    # that originate as numpy on the CPU (see the attention builders). Instead of
    # syncing the freshly-sliced device tensors back to the host, slice the CPU
    # length copies and compute these with numpy — no GPU sync, no kernel
    # launches. The copies are published by attach_tbo_cpu_lens (or created
    # lazily on first use as a fallback) and are TBO-only.
    ub_max_seqlen_k = attn_metadata.max_seqlen_k
    ub_max_seqlen_q = attn_metadata.max_seqlen_q

    if ub_num_reqs > 0:
        # max_seqlen_k: max over this ubatch's context lengths.
        ctx_cpu = _get_tbo_cpu_lens(attn_metadata, "context_lens")
        if ctx_cpu is not None:
            ub_ctx_cpu = ctx_cpu[req_start:req_end]
            ub_max_seqlen_k = int(ub_ctx_cpu.max())

        # max_seqlen_q: max per-request q length, matching the clamp/re-base that
        # produced ub_cu_seqlens_q above (clamp to the token window, then diff).
        cuq_cpu = _get_tbo_cpu_lens(attn_metadata, "cu_seqlens_q")
        if cuq_cpu is not None:
            seg = np.clip(cuq_cpu[req_start : req_end + 1], ts.start, ts.stop)
            ub_max_seqlen_q = int(np.diff(seg).max())

    ub_total_kv = None
    if attn_metadata.has_cached:
        if ub_cu_seqlens_k is not None and ub_num_reqs > 0:
            ub_total_kv = int(ub_cu_seqlens_k[ub_num_reqs].item())
        elif ub_context_lens is not None and ub_num_reqs > 0:
            ub_total_kv = int(ub_context_lens[:ub_num_reqs].sum().item())
        else:
            ub_total_kv = 0

    # MLA work buffers are set to None here — they will be recomputed
    # by decode_update_mla_metadata_v1 in UBatchWrapper before each micro-batch run.
    # Backend-specific fields (e.g. MLA sparse prefill) are handled by
    # the builder's build_ubatch_prefill_metadata override.
    ub_attn = AttentionMetaData(
        cu_seqlens_q=ub_cu_seqlens_q,
        cu_seqlens_k=ub_cu_seqlens_k,
        max_seqlen_q=ub_max_seqlen_q,
        max_seqlen_k=ub_max_seqlen_k,
        min_seqlen_q=attn_metadata.min_seqlen_q,
        slot_mapping=ub_slot_mapping,
        context_lens=ub_context_lens,
        block_tables=ub_block_tables,
        dropout_p=attn_metadata.dropout_p,
        kv_indptr=ub_kv_indptr,
        kv_indices=ub_kv_indices,
        kv_last_page_lens=ub_kv_last_page_lens,
        sparse_kv_indptr=ub_sparse_kv_indptr,
        has_cached=attn_metadata.has_cached,
        total_kv=ub_total_kv,
        batch_id_per_k_token=ub_batch_id_per_k_token,
        num_cached_tokens=ub_num_cached_tokens,
        seq_starts=ub_seq_starts,
        work_meta_data=None,
        work_indptr=None,
        work_info_set=None,
        reduce_indptr=None,
        reduce_final_map=None,
        reduce_partial_map=None,
    )
    # Carry over dtype_q if set
    if hasattr(attn_metadata, "dtype_q"):
        ub_attn.dtype_q = attn_metadata.dtype_q
    return ub_attn
