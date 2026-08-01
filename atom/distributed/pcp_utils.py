# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Prefill Context Parallel (PCP) helpers for DeepSeek-V4.

PCP splits the prefill token sequence across the PCP process group (an
independent parallel dimension, world = tp x pcp). Only the prefill query
side is sharded; each rank keeps the full KV (full-KV scheme), so decode is
unchanged. Load balancing uses round-robin splitting:
`token_idx % pcp_size == pcp_rank`.

Ported from SGLang's DSA round-robin CP path
(`layers/attention/dsa/utils.py:dsa_cp_round_robin_split_data` and
`layers/utils/cp_utils.py:cp_all_gather_rerange_output`).
"""

import logging
from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch
from aiter.dist.parallel_state import (
    get_pcp_group,
    get_prefill_context_model_parallel_rank,
    get_prefill_context_model_parallel_world_size,
)


class PcpBalGroup(NamedTuple):
    """One request group for PCP+TBO request-boundary split prefill.

    A prefill batch is split into request groups at request boundaries (never
    inside a sequence); each group is processed as an independent non-TBO PCP
    mini-batch (padded to a pcp multiple, round-robin striped, reindexed on its
    own). Consumed by ModelRunner.run_model (per-group stripe / restore) and the
    attn builder's `_build_ubatch_prefill_metadata_balanced` (slice + reindex).
    """

    req_start: int  # first request index of this group (inclusive)
    req_stop: int  # last request index of this group (exclusive)
    tok_start: int  # global token offset of the group's first token
    tok_end: int  # global token offset past the group's last REAL token
    pad_total: (
        int  # tok count padded to a pcp multiple = pcp_pad_len(tok_end-tok_start, pcp)
    )


logger = logging.getLogger("atom")

def plugin_attn_cp_enabled() -> bool:
    """Return the validated Config-level vLLM reuse-TP-as-CP mode.

    Reading the raw environment here would partially enable CP while constructing
    native ATOM or SGLang models. Plugin config translation is the authoritative gate.
    """
    try:
        from atom.config import get_current_atom_config

        config = get_current_atom_config()
    except Exception:
        return False
    plugin = getattr(config, "plugin_config", None)
    return bool(
        getattr(config, "vllm_attn_cp", False)
        and plugin is not None
        and getattr(plugin, "is_vllm", False)
    )


def get_pcp_world_size() -> int:
    return get_prefill_context_model_parallel_world_size()


def get_pcp_rank() -> int:
    return get_prefill_context_model_parallel_rank()


def pcp_is_enabled() -> bool:
    return get_pcp_world_size() > 1


def pcp_pad_len(
    total_tokens: int,
    pcp_size: Optional[int] = None,
    multiple: int = 1,
) -> int:
    """Padded token count so the global sequence is divisible by pcp_size * multiple.

    Round-robin split requires the global token count to be divisible by pcp_size
    (see SGLang `can_dsa_cp_split` assert / HIP `apply_cp_reindex`). `multiple` is
    an extra factor applied on top of pcp_size when the sequence must additionally
    be evenly divisible by some multiplier. Returns the padded length
    (>= total_tokens); callers pad per-token tensors to this length with dummy
    tokens (KV length 0) before splitting.

    """
    if pcp_size is None:
        pcp_size = get_pcp_world_size()
    divisor = pcp_size * max(multiple, 1)
    if divisor <= 1:
        return total_tokens
    rem = total_tokens % divisor
    if rem == 0:
        return total_tokens
    return total_tokens + (divisor - rem)


def pcp_round_robin_split(
    input_: torch.Tensor, pcp_size: Optional[int] = None, pcp_rank: Optional[int] = None
) -> torch.Tensor:
    """Take this rank's round-robin shard along dim 0.

    Selects rows `[pcp_rank, pcp_rank + pcp_size, pcp_rank + 2*pcp_size, ...]`.
    Requires `input_.shape[0] % pcp_size == 0` (pad upstream via pcp_pad_len).

    Mirrors SGLang `dsa_cp_round_robin_split_data`:
        input_.view(-1, pcp_size, *rest)[:, pcp_rank]
    """
    if pcp_size is None:
        pcp_size = get_pcp_world_size()
    if pcp_size <= 1:
        return input_
    if pcp_rank is None:
        pcp_rank = get_pcp_rank()
    # Divisibility by pcp_size is guaranteed upstream by pcp_pad_len (callers
    # pad before splitting); the view below would error if violated.
    rest = tuple(input_.shape[1:])
    shard = input_.view(-1, pcp_size, *rest)[:, pcp_rank]
    # The round-robin slice has inner stride `pcp_size` (it selects every
    # pcp_size-th row). `.contiguous()` normalises that to unit stride by
    # copying -- EXCEPT when the shard holds a single element (n_owned == 1,
    # e.g. a decode batch that pads to exactly pcp_size): a numel<=1 tensor
    # reports is_contiguous()==True regardless of stride, so `.contiguous()`
    # is a no-op and the stride-`pcp_size` view leaks downstream. That breaks
    # kernels asserting unit inner stride -- notably aiter rope
    # (`positions.stride(1) == 1`) on the 1/pcp query positions. `clone` with
    # an explicit contiguous format always allocates standard (unit-inner)
    # strides, so it fixes n_owned==1 while costing no more than the copy
    # `.contiguous()` already performs for the (always strided) n_owned>1 case.
    return shard.clone(memory_format=torch.contiguous_format)


def _pcp_ca_comm(group):
    """Return the group's custom-all-reduce communicator, or None.

    The custom AR comm (``ca_comm``) owns the capture-safe collective kernels
    (pre-registered IPC pool, ``_IS_CAPTURING`` handling). It exists on the
    reuse-TP-as-CP group because TP all-reduce already uses it.
    """
    dc = getattr(group, "device_communicator", None)
    ca = getattr(dc, "ca_comm", None) if dc is not None else None
    if ca is None or getattr(ca, "disabled", True):
        return None
    return ca


def _align_pad_rows_for_custom_ag(inp: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Append zero rows so the tensor's total byte size is a multiple of 16.

    aiter's capture-safe custom all-gather (``CustomAllreduce.should_custom_ag``)
    only accepts 16-byte-aligned tensors. Row-aligned 2-D gathers (hidden/k, whose
    row bytes are already a multiple of 16) never pad; only tiny 1-D int gathers
    (a single int64 query id = 8 B on an ``n_owned == 1`` decode shard) do. The
    pad rows sort to the tail after round-robin rerange, so the caller drops them
    by slicing with the original (unpadded) length.
    """
    import math

    row = inp.element_size()
    for s in inp.shape[1:]:
        row *= int(s)
    if row == 0:
        return inp, 0
    mult = 16 // math.gcd(16, row)  # rows needed for a 16 B-aligned total
    rem = inp.shape[0] % mult
    if rem == 0:
        return inp, 0
    pad = mult - rem
    pad_block = inp.new_zeros((pad,) + tuple(inp.shape[1:]))
    return torch.cat([inp, pad_block], dim=0), pad


def _custom_all_gather_dim0(group, x: torch.Tensor) -> torch.Tensor:
    """aiter custom (capture-safe) all-gather along dim 0.

    aiter's ``custom_all_gather`` maps int types to a same-width float for its
    memcpy kernel, but int64 -> float64 is rejected by the kernel
    (``Unsupported dtype: torch.float64``). 8-byte dtypes (int64/float64, e.g.
    ``input_ids`` / ``positions``) are reinterpreted as int32 pairs -- which the
    kernel supports (int32 -> float32) -- and restored afterwards. This is a pure
    bitcast (``view``), so values are exact and rank-major order is preserved
    (each rank's row stays a contiguous 2xint32 that views back to one int64).
    """
    if x.dtype in (torch.int64, torch.float64):
        g32 = group.all_gather(x.view(torch.int32), use_custom=True, dim=0)
        return g32.view(x.dtype)
    return group.all_gather(x, use_custom=True, dim=0)


def pcp_dim0_all_gather(group, inp: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Rank-major all-gather along dim 0, preferring the CAPTURE-SAFE custom path.

    The default aiter ``all_gather(use_custom=False)`` lowers to raw
    ``torch.distributed.all_gather_into_tensor`` (RCCL), whose host-side lazy
    init/copies invalidate HIP stream capture -- a full CUDA graph over a split
    decode batch dies with ``ncclUnhandledCudaError``. The custom all-gather
    (``ca_comm``, pre-registered IPC pool, int-dtype aware) IS capturable, so we
    take it whenever the tensor fits ``should_custom_ag``; otherwise (large
    prefill gathers, which run eager/piecewise and never enter a full graph) we
    fall back to RCCL.

    Returns ``(gathered_rank_major, pad_rows)``. When ``pad_rows > 0`` the extra
    rows are this rank's tail, so after rerange the real tokens are the first
    ``pcp_size * original_len`` entries.
    """
    inp = inp.contiguous()
    ca = _pcp_ca_comm(group)
    if ca is not None:
        cand, pad = _align_pad_rows_for_custom_ag(inp)
        if ca.should_custom_ag(cand):
            return _custom_all_gather_dim0(group, cand), pad
    return group.all_gather(inp, dim=0), 0


def pcp_tp_all_gather_dim0(input_: torch.Tensor) -> torch.Tensor:
    """Capture-safe rank-major all-gather (dim 0) for the reuse-TP-as-CP MoE
    path (all-gather -> experts -> reduce-scatter).

    Mirrors ``tensor_model_parallel_all_gather(x, dim=0)`` but (a) prefers the
    custom capturable collective (see :func:`pcp_dim0_all_gather`) so the MoE
    gather is legal inside a full CUDA graph over a split decode batch, and (b)
    routes through the DEDICATED CP group (``get_pcp_group()``) rather than TP,
    so it shares CP's isolated ca_comm / graph-buffer slot allocator with the
    embed & indexer gathers instead of TP all-reduce's. Hidden-state rows are
    always 16 B-aligned, so no padding is ever applied here.
    """
    gathered, _pad = pcp_dim0_all_gather(get_pcp_group(), input_)
    return gathered


def pcp_reduce_scatter_dim0(input_: torch.Tensor) -> torch.Tensor:
    """Capture-safe rank-major reduce-scatter (dim 0) over the dedicated CP group
    for the reuse-TP-as-CP MoE path (all-gather -> experts -> reduce-scatter).

    Inverse of :func:`pcp_tp_all_gather_dim0`: sums each rank's partial expert
    outputs over the full token set and scatters dim 0 back to this rank's 1/cp
    shard. Uses the custom (``use_custom=True``) capturable reduce-scatter on the
    CP group's own ca_comm so it is legal inside a full CUDA graph and shares the
    CP slot allocator with the matching all-gather (NOT TP all-reduce's).
    """
    group = get_pcp_group()
    if group.world_size == 1:
        return input_
    return group.reduce_scatter_tensor(input_, use_custom=True, dim=0)


def pcp_allgather_rerange(
    input_: torch.Tensor, pcp_size: Optional[int] = None
) -> torch.Tensor:
    """All-gather round-robin shards along dim 0 and restore original token order.

    Each rank holds `[L, *rest]` (its round-robin shard). After all-gather the
    naive layout is rank-major `[rank0_rows, rank1_rows, ...]`; the round-robin
    interleave is restored by `view(pcp, L, *rest).transpose(0, 1)` so that
    output[t] == global token t.

    Mirrors SGLang `cp_all_gather_rerange_output` (round-robin branch). Uses the
    capture-safe custom all-gather so this op is legal inside a full CUDA graph
    (see :func:`pcp_dim0_all_gather`).
    """
    if pcp_size is None:
        pcp_size = get_pcp_world_size()
    if pcp_size <= 1:
        return input_
    group = get_pcp_group()
    local_len = input_.shape[0]
    rest = tuple(input_.shape[1:])
    # rank-major concat [pcp*(L+pad), *rest]; pad rows are each rank's tail.
    gathered, pad = pcp_dim0_all_gather(group, input_)
    padded_len = local_len + pad
    # rank-major [pcp, L+pad, *rest] -> transpose -> token-major
    # [L+pad, pcp, *rest] -> flatten. Real global tokens occupy the first
    # pcp*L rows (row index l*pcp+r, l<L); pad rows (l>=L) sort to the tail.
    out = (
        gathered.view(pcp_size, padded_len, *rest)
        .transpose(0, 1)
        .reshape(pcp_size * padded_len, *rest)
    )
    return out[: pcp_size * local_len]


# ==== MoE-path PCP collectives (rank-major gather + reduce_scatter) ====
# Rank-major all_gather + reduce_scatter are a mutually-inverse pair:
#   - gather (1/W -> full): all_gather(dim=0) concats rank-major, so rank r's
#     1/W stripe lands at rows [r*L:(r+1)*L]. MoE is per-token so the rank-major
#     (not global) order is fine.
#   - reduce_scatter (full partial-sum -> 1/W): sums the pcp-half across ranks
#     AND scatters dim0 back so rank r receives the summed chunk r == its own
#     original stripe tokens. No rerange/slice needed.


def pcp_allgather_rankmajor(
    input_: torch.Tensor, pcp_size: Optional[int] = None
) -> torch.Tensor:
    """Gather this rank's 1/W stripe shard into the full rank-major sequence
    via a plain all_gather (dim=0). Inverse of pcp_reduce_scatter."""
    if pcp_size is None:
        pcp_size = get_pcp_world_size()
    if pcp_size <= 1:
        return input_
    return get_pcp_group().all_gather(input_.contiguous(), dim=0)


def pcp_reduce_scatter(
    input_: torch.Tensor, pcp_size: Optional[int] = None
) -> torch.Tensor:
    """Sum the pcp-half across ranks and scatter dim0 back to this rank's 1/W
    stripe via a plain reduce_scatter (dim=0). Inverse of pcp_allgather_rankmajor."""
    if pcp_size is None:
        pcp_size = get_pcp_world_size()
    if pcp_size <= 1:
        return input_
    return get_pcp_group().reduce_scatter(input_.contiguous(), dim=0)


def pcp_all_reduce(
    input_: torch.Tensor, pcp_size: Optional[int] = None
) -> torch.Tensor:
    """All-reduce (sum) over the PCP group, no token reshaping. DECODE path:
    tokens are pcp-redundant (every rank holds the same full batch), so just sum
    the pcp-half of the intermediate that combine_outputs' tp all_reduce missed.
    Uses aiter's compile-safe custom-op all_reduce.
    """
    if pcp_size is None:
        pcp_size = get_pcp_world_size()
    if pcp_size <= 1:
        return input_
    return get_pcp_group().all_reduce(input_)


def pcp_round_robin_query_indices(
    n_global_q: int, pcp_size: Optional[int] = None, pcp_rank: Optional[int] = None
) -> torch.Tensor:
    """Global query indices owned by this rank under round-robin split.

    Returns `[pcp_rank, pcp_rank+pcp_size, ...]` clipped to `< n_global_q`.
    `n_global_q` should already be padded to a multiple of pcp_size for the
    paddingless fast path; if not, the tail rank simply gets fewer queries.
    """
    if pcp_size is None:
        pcp_size = get_pcp_world_size()
    if pcp_rank is None:
        pcp_rank = get_pcp_rank()
    # Returns a CPU LongTensor of owned global query positions.
    return torch.arange(pcp_rank, n_global_q, pcp_size, dtype=torch.long)


# pcp_pad_indptr / pcp_pad_dense share the (tensor, n_pad) signature but pad two
# DIFFERENT metadata shapes, so they are kept separate on purpose:
#
#   dense (per-query: one value per token), e.g. skip_prefix_len_csa:
#       [5, 3, 8]  --pcp_pad_dense(.,1)-->  [5, 3, 8, 0]
#                                                     ^ dummy query q3 = 0 row
#
#   ragged (per-query variable-length segments, sliced by an indptr prefix-sum),
#   e.g. kv_indices grouped by kv_indptr:
#       kv_indptr  = [0, 2, 5, 6]   kv_indices = [a,b | c,d,e | f]
#       --pcp_pad_indptr(kv_indptr, 1)-->  [0, 2, 5, 6, 6]
#                                                       ^ dummy q3 segment =
#                                                         indices[6:6] = EMPTY
#       (kv_indices itself is NOT touched — the dummy query references no KV)
#
# So dense APPENDS ZERO ROWS; indptr APPENDS REPEATS OF THE LAST PREFIX-SUM
# VALUE (giving the dummy query a zero-length segment). Both make padded dummy
# queries contribute nothing to attention; they are sliced to 1/W by owned_q
# and dropped after the final all-gather.
def pcp_pad_indptr(kv_indptr: torch.Tensor, n_pad: int) -> torch.Tensor:
    """Pad a ragged prefix-sum indptr `[T+1]` to `[T+n_pad+1]`.

    Appends `n_pad` entries each repeating the last value, i.e. the padded
    (dummy) queries get zero-length KV segments. Used so per-query metadata
    matches the token sequence padded to a multiple of pcp_size; the dummy
    tokens then contribute nothing to attention.
    """
    if n_pad <= 0:
        return kv_indptr
    tail = kv_indptr[-1:].expand(n_pad)
    return torch.cat([kv_indptr, tail], dim=0)


def pcp_pad_dense(t: torch.Tensor, n_pad: int) -> torch.Tensor:
    """Pad a dense per-token tensor `[T, ...]` to `[T+n_pad, ...]` with zeros."""
    if n_pad <= 0:
        return t
    return torch.cat([t, t.new_zeros(n_pad, *t.shape[1:])], dim=0)


@dataclass(frozen=True)
class PcpTokenRowMap:
    """One explicit global-token to compact local-row mapping.

    ``owned_global_rows`` includes padded dummy rows so every CP rank executes the
    same number of model rows. ``global_to_local`` covers real global rows only;
    unowned rows contain -1. Request and phase helpers always derive from these two
    tensors, so indexer and main sparse metadata cannot invent different row orders.
    """

    num_real_tokens: int
    num_padded_tokens: int
    owned_global_rows: torch.Tensor
    global_to_local: torch.Tensor

    @property
    def num_local_rows(self) -> int:
        return int(self.owned_global_rows.numel())

    def local_range(self, global_start: int, global_end: int) -> tuple[int, int]:
        """Return the compact local slice for a contiguous global token range."""
        rows = self.owned_global_rows
        start = int(torch.searchsorted(rows, global_start, right=False).item())
        end = int(torch.searchsorted(rows, global_end, right=False).item())
        return start, end

    def owned_real_rows(self, global_start: int, global_end: int) -> torch.Tensor:
        """Return real global rows owned inside ``[global_start, global_end)``."""
        start, end = self.local_range(global_start, global_end)
        return self.owned_global_rows[start:end]


def pcp_build_token_row_map(
    num_real_tokens: int,
    pcp_size: Optional[int] = None,
    pcp_rank: Optional[int] = None,
) -> PcpTokenRowMap:
    """Build the canonical mapping used by every plugin mixed-CP consumer."""
    padded = pcp_pad_len(num_real_tokens, pcp_size)
    owned = pcp_round_robin_query_indices(padded, pcp_size, pcp_rank)
    global_to_local = torch.full((num_real_tokens,), -1, dtype=torch.long)
    real = owned[owned < num_real_tokens]
    if real.numel() > 0:
        global_to_local[real] = torch.arange(real.numel(), dtype=torch.long)
    return PcpTokenRowMap(num_real_tokens, padded, owned, global_to_local)


def pcp_split_true_decodes_and_context(
    query_start_loc: torch.Tensor,
    is_prefilling: torch.Tensor,
) -> tuple[int, int, int, bool]:
    """Split request-major rows into true decode and context-aware phases.

    vLLM's decode threshold may group short extends with decode. Plugin CP cannot:
    its paged decode path requires exactly one generated token per request. A true
    decode is therefore ``query_len == 1 and not is_prefilling``. Every subsequent
    request is handled by the context-aware prefill path. The final boolean flags a
    multi-token non-prefilling request (spec verification), for which CP is disabled.

    Returns:
        Number of true decode requests, true decode tokens, context requests, and
        whether the batch contains multi-token non-prefilling requests.
    """
    query_lens = torch.diff(query_start_loc.to(torch.int64))
    prefilling = is_prefilling[: query_lens.numel()].to(dtype=torch.bool)
    true_decode = (query_lens == 1) & ~prefilling
    non_decode = ~true_decode
    if torch.any(non_decode):
        first_context = int(non_decode.to(torch.int32).argmax().item())
    else:
        first_context = int(query_lens.numel())

    # Reordered CommonAttentionMetadata must have one contiguous true-decode prefix.
    if torch.any(true_decode[first_context:]):
        raise AssertionError(
            "true one-token decode rows must precede all context-aware rows"
        )
    decode_tokens = int(query_start_loc[first_context].item())
    has_multitoken_nonprefill = bool(torch.any((query_lens > 1) & ~prefilling))
    return (
        first_context,
        decode_tokens,
        int(query_lens.numel()) - first_context,
        has_multitoken_nonprefill,
    )


def pcp_owned_request_rows(
    row_map: PcpTokenRowMap,
    query_start_loc: torch.Tensor,
    request_start: int,
    request_end: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map owned flattened tokens in a request range back to request rows.

    Returns the owned global token rows and their request-major row indices. This
    is intentionally token-driven: request tensors must never be indexed directly
    by flattened token ids when query lengths are variable.
    """
    token_start = int(query_start_loc[request_start].item())
    token_end = int(query_start_loc[request_end].item())
    owned_tokens = row_map.owned_real_rows(token_start, token_end)
    if owned_tokens.numel() == 0:
        return owned_tokens, owned_tokens.new_empty(0)
    request_rows = torch.searchsorted(
        query_start_loc[1:].to(owned_tokens.device), owned_tokens, right=True
    )
    if not torch.all(
        (request_rows >= request_start) & (request_rows < request_end)
    ):
        raise AssertionError("CP owned token mapped outside its request segment")
    return owned_tokens, request_rows


def pcp_sparse_prefill_reindex(
    sparse_seqlen: torch.Tensor,  # [T] per-query selected-KV length (pre-clamp)
    req_id_per_token: torch.Tensor,  # [T] per-query request id
    slot_mapping: torch.Tensor,  # [T] full per-query slot mapping (KV write)
    index_topk: int,
    pcp_size: Optional[int] = None,
    pcp_rank: Optional[int] = None,
    row_map: Optional[PcpTokenRowMap] = None,
) -> dict:
    """Reduce the plugin sparse-MLA per-query metadata to this rank's 1/W queries.

    Plugin reuse-TP-as-CP mirror of the native model_runner
    ``AiterMLAImpl._apply_pcp_reindex`` (``aiter_mla.py``), adapted to the plugin
    ``AiterMlaSparseMetadataForVllm`` field names. Only *query-indexed* fields
    shrink to the round-robin owned subset; *per-sequence* / *KV-write* fields
    (``slot_mapping``, ``block_table``, ``seq_lens``) stay FULL so the full KV is
    still written and gathered. The global token count is padded to a multiple of
    ``pcp_size``; padded (dummy) queries get zero-length KV (they attend nothing
    and their output is dropped after the model's exit all-gather + unpad).

    Pure tensor math (no kernels), so it is unit-testable on CPU against a
    full-batch reference. Returns a dict of the owned-query tensors plus
    ``owned_q`` / ``n_owned`` so the builder can rebuild its work buffers.
    """
    device = sparse_seqlen.device
    s_real = int(sparse_seqlen.shape[0])
    if row_map is None:
        row_map = pcp_build_token_row_map(s_real, pcp_size, pcp_rank)
    elif row_map.num_real_tokens != s_real:
        raise ValueError(
            "CP token row map does not match sparse metadata: "
            f"map={row_map.num_real_tokens}, sparse={s_real}"
        )
    n_pad = row_map.num_padded_tokens - s_real
    owned_q = row_map.owned_global_rows.to(device)
    n_owned = row_map.num_local_rows

    # sparse_kv_indptr <- cumsum of min(sparse_seqlen, topk) over owned queries;
    # dummy queries padded to 0 => zero-length KV segment.
    owned_counts = pcp_pad_dense(sparse_seqlen, n_pad)[owned_q].to(torch.int64)
    owned_counts = torch.clamp(owned_counts, max=int(index_topk))
    paged_kv_indptr = torch.zeros(n_owned + 1, dtype=torch.int32, device=device)
    paged_kv_indptr[1:] = torch.cumsum(owned_counts, 0).to(torch.int32)

    # one query per row (incl dummies) => qo_indptr = arange, last_page_len = 1s.
    qo_indptr = torch.arange(n_owned + 1, dtype=torch.int32, device=device)
    paged_kv_last_page_len = torch.ones(n_owned, dtype=torch.int32, device=device)

    # per-query request id, padded (dummy -> 0) then owned-selected.
    req_id_owned = (
        pcp_pad_dense(req_id_per_token, n_pad)[owned_q].to(torch.int32).contiguous()
    )

    # owned slot_mapping for any fused q_out kernel; the real full-KV completion
    # write in the sparse layer overwrites every real slot, and dummy queries
    # clamp to the last real slot so they can never touch an unrelated slot.
    owned_clamped = torch.clamp(owned_q, max=max(s_real - 1, 0))
    slot_mapping_owned = slot_mapping[owned_clamped].contiguous()

    return {
        "owned_q": owned_q,
        "n_owned": n_owned,
        "paged_kv_indptr": paged_kv_indptr,
        "qo_indptr": qo_indptr,
        "paged_kv_last_page_len": paged_kv_last_page_len,
        "req_id_per_token": req_id_owned,
        "slot_mapping_owned": slot_mapping_owned,
    }


def pcp_reindex_ragged(
    kv_indptr: torch.Tensor,  # [T_global + 1] int32 — global per-query prefix sum
    kv_indices: torch.Tensor,  # [kv_indptr[-1]] — ragged packed values
    owned_q: torch.Tensor,  # [T_local] long — global query ids this rank owns
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reindex a ragged (indptr, indices) pair down to this rank's queries.

    Given global per-query ragged metadata and the global query ids this rank
    owns (round-robin shard), produce the compacted local `(indptr_local,
    indices_local)` so that for the i-th owned query:
        indices_local[indptr_local[i] : indptr_local[i+1]]
          == kv_indices[kv_indptr[g] : kv_indptr[g+1]]   where g = owned_q[i]

    Used to shard the per-query prefill index buffers (kv_indptr/kv_indices
    _prefix_swa / _extend) to 1/W while the values themselves still point into
    the full KV (paged unified_kv) / full extend kv tensor.
    """
    device = kv_indptr.device
    owned_q = owned_q.to(device)
    starts = kv_indptr[owned_q]  # [T_local]
    ends = kv_indptr[owned_q + 1]  # [T_local]
    lens = ends - starts  # [T_local] per-owned-query segment length
    indptr_local = torch.zeros(
        owned_q.shape[0] + 1, dtype=kv_indptr.dtype, device=device
    )
    torch.cumsum(lens, dim=0, out=indptr_local[1:])
    total = int(indptr_local[-1].item())
    if total == 0:
        return indptr_local, kv_indices.new_empty(0)
    # Build a gather map: for each output slot, which source index to read.
    # out_slot s in [indptr_local[i], indptr_local[i+1]) reads from
    # starts[i] + (s - indptr_local[i]).
    out_arange = torch.arange(total, device=device)
    # seg id per output slot via searchsorted on the local indptr.
    seg = torch.searchsorted(indptr_local[1:], out_arange, right=True)  # [total]
    src = starts[seg] + (out_arange - indptr_local[seg])
    indices_local = kv_indices[src]
    return indptr_local, indices_local
