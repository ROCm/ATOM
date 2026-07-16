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

from typing import Optional

import torch

from aiter.dist.parallel_state import (
    get_pcp_group,
    get_prefill_context_model_parallel_rank,
    get_prefill_context_model_parallel_world_size,
)


def plugin_attn_cp_enabled() -> bool:
    """True when the vLLM plugin reuses the TP group as the Context-Parallel
    group (env ``ATOM_VLLM_ATTN_CP``, RFC ROCm/ATOM#196).

    This is distinct from :func:`pcp_is_enabled`, which reflects the aiter
    ``_PCP`` group world size and is *also* true for native ATOM's separate
    ``-pcp`` dimension. ``plugin_attn_cp_enabled`` specifically flags the
    reuse-TP-as-CP true sequence-parallel path, whose construction-time model
    changes (full-head *replicated* attention, MoE all-gather/reduce-scatter,
    disabled fused-allreduce norms) must fire *before* the aiter ``_PCP`` group
    is aliased. It therefore keys off the static env flag, not the group world.

    The round-robin split/gather/reindex machinery is shared between native PCP
    and plugin CP and stays gated on :func:`pcp_is_enabled` / ``_pcp_active``.
    """
    from atom.utils import envs

    return bool(envs.ATOM_VLLM_ATTN_CP)


def get_pcp_world_size() -> int:
    return get_prefill_context_model_parallel_world_size()


def get_pcp_rank() -> int:
    return get_prefill_context_model_parallel_rank()


def pcp_is_enabled() -> bool:
    return get_pcp_world_size() > 1


def pcp_pad_len(
    total_tokens: int,
    pcp_size: Optional[int] = None,
) -> int:
    """Padded token count so the global sequence is divisible by pcp_size.

    Round-robin split requires the global token count to be divisible by pcp_size
    (see SGLang `can_dsa_cp_split` assert / HIP `apply_cp_reindex`). Returns the
    padded length (>= total_tokens); callers pad per-token tensors to this
    length with dummy tokens (KV length 0) before splitting.
    """
    if pcp_size is None:
        pcp_size = get_pcp_world_size()
    if pcp_size <= 1:
        return total_tokens
    rem = total_tokens % pcp_size
    if rem == 0:
        return total_tokens
    return total_tokens + (pcp_size - rem)


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


def pcp_sparse_prefill_reindex(
    sparse_seqlen: torch.Tensor,  # [T] per-query selected-KV length (pre-clamp)
    req_id_per_token: torch.Tensor,  # [T] per-query request id
    slot_mapping: torch.Tensor,  # [T] full per-query slot mapping (KV write)
    index_topk: int,
    pcp_size: Optional[int] = None,
    pcp_rank: Optional[int] = None,
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
    padded_total = pcp_pad_len(s_real, pcp_size)
    n_pad = padded_total - s_real
    owned_q = pcp_round_robin_query_indices(padded_total, pcp_size, pcp_rank).to(device)
    n_owned = int(owned_q.shape[0])

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
