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
from typing import NamedTuple, Optional

import torch


class PcpBalGroup(NamedTuple):
    """One request group for PCP+TBO balanced prefill.

    A prefill batch is split into request groups at request boundaries (never
    inside a sequence); each group is processed as an independent non-TBO PCP
    mini-batch (padded to a pcp multiple, round-robin striped, reindexed on its
    own). Consumed by ModelRunner.run_model (per-group stripe / restore) and the
    attn builder's `_build_ubatch_prefill_metadata_balanced` (slice + reindex).
    """

    req_start: int   # first request index of this group (inclusive)
    req_stop: int    # last request index of this group (exclusive)
    tok_start: int   # global token offset of the group's first token
    tok_end: int     # global token offset past the group's last REAL token
    pad_total: int   # tok count padded to a pcp multiple = pcp_pad_len(tok_end-tok_start, pcp)

from aiter.dist.parallel_state import (
    get_pcp_group,
    get_prefill_context_model_parallel_rank,
    get_prefill_context_model_parallel_world_size,
)

logger = logging.getLogger("atom")


def get_pcp_world_size() -> int:
    return get_prefill_context_model_parallel_world_size()


def get_pcp_rank() -> int:
    return get_prefill_context_model_parallel_rank()


def pcp_is_enabled() -> bool:
    return get_pcp_world_size() > 1


def pcp_pad_len(
    total_tokens: int,
    pcp_size: Optional[int] = None,
    num_ubatches: int = 1,
) -> int:
    """Padded token count so the global sequence is divisible by pcp_size * num_ubatches.

    Round-robin split requires the global token count to be divisible by pcp_size
    (see SGLang `can_dsa_cp_split` assert / HIP `apply_cp_reindex`). Returns the
    padded length (>= total_tokens); callers pad per-token tensors to this
    length with dummy tokens (KV length 0) before splitting.

    """
    if pcp_size is None:
        pcp_size = get_pcp_world_size()
    divisor = pcp_size * max(num_ubatches, 1)
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
    return input_.view(-1, pcp_size, *rest)[:, pcp_rank].contiguous()


def pcp_allgather_rerange(
    input_: torch.Tensor, pcp_size: Optional[int] = None
) -> torch.Tensor:
    """All-gather round-robin shards along dim 0 and restore original token order.

    Each rank holds `[L, *rest]` (its round-robin shard). After all-gather the
    naive layout is rank-major `[rank0_rows, rank1_rows, ...]`; the round-robin
    interleave is restored by `view(pcp, L, *rest).transpose(0, 1)` so that
    output[t] == global token t.

    Mirrors SGLang `cp_all_gather_rerange_output` (round-robin branch).
    """
    if pcp_size is None:
        pcp_size = get_pcp_world_size()
    if pcp_size <= 1:
        return input_
    group = get_pcp_group()
    # aiter all_gather(dim=0) returns rank-major concat: [pcp*L, *rest].
    gathered = group.all_gather(input_.contiguous(), dim=0)
    local_len = input_.shape[0]
    rest = tuple(input_.shape[1:])
    # rank-major [pcp, L, *rest] -> transpose -> token-major [L, pcp, *rest]
    # -> flatten to [L*pcp, *rest] == original global order.
    out = (
        gathered.view(pcp_size, local_len, *rest)
        .transpose(0, 1)
        .reshape(pcp_size * local_len, *rest)
    )
    return out


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


def pcp_all_reduce(input_: torch.Tensor, pcp_size: Optional[int] = None) -> torch.Tensor:
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
