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
from typing import NamedTuple

import numpy as np
import torch
from aiter.dist.parallel_state import (
    get_pcp_group,
    get_prefill_context_model_parallel_rank,
    get_prefill_context_model_parallel_world_size,
)

from atom.utils import upload_numpy


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


def get_pcp_world_size() -> int:
    return get_prefill_context_model_parallel_world_size()


def get_pcp_rank() -> int:
    return get_prefill_context_model_parallel_rank()


def pcp_is_enabled() -> bool:
    return get_pcp_world_size() > 1


def pcp_pad_len(
    total_tokens: int,
    pcp_size: int | None = None,
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
    input_: torch.Tensor, pcp_size: int | None = None, pcp_rank: int | None = None
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
    # Not `.contiguous()`: a one-row shard counts as contiguous while its row
    # stride is still pcp_size, which kernels that check strides reject (aiter's
    # cached-positions RoPE).
    return input_.view(-1, pcp_size, *rest)[:, pcp_rank].clone(
        memory_format=torch.contiguous_format
    )


def pcp_allgather_rerange(
    input_: torch.Tensor, pcp_size: int | None = None
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
    return _rerange(gathered, pcp_size, input_.shape[0])


def gather_rows(gather, rows):
    """`gather(list of tensors) -> list of tensors` over `rows`, one-byte
    dtypes carried as uint8: aiter's grouped `all_gatherv` copies bytes
    whatever the dtype, but its PyNCCL fallback has no fp8 / e8m0 / fp4x2."""
    gathered = gather(
        [t.view(torch.uint8) if t.element_size() == 1 else t for t in rows]
    )
    return tuple(g.view(t.dtype) for g, t in zip(gathered, rows))


def pcp_allgather_rerange_many(
    inputs: list[torch.Tensor], pcp_size: int | None = None
) -> list[torch.Tensor]:
    """`pcp_allgather_rerange` of every tensor in `inputs` (same row count) in
    one grouped collective."""
    if pcp_size is None:
        pcp_size = get_pcp_world_size()
    if pcp_size <= 1:
        return list(inputs)
    assert all(t.shape[0] == inputs[0].shape[0] for t in inputs), [
        t.shape[0] for t in inputs
    ]
    comm = get_pcp_group().device_communicator
    gathered = gather_rows(
        lambda ts: comm.all_gatherv(ts, dim=0), [t.contiguous() for t in inputs]
    )
    return [_rerange(g, pcp_size, t.shape[0]) for g, t in zip(gathered, inputs)]


def _rerange(gathered: torch.Tensor, pcp_size: int, local_len: int) -> torch.Tensor:
    # rank-major [pcp, L, *rest] -> transpose -> token-major [L, pcp, *rest]
    # -> flatten to [L*pcp, *rest] == original global order.
    rest = tuple(gathered.shape[1:])
    return (
        gathered.view(pcp_size, local_len, *rest)
        .transpose(0, 1)
        .reshape(pcp_size * local_len, *rest)
    )


# ==== MoE-path PCP collectives (rank-major gather + reduce_scatter) ====
# Rank-major all_gather + reduce_scatter are a mutually-inverse pair:
#   - gather (1/W -> full): all_gather(dim=0) concats rank-major, so rank r's
#     1/W stripe lands at rows [r*L:(r+1)*L]. MoE is per-token so the rank-major
#     (not global) order is fine.
#   - reduce_scatter (full partial-sum -> 1/W): sums the pcp-half across ranks
#     AND scatters dim0 back so rank r receives the summed chunk r == its own
#     original stripe tokens. No rerange/slice needed.


def pcp_reduce_scatter(
    input_: torch.Tensor, pcp_size: int | None = None
) -> torch.Tensor:
    """Sum the pcp-half across ranks and scatter dim0 back to this rank's 1/W
    stripe via a plain reduce_scatter (dim=0). Inverse of the rank-major gather
    (FusedMoE._run_experts gathers its routed rows that way)."""
    if pcp_size is None:
        pcp_size = get_pcp_world_size()
    if pcp_size <= 1:
        return input_
    return get_pcp_group().reduce_scatter(input_.contiguous(), dim=0)


def pcp_all_reduce(input_: torch.Tensor, pcp_size: int | None = None) -> torch.Tensor:
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
    n_global_q: int, pcp_size: int | None = None, pcp_rank: int | None = None
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


def pcp_owned_rows_by_request(
    cu_seqlens_q: np.ndarray, first: int, stride: int, owned: int
) -> np.ndarray:
    """Starts of the rows a PCP rank owns (`first`, `first + stride`, ...,
    `owned` of them) grouped by request (`cu_seqlens_q`). Padding rows join
    the last request's group: one group per `block_tables` row, and at
    position 0 they read no key."""
    cu = np.asarray(cu_seqlens_q, dtype=np.int64)
    below = (np.maximum(cu - first, 0) + stride - 1) // stride
    starts = below - below[0]
    starts[-1] = owned
    return starts.astype(np.int32)


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


def pcp_local_indptr(indptr: np.ndarray, owned: np.ndarray) -> np.ndarray:
    """Host side of `pcp_reindex_ragged`: the prefix sum of the `owned`
    queries' segment lengths in `indptr`. Owned ids past its last query are
    padding queries, whose segments are empty."""
    lens = np.diff(indptr)
    lens = np.pad(lens, (0, max(int(owned.max(initial=-1)) + 1 - lens.shape[0], 0)))
    out = np.zeros(owned.shape[0] + 1, dtype=np.int32)
    np.cumsum(lens[owned], out=out[1:])
    return out


def pcp_reindex_ragged(
    kv_indptr: torch.Tensor,  # [T_global + 1] int32 — global per-query prefix sum
    kv_indices: torch.Tensor,  # [kv_indptr[-1]] — ragged packed values
    owned_q: torch.Tensor,  # [T_local] long — global query ids this rank owns
    indptr_local_cpu: np.ndarray,  # [T_local + 1] — `pcp_local_indptr`
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

    `indptr_local` comes from the host (`indptr_local_cpu`), so sizing the
    output reads nothing back from the device.
    """
    device = kv_indptr.device
    owned_q = owned_q.to(device)
    starts = kv_indptr[owned_q]  # [T_local]
    indptr_local = upload_numpy(indptr_local_cpu, device)
    total = int(indptr_local_cpu[-1])
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
