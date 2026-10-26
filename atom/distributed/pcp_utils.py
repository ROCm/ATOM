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
from functools import cache
from typing import NamedTuple

import numpy as np
import torch
from aiter.dist.parallel_state import (
    get_pcp_group,
    get_prefill_context_model_parallel_rank,
    get_prefill_context_model_parallel_world_size,
    get_tp_group,
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


class PcpSharedGrid(NamedTuple):
    """How a pcp-redundant per-token module is cut. See `pcp_merged_tp_grid`.

    `size` ways, this rank holding shard `rank`, and `prescale` the factor its
    output needs before the tp/pcp collectives sum it -- `size / (pcp * tp)`,
    which is 1.0 exactly when the cut uses the whole folded grid.
    """

    size: int
    rank: int
    prescale: float


logger = logging.getLogger("atom")


@cache
def _warn_unaligned_fold(shard_dim: int, pcp_size: int, full: int, align: int) -> None:
    """Once per distinct shape, not once per layer -- every block asks."""
    logger.warning(
        "PCP merge: leaving a %d-wide tensor replicated across the %d pcp "
        "ranks -- no divisor of the %d-wide folded grid cuts it into "
        "%d-aligned pieces, so it keeps the 1/pcp_size pre-scale instead.",
        shard_dim,
        pcp_size,
        full,
        align,
    )


def get_pcp_world_size() -> int:
    return get_prefill_context_model_parallel_world_size()


def get_pcp_rank() -> int:
    return get_prefill_context_model_parallel_rank()


def pcp_is_enabled() -> bool:
    return get_pcp_world_size() > 1


def pcp_split_active(dsa_fallback_topk: int | None) -> bool:
    """Did PCP row-split the CURRENT prefill forward?

    The mechanism every model shares -- pcp enabled, real prefill, not a
    dummy/warmup run -- lives here once. The one genuine difference stays a
    parameter: a backend that falls back to dense MHA on short batches is not
    PCP-wired there, so it must pass its threshold and gets False below it.
    `aiter_mla.py` (the V3.2 / deepseek_v2 family) has that fallback keyed on
    `max_seqlen_k > index_topk`; DeepSeek-V4's attention has no such branch, so
    it passes None and every prefill counts.

    Merging the two conditions instead of parameterising them would be wrong in
    both directions: V4 would wrongly report "not split" for prompts under its
    own index_topk, and the v2 family would wrongly report "split" for batches
    that never reached the sparse path.

    Reads only batch-global state, so every PCP rank gets the same answer --
    which is what lets a caller gate a collective on it. It is NOT stable across
    a step: a drafter that rewrites `is_prefill` or bumps `max_seqlen_k` changes
    the answer, so evaluate it while the target's own forward context still
    stands, not partway through drafting.
    """
    from atom.utils.forward_context import get_forward_context

    if not pcp_is_enabled():
        return False
    ctx = get_forward_context()
    context = getattr(ctx, "context", None)
    if context is None:
        return False
    if not bool(context.is_prefill) or bool(getattr(context, "is_dummy_run", False)):
        return False
    if dsa_fallback_topk is None:
        return True
    attn_metadata = getattr(ctx, "attn_metadata", None)
    if attn_metadata is None:
        return False
    return int(getattr(attn_metadata, "max_seqlen_k", 0)) > int(dsa_fallback_topk)


def pcp_merged_tp_grid(
    shard_dim: int | None = None, align: int = 1
) -> PcpSharedGrid | None:
    """How to cut a pcp-redundant per-token module, or None to leave it on TP.

    Lets a module that is otherwise replicated across PCP shard on (a divisor
    of) the same `pcp_size * tp_size` grid FusedMoE folds onto, so the existing
    tp all_reduce + pcp reduce_scatter/all_reduce sum its partials for free.

    The counting argument is the whole correctness story. Those two collectives
    between them sum all `pcp_size * tp_size` rank contributions. Cut `g` ways
    with shard index `flat_rank % g`, and `g | pcp*tp` makes every shard land on
    exactly `pcp*tp/g` ranks, so the sum comes back `pcp*tp/g` too large --
    hence `prescale`. The two hand-written cases this generalises: `g == pcp*tp`
    needs no scale, and `g == tp_size` (replicated across PCP, i.e. this
    returning None) is the familiar `1/pcp_size`.

    `shard_dim` with `align` is what keeps `g` from outrunning the checkpoint:
    block-quantised scales only slice on `align` boundaries, so a `g` that cuts
    `shard_dim` into non-`align` pieces would fault in the weight loader. We
    step down to the largest divisor that stays clean rather than failing, and
    return None once that is just the TP group -- the path the model has always
    taken. `flat_rank` stays `pcp_rank * tp_size + tp_rank`, matching
    `FusedMoEParallelConfig.make`, so a TP group owns a contiguous run and its
    PCP partner the next one.

    One level of that fold is deliberately NOT mirrored: `make` folds DP into TP
    first when `flatten_tp_across_dp_for_moe`. A module summed only by the tp
    all_reduce + the pcp collective never gets a DP reduce, so following it
    there would shard into pieces nothing adds back up. That combination is
    refused below instead.
    """
    from atom.config import get_current_atom_config
    from atom.utils import envs

    pcp_size = get_pcp_world_size()
    if pcp_size <= 1 or not bool(envs.ATOM_PCP_MOE_MERGE):
        return None
    cfg = get_current_atom_config()
    # Same disjunction as moe.py's `flatten_tp_across_dp_for_moe`. The first
    # half is already refused at startup (llm_engine.py, PCP + DP-attention);
    # the second defaults off and only the vLLM plugin sets it, so this guards
    # a door that is shut rather than one anyone walks through today.
    if getattr(cfg, "enable_dp_attention", False) or getattr(
        cfg, "moe_ep_flatten_tp_across_dp", False
    ):
        raise ValueError(
            "ATOM_PCP_MOE_MERGE=1 cannot be combined with MoE's "
            "flatten-TP-across-DP (enable_dp_attention or "
            "moe_ep_flatten_tp_across_dp): the routed experts would shard "
            "across pcp*dp*tp while the pcp-folded modules shard across "
            "pcp*tp, and no collective sums the dp dimension back for the "
            "latter. Set ATOM_PCP_MOE_MERGE=0 or drop the flatten."
        )
    tp_size = get_tp_group().world_size
    tp_rank = 0 if tp_size == 1 else get_tp_group().rank_in_group
    full = pcp_size * tp_size

    size = tp_size
    for cand in range(full, tp_size - 1, -1):
        if full % cand:
            continue
        if shard_dim is None or (
            shard_dim % cand == 0 and (shard_dim // cand) % align == 0
        ):
            size = cand
            break
    if size <= tp_size:
        # Nothing the TP group does not already cut -- stay on it.
        if shard_dim is not None and full > tp_size:
            _warn_unaligned_fold(shard_dim, pcp_size, full, align)
        return None
    return PcpSharedGrid(
        size=size,
        rank=(get_pcp_rank() * tp_size + tp_rank) % size,
        prescale=size / full,
    )


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
    return input_.view(-1, pcp_size, *rest)[:, pcp_rank].contiguous()


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
    input_: torch.Tensor, pcp_size: int | None = None
) -> torch.Tensor:
    """Gather this rank's 1/W stripe shard into the full rank-major sequence
    via a plain all_gather (dim=0). Inverse of pcp_reduce_scatter."""
    if pcp_size is None:
        pcp_size = get_pcp_world_size()
    if pcp_size <= 1:
        return input_
    return get_pcp_group().all_gather(input_.contiguous(), dim=0)


def pcp_reduce_scatter(
    input_: torch.Tensor, pcp_size: int | None = None
) -> torch.Tensor:
    """Sum the pcp-half across ranks and scatter dim0 back to this rank's 1/W
    stripe via a plain reduce_scatter (dim=0). Inverse of pcp_allgather_rankmajor."""
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
    `owned` of them) grouped by request (`cu_seqlens_q`), the padding rows
    past the last request folded into its group. Rows below `x` it owns
    number `ceil((x - first) / stride)`, so this is arithmetic on the host.

    One group per request, never one more: the FP4 scorer reads group b's
    keys off `block_tables[b]`, which has a row per request. Padding rows sit
    at position 0, so their bound is 0 and they read no key."""
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
