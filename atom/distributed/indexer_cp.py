# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax-M3 indexer-only context parallel: group access and the exchange.

The indexer scores every 128-token block to pick the top-k that sparse attention
will read. Under TP the index-Q heads shard like KV heads, so at TP4 each rank
gets one head and the decode scorer's ``tl.dot`` runs with M=1 against a
hardware floor of 16. Here every rank instead scores ALL index heads over 1/P of
the blocks (round-robin, ``owner(block) = block % P``) -- same multiply count,
full MMA occupancy -- and an all-to-all routes each head's top-k candidates to
the rank that owns that head. Sparse attention, KV geometry and MoE stay TP.

Scope: v1 requires ``tp_size == num_kv_heads``, which makes the CP group exactly
the TP group. That is why there is no ``new_group`` here: rank r already owns kv
head ``r // num_kv_head_replicas`` (``linear.py``), and with one replica per head
a rank's TP position IS the index head it owns, so the all-to-all's implicit
"chunk j goes to group rank j" mapping is already the right routing. Widening to
TP > num_kv_heads means a strided subgroup per head-set and a collective
``new_group`` on every rank; not wired.

This module is ATOM-native only -- the vLLM/SGLang bridges keep the TP path.
"""

import torch
import torch.distributed as dist

from atom.config import get_current_atom_config
from atom.plugin.prepare import is_plugin_mode
from atom.utils import envs


def indexer_cp_enabled() -> bool:
    """True when M3 indexer-only CP is on and validated for this deployment.

    ``Config.__post_init__`` has already cleared the flag (with a warning) for
    every topology ``indexer_cp_unsupported_reason`` rejects, so reading it here
    needs no re-validation.

    Plugin mode is excluded here rather than documented as a convention, because
    getting it wrong is silent. CP widens ``index_q`` to every index head, and
    both bridges reshape it with a ``view`` on the width they assume
    (``plugin/vllm/attention/minimax_m3_attnetion.py:220``,
    ``plugin/sglang/attention_backend/minimax_m3_sparse.py:869``) -- a 4-head
    tensor viewed as 1 head yields 4x the rows and no error at all. The bridges
    keep the TP path, so the flag must not be able to reach them.
    """
    if is_plugin_mode():
        return False
    config = get_current_atom_config()
    return bool(getattr(config.dcp_config, "indexer_dcp_only", False))


def get_indexer_cp_group():
    """The process group the indexer shards context over.

    Exactly the TP group: see the module docstring for why v1 requires the
    square ``tp_size == num_kv_heads`` case.
    """
    from aiter.dist.parallel_state import get_tp_group

    return get_tp_group()


def get_indexer_cp_rank() -> int:
    """This rank's shard index, which is also the index head it owns."""
    return get_indexer_cp_group().rank_in_group


def get_indexer_cp_world_size() -> int:
    return get_indexer_cp_group().world_size


# Above this many bytes of all-to-all payload, RCCL's launch overhead is
# amortized and the all-gather's 4x traffic is the dominant term; below it the
# overhead dominates and the wider-but-cheaper-to-launch collective wins.
#
# Measured on 4xMI355 inside a CUDA graph, us per exchange, sandwiched between
# two kernels exactly as production runs it (_bench_a2a_vs_allgather.py):
#
#   payload   topk=32          topk=64          topk=128
#   8KB       22.1 -> 11.2     --               --
#   128KB     14.9 -> 13.0     17.9 -> 15.5     14.9 -> 13.0
#   512KB     20.6 -> 20.2     20.8 -> 20.1     20.7 -> 20.2   <- crossover
#   640KB     --               21.1 -> 23.2     20.6 -> 22.2
#   1024KB    21.4 -> 29.2     25.7 -> 48.3     21.3 -> 29.2
#
# The crossover lands at 512KB for all three top-k values, i.e. it tracks total
# BYTES and not the tokens/topk factorization -- which is why the threshold is
# expressed in bytes rather than as a token count.
_ALLGATHER_MAX_PAYLOAD_BYTES = 512 * 1024


def exchange_candidates(keys: torch.Tensor) -> torch.Tensor:
    """Route each head's local candidates to the rank that owns that head.

    ``keys`` is [heads, tokens, topk] packed sort keys, head-major so that
    ``all_to_all_single``'s split of dim 0 sends head j to group rank j with no
    permutation. The return is [source_shard, tokens, topk] for this rank's one
    head, which ``merge_candidate_keys`` reads with a stride rather than
    copying.

    The payload is ``world * topk`` keys per token regardless of context length,
    which is what makes this cheaper than exchanging full score rows.

    Two transports, picked by payload size. The all-to-all is the minimal one --
    it moves exactly the keys this rank needs -- but RCCL's generic device
    kernel leaves ~5us of idle GPU on EACH side of itself inside a captured
    graph (measured: 5.3us before, 4.9us after, 2.87ms over the 300 calls in one
    decode trace, which is MORE than the 2.75ms the collective itself costs).
    aiter's registered-buffer collectives do not: in the same graph on the same
    stream, its all-reduce of comparable duration leaves 0.00us on both sides.
    Below the crossover that ~10us bubble outweighs sending 4x the bytes.
    """
    group = get_indexer_cp_group()
    if _can_all_gather(keys, group):
        return _exchange_via_all_gather(keys, group)
    received = torch.empty_like(keys)
    dist.all_to_all_single(received, keys, group=group.device_group)
    return received


def _can_all_gather(keys: torch.Tensor, group) -> bool:
    """Whether this exchange takes the all-gather transport.

    EVERY TERM MUST BE RANK-INVARIANT, and that is this function's whole reason
    to exist rather than an incidental property. The two transports are
    different collectives on the same group: if one rank answered True and
    another False, the first would sit in ``custom_all_gather`` while the second
    sat in ``all_to_all_single``, and the group would deadlock with no error at
    all. An earlier version wrapped the collective in ``except (AssertionError,
    RuntimeError, AttributeError): return None`` and fell back per rank, which
    is precisely that hazard -- a per-rank exception is not a per-rank
    recoverable event when the thing that raised is a collective.

    So the decision reads only inputs that are identical on every rank by
    construction, and nothing here can raise:

    * ``ATOM_USE_CUSTOM_ALL_GATHER`` -- the repo-wide opt-out for aiter's custom
      gather (``embed_head.py`` passes it as ``use_custom``). It is set
      process-wide, never per rank. Honouring it is also what makes this
      transport switchable at all: without it, an operator who disabled the
      custom gather everywhere else still got it here.
    * the group's ``ca_comm`` and its ``disabled`` flag -- both fixed when the
      group was built, from world size and topology, which every rank shares.
    * ``should_custom_ag`` -- a pure function of payload bytes and contiguity.
      ``keys`` is [heads, tokens, topk], the same shape on every rank.

    The byte threshold is checked first because it is the performance decision;
    the rest is availability. When availability says no, the all-to-all is
    correct at every size -- just with the bubble documented above.
    """
    if keys.numel() * keys.element_size() > _ALLGATHER_MAX_PAYLOAD_BYTES:
        return False
    if not envs.ATOM_USE_CUSTOM_ALL_GATHER:
        return False
    ca_comm = getattr(getattr(group, "device_communicator", None), "ca_comm", None)
    if ca_comm is None or ca_comm.disabled:
        return False
    return bool(ca_comm.should_custom_ag(keys.view(torch.int32)))


def _exchange_via_all_gather(keys: torch.Tensor, group) -> torch.Tensor:
    """The same routing as the all-to-all, via aiter's bubble-free all-gather.

    Every rank receives every shard's candidates for every head, and then reads
    only its own head -- so this returns the same [source_shard, tokens, topk]
    view the all-to-all writes, sliced out of a 4x larger buffer with no copy.
    ``merge_candidate_keys`` already reads its input through ``SRC_STRIDE``, so
    a strided view costs it nothing.

    Callers must clear ``_can_all_gather`` first. There is deliberately no
    try/except here: see that function for why a per-rank fallback around a
    collective deadlocks rather than recovers.
    """
    world, tokens, topk = keys.shape
    # int64 must be viewed as int32 pairs rather than passed through: aiter's
    # own _INT_TO_FP_VIEW maps int64 -> float64, and float64 is missing from
    # _aiter_dtype_id's table, so the int64 path raises "Unsupported dtype:
    # torch.float64" (aiter/utility/dtypes.py:57). The kernel is a pure memcpy
    # parametrized only by sizeof(T), so splitting each key into two int32 lanes
    # is value-preserving.
    gathered = group.custom_all_gather(keys.view(torch.int32))
    # [world(src), world(head), tokens, topk] -- take this rank's head from
    # every source shard.
    return gathered.view(torch.int64).view(world, world, tokens, topk)[
        :, group.rank_in_group
    ]


def warmup_exchange(device) -> None:
    """Run one eager all-to-all so capture never sees NCCL's lazy setup.

    NCCL establishes peer connections on a group's first collective. Doing that
    inside a CUDA graph capture hangs, so this must run before the capture loop
    -- it is a correctness requirement, not a warm cache. The model warmup pass
    does NOT cover it: that pass is prefill-only, and prefill stays on the TP
    path, which issues no all-to-all at all.

    The payload is one int64 per peer. Connection setup is per-group, not
    per-shape, and an all-to-all touches every peer whatever the size -- so the
    only thing that matters is that every rank passes the SAME shape, which
    taking no shape argument guarantees.

    BOTH transports are warmed explicitly rather than by calling
    ``exchange_candidates``. A tiny probe routes to the all-gather branch, so
    going through the dispatcher would warm aiter and leave RCCL cold -- and
    then a large batch at capture time would hit the all-to-all's lazy setup
    inside the graph, which is exactly the hang this function exists to prevent.

    The all-gather half still goes through ``_can_all_gather``, so a deployment
    that disabled aiter's custom gather warms only what it will actually run
    instead of raising here. That predicate is monotonic in payload size, so
    clearing it for the probe implies clearing it for the larger production
    payload -- the warmup cannot pass for a transport production then skips.
    """
    if not indexer_cp_enabled():
        return
    world = get_indexer_cp_world_size()
    probe = torch.zeros((world, 1, 1), dtype=torch.int64, device=device)
    group = get_indexer_cp_group()
    received = torch.empty_like(probe)
    dist.all_to_all_single(received, probe, group=group.device_group)
    if _can_all_gather(probe, group):
        _exchange_via_all_gather(probe, group)
    torch.cuda.synchronize()
