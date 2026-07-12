# SPDX-License-Identifier: Apache-2.0
"""Prepare/Finalize using mori dispatch_combine_v2 (FlyDSL/cco, gfx1250 wave32).

The production mori v1 (``mori.ops.EpDispatchCombineOp``) is authored for
gfx942/950 HIP kernels and does not run on gfx1250. dispatch_combine_v2 is the
gfx1250-capable cco/FlyDSL implementation. This module wires it into ATOM's
FusedMoEModularKernel as a drop-in replacement for MoriPrepareAndFinalize,
gated by ``ATOM_MORI_V2=1``.

Pipeline (mirrors the validated standalone test_moe_layer_ep.py):
    recv_x, recv_w, _, recv_idx, total_recv, routing = op.dispatch(
        a1, topk_weights, None, topk_ids, return_routing=True)
    dispatch_a1 = recv_x[:total_recv].clone()   # out of the cco VMM window
    fused_out = aiter.fused_moe(dispatch_a1, ...)   # driven by the modular kernel
    out, _ = op.combine(fused_out, routing=routing)

Shared experts are NOT fused in the mori EP+DP path (ATOM disables fusion there,
see topK.is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config), so
topk_ids carry only routed expert ids and mori routes them cleanly.
"""

import logging
import os
import sys
from functools import lru_cache
from typing import Any

import torch
import torch.distributed as dist

import atom.model_ops.fused_moe.modular_kernel as mk
from atom.model_ops.fused_moe.config import FusedMoEQuantConfig
from aiter import QuantType

try:
    import mori
    from mori.cco import Communicator

    MORI_AVAILABLE = True
except ImportError:  # pragma: no cover
    mori = None  # type: ignore
    Communicator = None  # type: ignore
    MORI_AVAILABLE = False

logger = logging.getLogger("atom")

# Populated lazily by _import_v2(); the v2 module is test-only packaged and its
# internal imports (``from intranode_kernels import ...``) require its own
# directory on sys.path.
EpDispatchCombineConfig = None
EpDispatchCombineOp = None
_V2_IMPORTED = False


def _import_v2() -> None:
    global EpDispatchCombineConfig, EpDispatchCombineOp, _V2_IMPORTED
    if _V2_IMPORTED:
        return
    if not MORI_AVAILABLE:
        raise ImportError("mori is required for MoriV2PrepareAndFinalize")
    v2_dir = os.path.join(
        os.path.dirname(mori.__file__), "ops", "dispatch_combine_v2"
    )
    if v2_dir not in sys.path:
        sys.path.insert(0, v2_dir)
    from dispatch_combine_op import (  # type: ignore  # noqa: E402
        EpDispatchCombineConfig as _Cfg,
        EpDispatchCombineOp as _Op,
    )

    EpDispatchCombineConfig = _Cfg
    EpDispatchCombineOp = _Op
    _V2_IMPORTED = True


@lru_cache(maxsize=1)
def _init_cco_comm(
    ep_size: int,
    ep_rank: int,
    ep_src_global_rank: int,
    per_rank_vmm: int,
) -> Any:
    """Collective: create a persistent cco Communicator over the EP group.

    The mori cco unique-id is generated on the EP leader and broadcast over the
    EP gloo cpu_group (mirrors mori.shmem.shmem_torch_process_group_init but for
    the cco fabric). All EP ranks must call this together.
    """
    from aiter.dist.parallel_state import get_ep_group

    ep = get_ep_group()
    uid = Communicator.get_unique_id() if ep_rank == 0 else None
    objs = [uid]
    dist.broadcast_object_list(objs, src=ep_src_global_rank, group=ep.cpu_group)
    uid = objs[0]
    comm = Communicator.init(ep_size, ep_rank, uid, per_rank_vmm=per_rank_vmm)
    comm.barrier()
    logger.info(
        "[MORI-V2] cco Communicator ready: ep_rank=%d ep_size=%d "
        "per_rank_vmm=%.2fGiB",
        ep_rank,
        ep_size,
        per_rank_vmm / (1 << 30),
    )
    return comm


@lru_cache(maxsize=4)
def init_mori_v2_op(
    ep_rank: int,
    ep_size: int,
    ep_src_global_rank: int,
    hidden_dim: int,
    max_num_inp_token_per_rank: int,
    num_local_experts: int,
    num_experts_per_token: int,
    data_type_itemsize: int,
) -> Any:
    """Create (and cache) a dispatch_combine_v2 op bound to the EP cco comm."""
    _import_v2()

    data_type = torch.bfloat16
    for dt in (torch.float8_e4m3fnuz, torch.float8_e4m3fn, torch.bfloat16):
        if dt.itemsize == data_type_itemsize:
            data_type = dt
            break

    # Size the cco symmetric VMM for the worst-case all-to-all: every rank could
    # send all its tokens to one peer -> ws * M recv slots, plus a 2x headroom
    # (tokens + combine buffers) and a fixed slack, matching test_moe_layer_ep.py.
    tok_bytes = max_num_inp_token_per_rank * hidden_dim * data_type.itemsize
    win_bytes = ep_size * tok_bytes * 2 + (1 << 24)
    per_rank_vmm = 2 * win_bytes + (1 << 28)

    comm = _init_cco_comm(ep_size, ep_rank, ep_src_global_rank, per_rank_vmm)

    cfg = EpDispatchCombineConfig(
        rank=ep_rank,
        world_size=ep_size,
        hidden_dim=hidden_dim,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        num_experts_per_rank=num_local_experts,
        num_experts_per_token=num_experts_per_token,
        data_type=data_type,
        combine_mode="gather",
        quant_type="none",
    )
    op = EpDispatchCombineOp(cfg, comm)
    comm.barrier()
    logger.info(
        "[MORI-V2] Created dispatch_combine_v2 op: ep_rank=%d ep_size=%d "
        "hidden=%d num_local_experts=%d topk=%d M=%d",
        ep_rank,
        ep_size,
        hidden_dim,
        num_local_experts,
        num_experts_per_token,
        max_num_inp_token_per_rank,
    )
    return op


class MoriV2PrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """Prepare/Finalize backed by mori dispatch_combine_v2 (sync path only)."""

    def __init__(
        self,
        mori_v2_op: Any,
        max_tokens_per_rank: int,
        num_dispatchers: int,
    ):
        if not MORI_AVAILABLE:
            raise ImportError(
                "mori is required for MoriV2PrepareAndFinalize but not installed."
            )
        super().__init__()
        self._op = mori_v2_op
        self.max_tokens_per_rank = max_tokens_per_rank
        self.num_dispatchers_ = num_dispatchers
        # Routing handle stashed between prepare() and finalize() of one forward.
        self._routing = None

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def output_is_reduced(self) -> bool:
        return True

    def num_dispatchers(self) -> int:
        return self.num_dispatchers_

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def supports_async(self) -> bool:
        return False

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig,
        quant_type: QuantType = QuantType.No,
    ) -> mk.PrepareResultType:
        assert (
            not apply_router_weight_on_input
        ), "mori does not support apply_router_weight_on_input=True now."

        # bf16 dispatch, no wire quant: scales=None. indices carry global expert
        # ids (0..global_num_experts-1); mori routes id -> rank = id // EPR.
        recv_x, recv_w, _recv_s, recv_idx, total_recv_t, routing = self._op.dispatch(
            a1,
            topk_weights.to(torch.float32),
            None,
            topk_ids.to(torch.int32),
            return_routing=True,
        )
        self._routing = routing
        total_recv = int(total_recv_t.item())

        # aiter FlyDSL kernels must not read cco symmetric VMM memory: clone the
        # dispatched tokens/routing out of the arena window before the GEMM.
        dispatch_a1 = recv_x[:total_recv].clone()
        dispatch_ids = recv_idx[:total_recv].clone()
        dispatch_weights = recv_w[:total_recv].clone()

        # num_local_tokens is left unset (expert_num_tokens=None): the grouped
        # a8w4 path derives per-expert routing from the (already trimmed) global
        # ids + expert_mask, exactly as test_moe_layer_ep.py does.
        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=None, expert_num_tokens_cpu=None
        )
        return (
            dispatch_a1,
            None,
            expert_tokens_meta,
            dispatch_ids,
            dispatch_weights,
        )

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        # topk_ids here is the ORIGINAL (pre-dispatch) routing, so shape[0] == ct.
        num_token = topk_ids.shape[0]
        assert self._routing is not None, "finalize() called before prepare()"
        out, _ = self._op.combine(fused_expert_output, routing=self._routing)
        self._routing = None
        return out[:num_token]


class MoriV2ModularKernel(mk.FusedMoEModularKernel):
    """Modular kernel for the v2 path: prepare() already trims to the exact
    received-token count, so skip the graph_bs-based dead-tail trim (which would
    otherwise cut valid rows)."""

    def _maybe_trim_dispatch_output(
        self,
        dispatch_a1: torch.Tensor,
        dispatch_scale: torch.Tensor | None,
        dispatch_ids: torch.Tensor,
        dispatch_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_tokens_meta,
    ):
        return dispatch_a1, dispatch_scale, dispatch_ids, dispatch_weights


def make_mori_v2_prepare_finalize(moe, all2all_manager) -> MoriV2PrepareAndFinalize:
    """Build a MoriV2PrepareAndFinalize for the given MoE config + EP group."""
    from aiter.dist.parallel_state import get_ep_group

    ep_group = get_ep_group()
    ep_src_global_rank = ep_group.ranks[0]

    op = init_mori_v2_op(
        ep_rank=all2all_manager.rank,
        ep_size=all2all_manager.world_size,
        ep_src_global_rank=ep_src_global_rank,
        hidden_dim=moe.hidden_dim,
        max_num_inp_token_per_rank=moe.max_num_tokens,
        num_local_experts=moe.num_experts // all2all_manager.world_size,
        num_experts_per_token=moe.experts_per_token,
        data_type_itemsize=moe.in_dtype.itemsize,
    )
    return MoriV2PrepareAndFinalize(
        op,
        max_tokens_per_rank=moe.max_num_tokens,
        num_dispatchers=all2all_manager.world_size,
    )
