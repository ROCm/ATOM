# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from functools import lru_cache
from typing import Any, Callable

import torch

import atom.model_ops.fused_moe.modular_kernel as mk
from atom.model_ops.fused_moe.config import FusedMoEQuantConfig
from atom.utils.forward_context import get_forward_context
from aiter import QuantType, dtypes

try:
    import mori

    MORI_AVAILABLE = True
except ImportError:
    mori = None  # type: ignore
    MORI_AVAILABLE = False

logger = logging.getLogger("atom")


_NUM_TBO_UBATCHES = 2


@lru_cache(maxsize=8)
def init_mori_op(
    rank: int,
    world_size: int,
    hidden_dim: int,
    scale_dim: int,
    max_num_inp_token_per_rank: int,
    num_local_experts: int,
    num_experts_per_token: int,
    gpu_per_node: int,
    data_type_itemsize: int,
    max_token_type_size: int,
    low_latency: bool = False,
    instance_id: int = 0,
) -> Any:
    """
    Create a mori op instance.
      - low_latency=True  → AsyncLL (dispatch_send/recv, combine_send/recv)
      - low_latency=False → IntraNode
    """
    import mori

    data_type = torch.float8_e4m3fnuz
    for dt in [torch.float8_e4m3fnuz, torch.float8_e4m3fn, torch.bfloat16]:
        if dt.itemsize == data_type_itemsize:
            data_type = dt
            break

    if low_latency:
        kernel_type = mori.ops.EpDispatchCombineKernelType.AsyncLL
        warp_num_per_block = 8
        block_num = 64
        rdma_block_num = 32
    elif world_size <= 8:
        kernel_type = mori.ops.EpDispatchCombineKernelType.IntraNode
        warp_num_per_block = 16
        block_num = 80
        rdma_block_num = 0
    else:
        kernel_type = mori.ops.EpDispatchCombineKernelType.InterNodeV1
        warp_num_per_block = 16
        block_num = 32
        rdma_block_num = 16

    mori_config = mori.ops.EpDispatchCombineConfig(
        rank=rank,
        world_size=world_size,
        data_type=data_type,
        hidden_dim=hidden_dim,
        scale_dim=scale_dim,
        scale_type_size=torch.float32.itemsize,
        max_token_type_size=max_token_type_size,
        max_num_inp_token_per_rank=max_num_inp_token_per_rank,
        num_experts_per_rank=num_local_experts,
        num_experts_per_token=num_experts_per_token,
        warp_num_per_block=warp_num_per_block,
        block_num=block_num,
        kernel_type=kernel_type,
        gpu_per_node=gpu_per_node,
        rdma_block_num=rdma_block_num,
        **({"num_qp_per_pe": 2} if low_latency else {}),
    )
    mori_op = mori.ops.EpDispatchCombineOp(mori_config)
    logger.info(
        f"[MORI] Created {kernel_type} mori_op instance_id={instance_id}: "
        f"{rank=} {world_size=} {hidden_dim=} {num_local_experts=} "
        f"{num_experts_per_token=}"
    )
    return mori_op


class MoriPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Prepare/Finalize using MoRI kernels.
    """

    def __init__(
        self,
        mori_op: Any,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        use_fp8_dispatch: bool = False,
        quant_type=None,
        quant_dtype: torch.dtype = None,
        is_async: bool = False,
        tbo_mori_ops: list | None = None,
        low_latency: bool = False,
    ):
        if not MORI_AVAILABLE:
            raise ImportError(
                "mori is required for MoriPrepareAndFinalize but not installed. "
                "Please install mori to use this feature."
            )
        super().__init__()
        self._sync_mori_op = mori_op
        self._tbo_mori_ops = tbo_mori_ops  # per-ubatch ops for TBO (IntraNode)
        self.num_dispatchers_ = num_dispatchers
        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch
        self.quant_type = quant_type
        self.quant_dtype = quant_dtype
        self._is_async = is_async
        self._low_latency = low_latency

    @property
    def activation_format(self) -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    def output_is_reduced(self) -> bool:
        return True

    def num_dispatchers(self):
        return self.num_dispatchers_

    def max_num_tokens_per_rank(self) -> int | None:
        return self.max_tokens_per_rank

    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def supports_async(self) -> bool:
        if not self._is_async:
            return False
        from atom.utils.tbo.ubatching import tbo_active

        return tbo_active()

    def _get_dispatch_config(self):
        """Return (block_num, warp_per_block) based on prefill vs decode."""
        context = get_forward_context().context
        if context.is_prefill:
            return 128, 16
        return 64, 4

    # ---- Synchronous (non-TBO) path ----

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
        """
        Returns a tuple of:
        - quantized + dispatched a.
        - Optional quantized + dispatched a1_scales.
        - Optional ExpertTokensMetadata containing gpu/cpu tensors
          as big as the number of local experts with the information about the
          number of tokens assigned to each local expert.
        - Optional dispatched expert topk IDs
        - Optional dispatched expert topk weight
        """
        assert (
            not apply_router_weight_on_input
        ), "mori does not support apply_router_weight_on_input=True now."
        scale = None
        if self.use_fp8_dispatch:
            from aiter import get_hip_quant

            quant_func = get_hip_quant(quant_type)
            a1, scale = quant_func(a1, quant_dtype=dtypes.fp8)

        block_num, warp_per_block = self._get_dispatch_config()

        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = self._sync_mori_op.dispatch(
            a1, topk_weights, scale, topk_ids, block_num, warp_per_block
        )

        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=dispatch_recv_token_num, expert_num_tokens_cpu=None
        )

        return (
            dispatch_a1,
            dispatch_scale,
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
        num_token = topk_ids.shape[0]

        block_num, warp_per_block = self._get_dispatch_config()

        result = self._sync_mori_op.combine(
            fused_expert_output,
            None,
            topk_ids,
            block_num,
            warp_per_block,
        )[0]
        return result[:num_token]

    # 1. IntraNode (default TBO): dispatch()/combine() on comm_stream
    # 2. AsyncLL (--low-latency): dispatch_send/recv, combine_send/recv (CU-free)
    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
    ) -> mk.ReceiverType:
        assert (
            not apply_router_weight_on_input
        ), "mori does not support apply_router_weight_on_input=True now."

        scale = None
        if self.use_fp8_dispatch:
            from aiter import get_hip_quant

            num_tokens = a1.shape[0]
            if num_tokens > 0:
                quant_func = get_hip_quant(QuantType.per_1x128)
                a1, scale = quant_func(a1, quant_dtype=dtypes.fp8)
            else:
                hidden_size = a1.shape[1] if a1.dim() > 1 else 0
                a1 = torch.empty(a1.shape, dtype=dtypes.fp8, device=a1.device)
                scale = torch.empty(
                    (0, hidden_size // 128),
                    dtype=torch.float32,
                    device=a1.device,
                )

        if self._low_latency:
            return self._prepare_async_ll(a1, topk_weights, topk_ids, scale)
        return self._prepare_async_comm_stream(a1, topk_weights, topk_ids, scale)

    def _prepare_async_ll(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        scale: torch.Tensor | None,
    ) -> tuple[Callable, mk.ReceiverType]:
        """AsyncLL path: dispatch_send (CU-free) → yield → dispatch_recv."""
        from atom.utils.tbo.ubatching import tbo_current_ubatch_id

        ubatch_id = tbo_current_ubatch_id()
        mori_op = self._tbo_mori_ops[ubatch_id]

        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = mori_op.dispatch_send(a1, topk_weights, scale, topk_ids)

        def hook():
            mori_op.dispatch_recv()

        def receiver() -> mk.PrepareResultType:
            expert_tokens_meta = mk.ExpertTokensMetadata(
                expert_num_tokens=dispatch_recv_token_num,
                expert_num_tokens_cpu=None,
            )
            return (
                dispatch_a1,
                dispatch_scale,
                expert_tokens_meta,
                dispatch_ids,
                dispatch_weights,
            )

        return (hook, receiver)

    def _prepare_async_comm_stream(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        scale: torch.Tensor | None,
    ) -> mk.ReceiverType:
        from atom.utils.tbo.ubatching import (
            tbo_current_ubatch_id,
            tbo_yield_and_switch_from_compute_to_comm,
            tbo_switch_to_compute_sync,
        )

        block_num, warp_per_block = self._get_dispatch_config()

        ubatch_id = tbo_current_ubatch_id()
        mori_op = self._tbo_mori_ops[ubatch_id]

        tbo_yield_and_switch_from_compute_to_comm()

        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = mori_op.dispatch(
            a1, topk_weights, scale, topk_ids, block_num, warp_per_block
        )

        tbo_switch_to_compute_sync()

        def receiver() -> mk.PrepareResultType:
            expert_tokens_meta = mk.ExpertTokensMetadata(
                expert_num_tokens=dispatch_recv_token_num,
                expert_num_tokens_cpu=None,
            )
            return (
                dispatch_a1,
                dispatch_scale,
                expert_tokens_meta,
                dispatch_ids,
                dispatch_weights,
            )

        return receiver

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> Callable:
        num_token = topk_ids.shape[0]
        if self._low_latency:
            return self._finalize_async_ll(num_token, fused_expert_output, topk_ids)
        return self._finalize_async_comm_stream(
            num_token,
            fused_expert_output,
            topk_ids,
        )

    def _finalize_async_ll(
        self,
        num_token: int,
        fused_expert_output: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[Callable, Callable]:
        """AsyncLL path: combine_send (CU-free) → yield → combine_recv."""
        from atom.utils.tbo.ubatching import tbo_current_ubatch_id

        ubatch_id = tbo_current_ubatch_id()
        mori_op = self._tbo_mori_ops[ubatch_id]

        combined_hidden_states = mori_op.combine_send(
            fused_expert_output, None, topk_ids
        )

        def hook():
            mori_op.combine_recv()

        def receiver():
            return combined_hidden_states[0][:num_token]

        return (hook, receiver)

    def _finalize_async_comm_stream(
        self,
        num_token: int,
        fused_expert_output: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> Callable:
        from atom.utils.tbo.ubatching import (
            tbo_current_ubatch_id,
            tbo_yield_and_switch_from_compute_to_comm,
            tbo_switch_to_compute_sync,
        )

        block_num, warp_per_block = self._get_dispatch_config()

        ubatch_id = tbo_current_ubatch_id()
        mori_op = self._tbo_mori_ops[ubatch_id]

        # Yield to other thread FIRST, then switch to comm stream.
        tbo_yield_and_switch_from_compute_to_comm()

        result = mori_op.combine(
            fused_expert_output,
            None,
            topk_ids,
            block_num,
            warp_per_block,
        )[0]

        tbo_switch_to_compute_sync()

        def receiver():
            return result[:num_token]

        return receiver
