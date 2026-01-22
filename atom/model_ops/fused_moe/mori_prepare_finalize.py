# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

from atom.utils.forward_context import get_forward_context
import torch

import atom.model_ops.fused_moe.modular_kernel as mk
from atom.model_ops.fused_moe.config import FusedMoEQuantConfig
from aiter import dtypes
from aiter import QuantType
from atom.utils.dbo.ubatching import (
    dbo_current_ubatch_id,
    dbo_enabled,
    dbo_yield,
    dbo_switch_to_comm,
    dbo_switch_to_comm_sync,
    dbo_switch_to_compute,
    dbo_switch_to_compute_sync,
    dbo_yield_and_switch_from_comm_to_compute,
    dbo_yield_and_switch_from_compute_to_comm,
)

# Lazy import mori
try:
    import mori
    MORI_AVAILABLE = True
except ImportError:
    mori = None  # type: ignore
    MORI_AVAILABLE = False


@dataclass
class DispatchIntermediateStateAsyncLL:
    packed_recv_hidden: torch.Tensor
    recv_topk_weights: torch.Tensor
    recv_scales: torch.Tensor | None
    recv_topk_ids: torch.Tensor
    packed_recv_count: torch.Tensor
    origin_topk_weights: torch.Tensor
    origin_topk_ids: torch.Tensor


@dataclass
class CombineIntermediateStateAsyncLL:
    combined_hidden_states: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    output: torch.Tensor
    num_token: int


class MoriPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Supports two modes:
    1. Standard mode: uses dispatch() and combine() API (original behavior)
    2. AsyncLL mode: uses dispatch_send/recv and combine_send/recv API for true overlap
    """

    def __init__(
        self,
        mori_op: Any,  # mori.ops.EpDispatchCombineOp when mori is available
        max_tokens_per_rank: int,
        num_dispatchers: int,
        use_fp8_dispatch: bool = False,
        quant_type=None,
        quant_dtype: torch.dtype = None,
    ):
        if not MORI_AVAILABLE:
            raise ImportError(
                "mori is required for MoriPrepareAndFinalize but not installed. "
                "Please install mori to use this feature."
            )
        super().__init__()
        self.mori_op = mori_op
        self.num_dispatchers_ = num_dispatchers
        self.max_tokens_per_rank = max_tokens_per_rank
        self.use_fp8_dispatch = use_fp8_dispatch
        self.quant_type = quant_type
        self.quant_dtype = quant_dtype
        
        # Check if mori supports AsyncLL mode
        self._use_async_ll = self._check_async_ll_support()
        
        if self._use_async_ll:
            self._dispatch_states_ll: list[DispatchIntermediateStateAsyncLL | None] = [None, None]
            self._combine_states_ll: list[CombineIntermediateStateAsyncLL | None] = [None, None]
        else:
            self.handles = [None, None]

    def _check_async_ll_support(self) -> bool:
        """Check if mori_op supports AsyncLL (Low Latency) mode."""
        try:
            if hasattr(self.mori_op, 'config') and hasattr(self.mori_op.config, 'kernel_type'):
                return self.mori_op.config.kernel_type is mori.ops.EpDispatchCombineKernelType.AsyncLL
        except:
            pass
        return False

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
        return True

    def _dispatch_a_async_ll(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        scale: torch.Tensor | None,
    ) -> None:
        """AsyncLL Phase A: call dispatch_send (non-blocking)."""
        ubatch_id = dbo_current_ubatch_id()
        
        (
            packed_recv_hidden,
            recv_topk_weights,
            recv_scales,
            recv_topk_ids,
            packed_recv_count,
        ) = self.mori_op.dispatch_send(a1, topk_weights, scale, topk_ids)
        
        self._dispatch_states_ll[ubatch_id] = DispatchIntermediateStateAsyncLL(
            packed_recv_hidden=packed_recv_hidden,
            recv_topk_weights=recv_topk_weights,
            recv_scales=recv_scales,
            recv_topk_ids=recv_topk_ids,
            packed_recv_count=packed_recv_count,
            origin_topk_weights=topk_weights,
            origin_topk_ids=topk_ids,
        )

    def _dispatch_b_async_ll(self) -> mk.PrepareResultType:
        """AsyncLL Phase B: call dispatch_recv to complete."""
        ubatch_id = dbo_current_ubatch_id()
        
        state = self._dispatch_states_ll[ubatch_id]
        if state is None:
            raise RuntimeError(f"dispatch_b called without dispatch_a for ubatch {ubatch_id}")
        self._dispatch_states_ll[ubatch_id] = None
        
        self.mori_op.dispatch_recv()
        
        expert_tokens_meta = mk.ExpertTokensMetadata(
            expert_num_tokens=state.packed_recv_count,
            expert_num_tokens_cpu=None,
        )
        
        return (
            state.packed_recv_hidden,
            state.recv_scales,
            expert_tokens_meta,
            state.recv_topk_ids,
            state.recv_topk_weights,
        )

    def _combine_a_async_ll(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        """AsyncLL Phase A: call combine_send (non-blocking)."""
        ubatch_id = dbo_current_ubatch_id()
        
        combined_hidden_states = self.mori_op.combine_send(
            fused_expert_output, None, topk_ids
        )
        
        self._combine_states_ll[ubatch_id] = CombineIntermediateStateAsyncLL(
            combined_hidden_states=combined_hidden_states,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            output=output,
            num_token=output.shape[0],
        )

    def _combine_b_async_ll(self) -> None:
        """AsyncLL Phase B: call combine_recv to complete."""
        ubatch_id = dbo_current_ubatch_id()
        
        state = self._combine_states_ll[ubatch_id]
        if state is None:
            raise RuntimeError(f"combine_b called without combine_a for ubatch {ubatch_id}")
        self._combine_states_ll[ubatch_id] = None
        
        self.mori_op.combine_recv()
        
        result = state.combined_hidden_states[0] if isinstance(state.combined_hidden_states, tuple) else state.combined_hidden_states
        state.output.copy_(result[:state.num_token], non_blocking=True)

    def _do_dispatch(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        scale: torch.Tensor | None,
        topk_ids: torch.Tensor,
    ) -> Callable:
        """
        Standard mode dispatch using original API.
        """
        # Yield before dispatch to allow other ubatch's compute to be queued
        dbo_yield_and_switch_from_compute_to_comm()

        context = get_forward_context().context
        if context.is_prefill:
            block_num = 128
            warp_per_block = 16
        else:
            block_num = 64
            warp_per_block = 4

        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = self.mori_op.dispatch(
            a1, topk_weights, scale, topk_ids, block_num, warp_per_block
        )

        handle = True
        a2a_idx = dbo_current_ubatch_id()
        self.handles[a2a_idx] = handle

        dbo_switch_to_compute_sync()

        def _receiver() -> mk.PrepareResultType:
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

        return _receiver

    def _do_finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
        do_async: bool,
    ) -> Callable | None:
        """Standard mode finalize using original API."""
        num_token = output.shape[0]

        a2a_idx = dbo_current_ubatch_id()
        handle = self.handles[a2a_idx]
        self.handles[a2a_idx] = None

        if handle is None and dbo_enabled():
            raise RuntimeError(
                f"finalize called without matching prepare for ubatch {a2a_idx}"
            )

        # dbo_yield_and_switch_from_compute_to_comm()
        dbo_switch_to_comm_sync()
        
        context = get_forward_context().context
        if context.is_prefill:
            block_num = 128
            warp_per_block = 16
        else:
            block_num = 64
            warp_per_block = 4

        result = self.mori_op.combine(
            fused_expert_output,
            None,
            topk_ids,
            block_num,
            warp_per_block,
        )[0]

        dbo_switch_to_compute()

        if do_async:
            def _receiver():
                dbo_switch_to_comm()
                output.copy_(result[:num_token], non_blocking=True)
                dbo_yield_and_switch_from_comm_to_compute()

            return _receiver
        else:
            output.copy_(result[:num_token])
            return None

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig = None,
        quant_type: QuantType = QuantType.No,
    ) -> mk.PrepareResultType | mk.ReceiverType:
        """
        Async prepare - uses AsyncLL if available, otherwise standard mode.
        """
        assert not apply_router_weight_on_input, (
            "mori does not support apply_router_weight_on_input=True"
        )

        scale = None
        if self.use_fp8_dispatch:
            from aiter import get_hip_quant
            quant_func = get_hip_quant(self.quant_type)
            a1, scale = quant_func(a1, quant_dtype=dtypes.fp8)

        if self._use_async_ll:
            self._dispatch_a_async_ll(a1, topk_weights, topk_ids, scale)
            dbo_yield()
            return self._dispatch_b_async_ll()
        else:
            return self._do_dispatch(a1, topk_weights, scale, topk_ids)

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> Callable | None:

        if self._use_async_ll:
            self._combine_a_async_ll(output, fused_expert_output, topk_ids, topk_weights)
            dbo_yield()
            self._combine_b_async_ll()
            dbo_yield()
            return None  # Already completed
        else:
            return self._do_finalize(
                output, fused_expert_output, topk_weights, topk_ids,
                apply_router_weight_on_input, do_async=True
            )

    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_config: FusedMoEQuantConfig = None,
        quant_type: QuantType = QuantType.No,
    ) -> mk.PrepareResultType:
        assert not apply_router_weight_on_input, (
            "mori does not support apply_router_weight_on_input=True"
        )

        scale = None
        if self.use_fp8_dispatch:
            from aiter import get_hip_quant
            quant_func = get_hip_quant(self.quant_type)
            a1, scale = quant_func(a1, quant_dtype=dtypes.fp8)

        context = get_forward_context().context
        if context.is_prefill:
            block_num = 128
            warp_per_block = 16
        else:
            block_num = 64
            warp_per_block = 4

        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = self.mori_op.dispatch(
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
    ) -> None:
        """Synchronous finalize for non-DBO mode."""
        context = get_forward_context().context
        if context.is_prefill:
            block_num = 128
            warp_per_block = 16
        else:
            block_num = 64
            warp_per_block = 4

        num_token = output.shape[0]
        result = self.mori_op.combine(
            fused_expert_output,
            None,
            topk_ids,
            block_num,
            warp_per_block,
        )[0]
        output.copy_(result[:num_token])
