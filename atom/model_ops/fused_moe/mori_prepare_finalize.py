# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from typing import Any

import torch
from aiter import QuantType
from aiter.jit.utils.chip_info import get_cu_num

import atom.model_ops.fused_moe.modular_kernel as mk
from atom.model_ops.fused_moe.config import FusedMoEQuantConfig
from atom.model_ops.fused_moe.routed_rows import RoutedRowsWire
from atom.utils.forward_context import get_forward_context

try:
    import mori

    MORI_AVAILABLE = True
except ImportError:
    mori = None  # type: ignore
    MORI_AVAILABLE = False


# Decode launch geometry switches above this many tokens on the rank; see
# MoriPrepareAndFinalize._get_dispatch_config.
_DECODE_SMALL_TOKENS = 48


class MoriPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Prepare/Finalize using MoRI kernels.
    """

    def __init__(
        self,
        mori_ops: list[Any],
        max_tokens_per_rank: int,
        num_dispatchers: int,
        dispatch_format: RoutedRowsWire,
        *,
        low_latency: bool,
        internode: bool,
        quant_dtype: torch.dtype = None,
    ):
        if not MORI_AVAILABLE:
            raise ImportError(
                "mori is required for MoriPrepareAndFinalize but not installed. "
                "Please install mori to use this feature."
            )
        if not mori_ops:
            raise ValueError("mori_ops must contain at least one handle")
        super().__init__()
        self._mori_ops = tuple(mori_ops)
        self.num_dispatchers_ = num_dispatchers
        self.max_tokens_per_rank = max_tokens_per_rank
        self.dispatch_format = dispatch_format
        self.quant_dtype = quant_dtype
        self._use_split_send_recv = low_latency and not internode

    @property
    def quant_type(self):
        return self.dispatch_format.quant_type

    def _in_wire(
        self, a1: torch.Tensor, x_scale: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """`a1` as dispatch sends it: quantized here, or already in the wire
        (a gathered routed row) with its `x_scale`."""
        if x_scale is None:
            return self.dispatch_format.quantize(a1)
        assert a1.dtype == self.dispatch_format.dtype, (
            f"a1 arrived as {a1.dtype} with scales, the wire is "
            f"{self.dispatch_format.dtype}"
        )
        return a1, x_scale

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
        from atom.utils.tbo.ubatching import tbo_active

        return len(self._mori_ops) > 1 and tbo_active()

    def _current_tbo_mori_op(self):
        from atom.utils.tbo.ubatching import tbo_current_ubatch_id

        return self._mori_ops[tbo_current_ubatch_id()]

    def _get_dispatch_config(self, num_tokens: int | None = None) -> tuple[int, int]:
        """Return (block_num, warp_per_block) based on runtime mode.

        Default policy keys off the forward-context prefill/decode flag.
        atom-vllm has no stable prefill/decode flag at this call site and
        instead selects by a token-count threshold; it overrides this method
        via a plugin patch, so keep this body frontend-agnostic.

        block_num is capped at the device CU count: mori's IntraNode
        dispatch/combine use a hand-rolled grid-wide barrier
        (CrossDeviceBarrierIntraNodeKernel) that spins until *all* gridDim.x
        blocks have arrived, which requires every block to be co-resident. The
        combine block (1024 threads + larger dynamic smem) gets ~1 block/CU
        occupancy, so launching more blocks than CUs (e.g. 128 on the 80-CU
        MI308X) leaves the surplus blocks unscheduled -> the barrier never
        completes -> warmup deadlocks. Capping at multi_processor_count keeps
        big-CU GPUs (MI300X/MI355X, >=128 CU) at 128 with no perf loss.

        Decode by this rank's token count, measured on MI355X EP2 (V4.1,
        hidden 5120, top-6): 128x8 is the fastest up to 48 tokens, 256x16 from
        96 on (384 tokens: 185 us a dispatch+combine round against 210 at the
        64x4 this replaced).
        """
        mp = get_cu_num()
        context = get_forward_context().context
        if context.is_prefill:
            return min(128, mp), 16
        if num_tokens is not None and num_tokens <= _DECODE_SMALL_TOKENS:
            return min(128, mp), 8
        return min(256, mp), 16

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
        x_scale: torch.Tensor | None = None,
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
        a1, scale = self._in_wire(a1, x_scale)
        block_num, warp_per_block = self._get_dispatch_config(a1.shape[0])

        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = self._mori_ops[0].dispatch(
            a1,
            topk_weights,
            scale,
            topk_ids,
            block_num=block_num,
            warp_per_block=warp_per_block,
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

        block_num, warp_per_block = self._get_dispatch_config(num_token)

        result = self._mori_ops[0].combine(
            fused_expert_output,
            None,
            topk_ids,
            block_num=block_num,
            warp_per_block=warp_per_block,
        )[0]
        return result[:num_token]

    # 1. comm-stream (IntraNode / InterNodeV1 / InterNodeV1LL): dispatch() and
    #    combine() on comm_stream. `_use_split_send_recv` is False for these;
    #    mori only exposes the split send/recv API on AsyncLL.
    # 2. AsyncLL (--low-latency, intra-node): dispatch_send/recv,
    #    combine_send/recv (CU-free)
    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        x_scale: torch.Tensor | None = None,
    ) -> mk.ReceiverType:
        assert (
            not apply_router_weight_on_input
        ), "mori does not support apply_router_weight_on_input=True now."

        a1, scale = self._in_wire(a1, x_scale)
        if self._use_split_send_recv:
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
        mori_op = self._current_tbo_mori_op()

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
            tbo_switch_to_compute_sync,
            tbo_yield_and_switch_from_compute_to_comm,
        )

        block_num, warp_per_block = self._get_dispatch_config(a1.shape[0])
        mori_op = self._current_tbo_mori_op()

        tbo_yield_and_switch_from_compute_to_comm()

        (
            dispatch_a1,
            dispatch_weights,
            dispatch_scale,
            dispatch_ids,
            dispatch_recv_token_num,
        ) = mori_op.dispatch(
            a1,
            topk_weights,
            scale,
            topk_ids,
            block_num=block_num,
            warp_per_block=warp_per_block,
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
        if self._use_split_send_recv:
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
        mori_op = self._current_tbo_mori_op()

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
            tbo_switch_to_compute_sync,
            tbo_yield_and_switch_from_compute_to_comm,
        )

        block_num, warp_per_block = self._get_dispatch_config(num_token)
        mori_op = self._current_tbo_mori_op()

        # Yield to other thread FIRST, then switch to comm stream.
        tbo_yield_and_switch_from_compute_to_comm()

        result = mori_op.combine(
            fused_expert_output,
            None,
            topk_ids,
            block_num=block_num,
            warp_per_block=warp_per_block,
        )[0]

        tbo_switch_to_compute_sync()

        def receiver():
            return result[:num_token]

        return receiver
