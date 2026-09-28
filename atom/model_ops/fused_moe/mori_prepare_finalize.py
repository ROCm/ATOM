# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from aiter import QuantType, dtypes
from aiter.jit.utils.chip_info import get_cu_num

import atom.model_ops.fused_moe.modular_kernel as mk
from atom.model_ops.fused_moe.config import FusedMoEQuantConfig
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

try:
    import mori

    MORI_AVAILABLE = True
except ImportError:
    mori = None  # type: ignore
    MORI_AVAILABLE = False


_FP8_DTYPES = (
    torch.float8_e4m3fn,
    torch.float8_e4m3fnuz,
    torch.float8_e5m2,
    torch.float8_e5m2fnuz,
)

# Read once at import: the wire format is fixed for the life of the process, and
# make_prepare_finalize runs per MoE layer (58x for DSR1). envs.__getattr__ does
# not cache, so binding here also keeps it to a single getenv.
_FP4_DISPATCH = envs.ATOM_MORI_FP4_DISPATCH


@dataclass(frozen=True)
class MoriDispatchFormat:
    dtype: torch.dtype  # what dispatch() receives -> picks the MoRI kernel
    quant_type: Any | None  # aiter QuantType for the pre-dispatch quantizer
    scale_dim: int
    scale_type_size: int

    @property
    def is_fp4(self) -> bool:
        return self.dtype == dtypes.fp4x2

    @property
    def is_fp8(self) -> bool:
        return self.dtype in _FP8_DTYPES


def resolve_mori_dispatch(
    in_dtype: torch.dtype,
    hidden_dim: int,
    quant_config: FusedMoEQuantConfig | None = None,
) -> MoriDispatchFormat:
    """Decide the MoRI wire format. Call once per layer construction."""
    if _FP4_DISPATCH:
        # fp4 blockwise is one scale per 32 elements, not per 128 as fp8 uses,
        # and get_hip_quant(per_1x32) emits e8m0 scales (1 byte), not fp32.
        return MoriDispatchFormat(
            dtype=dtypes.fp4x2,
            quant_type=QuantType.per_1x32,
            scale_dim=hidden_dim // 32,
            scale_type_size=torch.float8_e8m0fnu.itemsize,
        )
    return MoriDispatchFormat(
        dtype=in_dtype,
        quant_type=None,
        scale_dim=0,
        scale_type_size=torch.float32.itemsize,
    )


class MoriPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Prepare/Finalize using MoRI kernels.
    """

    def __init__(
        self,
        mori_ops: list[Any],
        max_tokens_per_rank: int,
        num_dispatchers: int,
        dispatch_format: MoriDispatchFormat,
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

    # Derived from the resolved format so there is no second copy to keep in
    # sync with the staging config.
    @property
    def use_fp4_dispatch(self) -> bool:
        return self.dispatch_format.is_fp4

    @property
    def use_fp8_dispatch(self) -> bool:
        return self.dispatch_format.is_fp8

    @property
    def quant_type(self):
        return self.dispatch_format.quant_type

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
        """
        mp = get_cu_num()
        context = get_forward_context().context
        if context.is_prefill:
            return min(128, mp), 16
        return min(64, mp), 4

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
        if self.use_fp4_dispatch:
            from aiter import get_hip_quant

            quant_func = get_hip_quant(self.quant_type or quant_type)
            a1, scale = quant_func(a1, quant_dtype=dtypes.fp4x2)
        elif self.use_fp8_dispatch:
            from aiter import get_hip_quant

            quant_func = get_hip_quant(quant_type)
            a1, scale = quant_func(a1, quant_dtype=dtypes.fp8)

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
    ) -> mk.ReceiverType:
        assert (
            not apply_router_weight_on_input
        ), "mori does not support apply_router_weight_on_input=True now."

        scale = None
        if self.use_fp4_dispatch:
            from aiter import get_hip_quant

            num_tokens = a1.shape[0]
            if num_tokens > 0:
                quant_func = get_hip_quant(self.quant_type or QuantType.per_1x32)
                a1, scale = quant_func(a1, quant_dtype=dtypes.fp4x2)
            else:
                hidden_size = a1.shape[1] if a1.dim() > 1 else 0
                a1 = torch.empty(a1.shape, dtype=dtypes.fp4x2, device=a1.device)
                # per_1x32 emits e8m0 scales, one byte each -- must match the
                # scale_type_size handed to MoRI in moe.py.
                scale = torch.empty(
                    (0, hidden_size // 32),
                    dtype=torch.float8_e8m0fnu,
                    device=a1.device,
                )
        elif self.use_fp8_dispatch:
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
