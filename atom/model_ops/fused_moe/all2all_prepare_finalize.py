# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
ATOM-side prepare/finalize for the aiter All2AllManager EP backend.

All2AllPrepareAndFinalize wraps an op handle returned by an aiter
All2AllManager (e.g. FlyDSLAll2AllManager) and implements the same
FusedMoEPrepareAndFinalize interface as the mori backend.  The concrete
provider and topology (intranode, internode, …) are aiter-internal details;
ATOM only calls the standard dispatch()/combine() surface on the handle.
"""

import logging
from functools import lru_cache
from typing import Any

import torch

import atom.model_ops.fused_moe.modular_kernel as mk
from atom.model_ops.fused_moe.config import FusedMoEQuantConfig
from aiter import QuantType, dtypes
from aiter import get_hip_quant

logger = logging.getLogger("atom")


_NUM_TBO_UBATCHES = 2


@lru_cache(maxsize=8)
def _make_comm_op(all2all_manager: Any, config_key: tuple, instance_id: int):
    """Return a cached comm op for the given (manager, config, instance) key.

    Mirrors mori's init_mori_op @lru_cache pattern: the same key across all
    MoE layers returns the same op, avoiding per-layer shmem re-allocation.
    Different instance_id values produce independent ops with separate buffers.
    """
    return all2all_manager.create_handle(dict(config_key))


class All2AllPrepareAndFinalize(mk.FusedMoEPrepareAndFinalize):
    """
    Prepare/Finalize backed by an aiter All2AllManager comm op.

    Holds one primary comm op (_comm_op) for the synchronous path and an
    optional list of independent comm ops (_comm_ops) for the overlapped
    async path.  The provider (FlyDSL or future alternatives) is transparent:
    only the standard dispatch()/combine() surface is used.

    Dispatch quantization (fp8/fp4 per_1x32+e8m0) is applied before handing
    the tensor to the comm op; the op buffer is sized for the bf16 input so
    the same op instance handles both quantized and unquantized calls without
    reallocation.
    """

    def __init__(
        self,
        comm_op: Any,
        max_tokens_per_rank: int,
        num_dispatchers: int,
        dispatch_quant_dtype: torch.dtype | None = None,
        is_async: bool = False,
        comm_ops: list | None = None,
        low_latency: bool = False,
    ):
        super().__init__()
        self._comm_op = comm_op
        self._comm_ops = comm_ops
        self.num_dispatchers_ = num_dispatchers
        self.max_tokens_per_rank = max_tokens_per_rank
        self.dispatch_quant_dtype = dispatch_quant_dtype
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

    def _quantize_dispatch_input(
        self, a1: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Quantize a1 for dispatch (per_1x32 + e8m0 scale). Handles empty ranks."""
        dispatch_quant_dtype = self.dispatch_quant_dtype
        if dispatch_quant_dtype is None:
            return a1, None


        quant_func = get_hip_quant(QuantType.per_1x32)

        if a1.shape[0] > 0:
            if dispatch_quant_dtype == dtypes.fp8:
                return quant_func(a1, quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0)
            if dispatch_quant_dtype == dtypes.fp4x2:
                return quant_func(a1, quant_dtype=dtypes.fp4x2, scale_type=dtypes.fp8_e8m0)
            raise ValueError(f"Unsupported dispatch quant dtype: {dispatch_quant_dtype}")

        # Empty-token rank: return correctly-typed empty tensors so the comm
        # op receives consistent dtypes without running the quant kernel.
        hidden_size = a1.shape[1] if a1.dim() > 1 else 0
        if dispatch_quant_dtype == dtypes.fp4x2:
            a_out = torch.empty((0, hidden_size // 2), dtype=dtypes.fp4x2, device=a1.device)
        else:
            a_out = torch.empty(a1.shape, dtype=dispatch_quant_dtype, device=a1.device)
        scale_out = torch.empty(
            (0, hidden_size // 32), dtype=dtypes.fp8_e8m0, device=a1.device
        )
        return a_out, scale_out

    # ---- Synchronous path ----

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
        ), "All2AllPrepareAndFinalize does not support apply_router_weight_on_input=True."

        a1, scale = self._quantize_dispatch_input(a1)
        out_tok, out_wts, out_scales, out_idx, total_recv = self._comm_op.dispatch(
            a1, topk_weights, scale, topk_ids
        )
        meta = mk.ExpertTokensMetadata(
            expert_num_tokens=total_recv, expert_num_tokens_cpu=None
        )
        return (out_tok, out_scales, meta, out_idx, out_wts)

    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        num_token = topk_ids.shape[0]
        out_tok, out_wts = self._comm_op.combine(
            fused_expert_output, None, topk_ids, cur_tok=num_token
        )
        return out_tok[:num_token]

    # ---- Async (overlapped comm-stream) path ----

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
    ) -> mk.ReceiverType:
        from atom.utils.tbo.ubatching import (
            tbo_current_ubatch_id,
            tbo_yield_and_switch_from_compute_to_comm,
            tbo_switch_to_compute_sync,
        )

        a1, scale = self._quantize_dispatch_input(a1)
        ubatch_id = tbo_current_ubatch_id()
        comm_op = self._comm_ops[ubatch_id]

        tbo_yield_and_switch_from_compute_to_comm()
        out_tok, out_wts, out_scales, out_idx, total_recv = comm_op.dispatch(
            a1, topk_weights, scale, topk_ids
        )
        tbo_switch_to_compute_sync()

        def receiver() -> mk.PrepareResultType:
            meta = mk.ExpertTokensMetadata(
                expert_num_tokens=total_recv, expert_num_tokens_cpu=None
            )
            return (out_tok, out_scales, meta, out_idx, out_wts)

        return receiver

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ):
        from atom.utils.tbo.ubatching import (
            tbo_current_ubatch_id,
            tbo_yield_and_switch_from_compute_to_comm,
            tbo_switch_to_compute_sync,
        )

        num_token = topk_ids.shape[0]
        ubatch_id = tbo_current_ubatch_id()
        comm_op = self._comm_ops[ubatch_id]

        tbo_yield_and_switch_from_compute_to_comm()
        out_tok, out_wts = comm_op.combine(
            fused_expert_output, None, topk_ids, cur_tok=num_token
        )
        tbo_switch_to_compute_sync()

        def receiver():
            return out_tok[:num_token]

        return receiver
