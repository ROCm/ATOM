# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 specific routed MoE implementation.

This module intentionally does not depend on ATOM's generic ``FusedMoE``.  It
keeps the same checkpoint-facing parameter layout (``w13_weight`` and
``w2_weight``) so ATOM's existing expert weight mapping can load MiniMax-M3
checkpoints, while the forward path uses vLLM's floating-point MoE Triton
helper kernels when they are available and falls back to a local routed GEMV
implementation otherwise.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import import_module
from typing import Optional

import torch
import triton
import triton.language as tl
from aiter.dist.parallel_state import get_tp_group
from torch import nn

from atom.config import QuantizationConfig, get_current_atom_config
from atom.model_loader.weight_utils import set_weight_attrs
from atom.model_ops import module_dispatch_ops as _module_dispatch_ops  # noqa: F401
from atom.model_ops.swiglu_oai import swiglu_oai_split
from atom.model_ops.utils import atom_parameter, normalize_e4m3fn_to_e4m3fnuz
from atom.quant_spec import LayerQuantConfig, QuantType


@lru_cache(maxsize=1)
def _get_vllm_moe_helpers():
    try:
        ops = import_module("vllm._custom_ops")
        fused_moe = import_module("vllm.model_executor.layers.fused_moe.fused_moe")
        moe_utils = import_module("vllm.model_executor.layers.fused_moe.utils")
    except ModuleNotFoundError as exc:
        if exc.name is not None and exc.name.startswith("vllm"):
            return None
        raise

    moe_kernel_quantize_input = getattr(moe_utils, "moe_kernel_quantize_input", None)
    return (
        ops,
        fused_moe._prepare_expert_assignment,
        fused_moe.invoke_fused_moe_triton_kernel,
        fused_moe.try_get_optimal_moe_config,
        moe_kernel_quantize_input,
    )


@triton.jit
def _routed_w13_kernel(
    hidden_ptr,
    w13_ptr,
    topk_ids_ptr,
    gate_up_ptr,
    hidden_size: tl.constexpr,
    two_intermediate_size: tl.constexpr,
    top_k: tl.constexpr,
    stride_hidden_m: tl.constexpr,
    stride_hidden_k: tl.constexpr,
    stride_w13_e: tl.constexpr,
    stride_w13_n: tl.constexpr,
    stride_w13_k: tl.constexpr,
    stride_topk_m: tl.constexpr,
    stride_topk_k: tl.constexpr,
    stride_gate_up_m: tl.constexpr,
    stride_gate_up_n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    routed_id = tl.program_id(0)
    n_block = tl.program_id(1)
    token_id = routed_id // top_k
    topk_id = routed_id - token_id * top_k
    expert_id = tl.load(
        topk_ids_ptr + token_id * stride_topk_m + topk_id * stride_topk_k
    ).to(tl.int64)

    offs_n = n_block * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, hidden_size, BLOCK_K):
        k = k_start + offs_k
        hidden = tl.load(
            hidden_ptr + token_id * stride_hidden_m + k * stride_hidden_k,
            mask=k < hidden_size,
            other=0.0,
        ).to(tl.float32)
        weight = tl.load(
            w13_ptr
            + expert_id * stride_w13_e
            + offs_n[:, None] * stride_w13_n
            + k[None, :] * stride_w13_k,
            mask=(offs_n[:, None] < two_intermediate_size) & (k[None, :] < hidden_size),
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(weight * hidden[None, :], axis=1)

    tl.store(
        gate_up_ptr + routed_id * stride_gate_up_m + offs_n * stride_gate_up_n,
        acc.to(gate_up_ptr.dtype.element_ty),
        mask=offs_n < two_intermediate_size,
    )


@triton.jit
def _routed_w2_reduce_kernel(
    act_ptr,
    w2_ptr,
    topk_ids_ptr,
    topk_weights_ptr,
    output_ptr,
    hidden_size: tl.constexpr,
    intermediate_size: tl.constexpr,
    top_k: tl.constexpr,
    stride_act_m: tl.constexpr,
    stride_act_i: tl.constexpr,
    stride_w2_e: tl.constexpr,
    stride_w2_h: tl.constexpr,
    stride_w2_i: tl.constexpr,
    stride_topk_ids_m: tl.constexpr,
    stride_topk_ids_k: tl.constexpr,
    stride_topk_weights_m: tl.constexpr,
    stride_topk_weights_k: tl.constexpr,
    stride_output_m: tl.constexpr,
    stride_output_h: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_I: tl.constexpr,
):
    token_id = tl.program_id(0)
    h_block = tl.program_id(1)
    offs_h = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_i = tl.arange(0, BLOCK_I)
    acc = tl.zeros((BLOCK_H,), dtype=tl.float32)

    for topk_id in range(0, top_k):
        routed_id = token_id * top_k + topk_id
        expert_id = tl.load(
            topk_ids_ptr + token_id * stride_topk_ids_m + topk_id * stride_topk_ids_k
        ).to(tl.int64)
        route_weight = tl.load(
            topk_weights_ptr
            + token_id * stride_topk_weights_m
            + topk_id * stride_topk_weights_k
        ).to(tl.float32)

        for i_start in range(0, intermediate_size, BLOCK_I):
            i = i_start + offs_i
            act = tl.load(
                act_ptr + routed_id * stride_act_m + i * stride_act_i,
                mask=i < intermediate_size,
                other=0.0,
            ).to(tl.float32)
            weight = tl.load(
                w2_ptr
                + expert_id * stride_w2_e
                + offs_h[:, None] * stride_w2_h
                + i[None, :] * stride_w2_i,
                mask=(offs_h[:, None] < hidden_size) & (i[None, :] < intermediate_size),
                other=0.0,
            ).to(tl.float32)
            acc += tl.sum(weight * act[None, :], axis=1) * route_weight

    tl.store(
        output_ptr + token_id * stride_output_m + offs_h * stride_output_h,
        acc.to(output_ptr.dtype.element_ty),
        mask=offs_h < hidden_size,
    )


def make_minimax_m3_expert_params_mapping(
    num_experts: int,
) -> list[tuple[str, str, int, str]]:
    """Return loader mapping for MiniMax-M3 split expert checkpoint weights."""
    mapping: list[tuple[str, str, int, str]] = []
    for expert_id in range(num_experts):
        for shard_id, weight_names in (
            ("w1", ("w1", "gate_proj")),
            ("w2", ("w2", "down_proj")),
            ("w3", ("w3", "up_proj")),
        ):
            if shard_id in ("w1", "w3"):
                param_prefix = "experts.w13_"
                scale_param = "experts.w13_weight_scale"
            else:
                param_prefix = "experts.w2_"
                scale_param = "experts.w2_weight_scale"
            for weight_name in weight_names:
                # ATOM's generic loader renames ``weight_scale_inv`` to
                # ``weight_scale`` before expert-prefix matching.  Keep the
                # original ``.scale`` form for checkpoints that use it directly.
                for scale_name in ("scale", "weight_scale"):
                    mapping.append(
                        (
                            scale_param,
                            f"experts.{expert_id}.{weight_name}.{scale_name}",
                            expert_id,
                            shard_id,
                        )
                    )
                mapping.append(
                    (
                        param_prefix,
                        f"experts.{expert_id}.{weight_name}.",
                        expert_id,
                        shard_id,
                    )
                )
    return mapping


class MiniMaxM3Bf16Experts(nn.Module):
    """Dedicated MiniMax-M3 routed experts.

    The implementation mirrors MiniMax-M3's fixed routing contract:
    sigmoid scores, optional routing-bias correction for expert choice,
    top-k renormalization, SwiGLU-OAI expert activation, and routed-output
    scaling before shared experts are added by the caller.
    """

    def __init__(
        self,
        *,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        scoring_func: str,
        routed_scaling_factor: float,
        swiglu_alpha: float,
        swiglu_beta: float,
        swiglu_limit: float,
        quant_config: Optional[QuantizationConfig] = None,
        params_dtype: Optional[torch.dtype] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if scoring_func != "sigmoid":
            raise ValueError(
                f"MiniMax-M3 MoE only supports sigmoid routing, got {scoring_func!r}."
            )

        layer_quant_config = (
            quant_config.get_layer_quant_config(prefix, check_children=True)
            if quant_config is not None
            and hasattr(quant_config, "get_layer_quant_config")
            else None
        )
        if layer_quant_config is None:
            layer_quant_config = LayerQuantConfig.no_quant(
                params_dtype or torch.get_default_dtype()
            )
        self.layer_quant_config = layer_quant_config
        self.quant_type = layer_quant_config.quant_type
        self.is_quantized = self.quant_type != QuantType.No
        self.quant_dtype = layer_quant_config.quant_dtype
        self.need_normalize_e4m3fn_to_e4m3fnuz = (
            self.quant_dtype == torch.float8_e4m3fnuz
        )
        if self.is_quantized and self.quant_type != QuantType.per_Token:
            raise ValueError(
                "MiniMax-M3 dedicated MoE runtime fp8 supports PTPC "
                f"(QuantType.per_Token) only, got {self.quant_type} for {prefix!r}."
            )
        if self.is_quantized and self.quant_dtype not in (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
        ):
            raise ValueError(
                "MiniMax-M3 dedicated MoE only supports fp8 expert quantization, "
                f"got dtype={self.quant_dtype} for {prefix!r}."
            )

        self.num_experts = num_experts
        self.top_k = top_k
        tp_group = get_tp_group()
        self.tp_size = tp_group.world_size
        self.tp_rank = tp_group.rank_in_group
        if intermediate_size % self.tp_size != 0:
            raise ValueError(
                "MiniMax-M3 intermediate_size must be divisible by tensor "
                f"parallel size: {intermediate_size} vs {self.tp_size}."
            )
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.routed_scaling_factor = routed_scaling_factor
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.swiglu_limit = swiglu_limit
        self.layer_name = prefix
        compilation_config = get_current_atom_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        self.params_dtype = params_dtype or layer_quant_config.quant_dtype
        dtype = self.quant_dtype if self.is_quantized else self.params_dtype
        self.w13_weight = atom_parameter(
            torch.empty(
                num_experts,
                2 * self.intermediate_size_per_partition,
                hidden_size,
                dtype=dtype,
            )
        )
        self.w2_weight = atom_parameter(
            torch.empty(
                num_experts,
                hidden_size,
                self.intermediate_size_per_partition,
                dtype=dtype,
            )
        )
        weight_attrs = {"weight_loader": self.weight_loader}
        set_weight_attrs(self.w13_weight, weight_attrs)
        set_weight_attrs(self.w2_weight, weight_attrs)
        self._create_weight_scales(weight_attrs)

    def _create_weight_scales(self, weight_attrs: dict) -> None:
        if not self.is_quantized:
            self.register_parameter("w13_weight_scale", None)
            self.register_parameter("w2_weight_scale", None)
            return

        if self.quant_type == QuantType.per_Tensor:
            w13_shape = (self.num_experts, 2)
            w2_shape = (self.num_experts,)
        elif self.quant_type == QuantType.per_Token:
            w13_shape = (self.num_experts, 2 * self.intermediate_size_per_partition)
            w2_shape = (self.num_experts, self.w2_weight.shape[1])
        elif self.quant_type in (QuantType.per_1x128, QuantType.per_1x32):
            if self.quant_type == QuantType.per_1x128:
                block_n, block_k = 128, 128
            else:
                block_n, block_k = 1, 32
            w13_shape = (
                self.num_experts,
                2 * ((self.intermediate_size_per_partition + block_n - 1) // block_n),
                (self.w13_weight.shape[2] + block_k - 1) // block_k,
            )
            w2_shape = (
                self.num_experts,
                (self.w2_weight.shape[1] + block_n - 1) // block_n,
                (self.intermediate_size_per_partition + block_k - 1) // block_k,
            )
        else:
            raise ValueError(
                "Unsupported MiniMax-M3 expert fp8 quantization type: "
                f"{self.quant_type}."
            )

        self.w13_weight_scale = atom_parameter(
            torch.ones(w13_shape, dtype=torch.float32)
        )
        self.w2_weight_scale = atom_parameter(torch.ones(w2_shape, dtype=torch.float32))
        set_weight_attrs(self.w13_weight_scale, weight_attrs)
        set_weight_attrs(self.w2_weight_scale, weight_attrs)

    def _select_experts(
        self,
        router_logits: torch.Tensor,
        e_score_correction_bias: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        routing_weights = torch.sigmoid(router_logits.float())
        scores_for_choice = routing_weights
        if e_score_correction_bias is not None:
            scores_for_choice = scores_for_choice + e_score_correction_bias

        topk_ids = torch.topk(
            scores_for_choice,
            self.top_k,
            dim=-1,
            sorted=False,
        ).indices
        topk_weights = routing_weights.gather(dim=-1, index=topk_ids)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(
            1e-20
        )
        return topk_weights, topk_ids.to(torch.int32)

    def _load_w13(
        self,
        expert_data: torch.Tensor,
        loaded_weight: torch.Tensor,
        shard_id: str,
    ) -> None:
        shard_size = expert_data.shape[0] // 2
        target = (
            expert_data.narrow(0, 0, shard_size)
            if shard_id == "w1"
            else expert_data.narrow(0, shard_size, shard_size)
        )
        self._copy_tp_shard(target, loaded_weight, shard_dim=0)

    def _load_w2(self, expert_data: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        self._copy_tp_shard(expert_data, loaded_weight, shard_dim=1)

    def _copy_tp_shard(
        self,
        target: torch.Tensor,
        loaded_weight: torch.Tensor,
        shard_dim: int,
    ) -> None:
        local_size = target.shape[shard_dim]
        loaded_size = loaded_weight.shape[shard_dim]
        if loaded_size == local_size:
            shard = loaded_weight
        else:
            if loaded_size % self.tp_size != 0:
                raise ValueError(
                    "Cannot shard MiniMax-M3 expert weight with shape "
                    f"{tuple(loaded_weight.shape)} across TP size {self.tp_size}."
                )
            load_shard_size = loaded_size // self.tp_size
            shard = loaded_weight.narrow(
                shard_dim,
                load_shard_size * self.tp_rank,
                load_shard_size,
            )
            if load_shard_size != local_size:
                target = target.narrow(shard_dim, 0, load_shard_size)

        fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
        if target.dtype in fp8_dtypes and shard.dtype in fp8_dtypes:
            # Preserve checkpoint FP8 bit patterns. On MI300, ATOM's runtime
            # dtype is e4m3fnuz while MiniMax-M3 PTPC checkpoints store e4m3fn;
            # scale normalization after loading preserves the dequantized value.
            target.view(torch.uint8).copy_(
                shard.to(device=target.device).view(torch.uint8)
            )
        else:
            target.copy_(shard.to(dtype=target.dtype, device=target.device))

    def _copy_scale_shard(
        self,
        target: torch.Tensor,
        loaded_weight: torch.Tensor,
        shard_dim: int | None,
    ) -> None:
        while loaded_weight.dim() > target.dim() and loaded_weight.shape[-1] == 1:
            loaded_weight = loaded_weight.squeeze(-1)
        if target.dim() == 0 or loaded_weight.dim() == 0 or shard_dim is None:
            if loaded_weight.numel() == target.numel():
                loaded_weight = loaded_weight.reshape(target.shape)
            target.copy_(loaded_weight.to(dtype=target.dtype, device=target.device))
            return
        self._copy_tp_shard(target, loaded_weight, shard_dim=shard_dim)

    def _load_w13_scale(
        self,
        expert_data: torch.Tensor,
        loaded_weight: torch.Tensor,
        shard_id: str,
    ) -> None:
        if shard_id == "w2":
            raise ValueError("w2 scale cannot be loaded into w13_weight_scale.")

        if self.quant_type == QuantType.per_Tensor:
            target = expert_data[0 if shard_id == "w1" else 1]
            self._copy_scale_shard(target, loaded_weight, shard_dim=None)
            return

        shard_size = expert_data.shape[0] // 2
        target = (
            expert_data.narrow(0, 0, shard_size)
            if shard_id == "w1"
            else expert_data.narrow(0, shard_size, shard_size)
        )
        self._copy_scale_shard(target, loaded_weight, shard_dim=0)

    def _load_w2_scale(
        self,
        expert_data: torch.Tensor,
        loaded_weight: torch.Tensor,
    ) -> None:
        shard_dim = (
            None
            if self.quant_type in (QuantType.per_Tensor, QuantType.per_Token)
            else 1
        )
        self._copy_scale_shard(expert_data, loaded_weight, shard_dim=shard_dim)

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str = "",
        shard_id: str = "",
        expert_id: int = 0,
    ) -> None:
        del weight_name
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(
                "MiniMax-M3 expert shard_id must be w1/w2/w3, " f"got {shard_id!r}."
            )

        if (
            loaded_weight.dim() == param.data.dim()
            and loaded_weight.shape[0] == param.data.shape[0]
        ):
            for local_expert_id in range(
                min(param.data.shape[0], loaded_weight.shape[0])
            ):
                if param is self.w13_weight:
                    self._load_w13(
                        param.data[local_expert_id],
                        loaded_weight[local_expert_id],
                        shard_id,
                    )
                elif param is self.w2_weight:
                    self._load_w2(
                        param.data[local_expert_id],
                        loaded_weight[local_expert_id],
                    )
                elif param is self.w13_weight_scale:
                    self._load_w13_scale(
                        param.data[local_expert_id],
                        loaded_weight[local_expert_id],
                        shard_id,
                    )
                elif param is self.w2_weight_scale:
                    self._load_w2_scale(
                        param.data[local_expert_id],
                        loaded_weight[local_expert_id],
                    )
                else:
                    raise ValueError("Unknown MiniMax-M3 expert parameter.")
            return

        if expert_id < 0 or expert_id >= self.num_experts:
            raise ValueError(
                f"MiniMax-M3 expert_id {expert_id} is outside [0, {self.num_experts})."
            )

        expert_data = param.data[expert_id]
        if param is self.w13_weight:
            self._load_w13(expert_data, loaded_weight, shard_id)
        elif param is self.w2_weight:
            self._load_w2(expert_data, loaded_weight)
        elif param is self.w13_weight_scale:
            self._load_w13_scale(expert_data, loaded_weight, shard_id)
        elif param is self.w2_weight_scale:
            self._load_w2_scale(expert_data, loaded_weight)
        else:
            raise ValueError("Unknown MiniMax-M3 expert parameter.")

    def process_weights_after_loading(self) -> None:
        self.w13_weight.data = self.w13_weight.data.contiguous()
        self.w2_weight.data = self.w2_weight.data.contiguous()
        if self.is_quantized:
            self.w13_weight_scale.data = self.w13_weight_scale.data.contiguous()
            self.w2_weight_scale.data = self.w2_weight_scale.data.contiguous()
            if self.need_normalize_e4m3fn_to_e4m3fnuz:
                (
                    self.w13_weight.data,
                    self.w13_weight_scale.data,
                    _,
                ) = normalize_e4m3fn_to_e4m3fnuz(
                    self.w13_weight.data,
                    self.w13_weight_scale.data,
                )
                (
                    self.w2_weight.data,
                    self.w2_weight_scale.data,
                    _,
                ) = normalize_e4m3fn_to_e4m3fnuz(
                    self.w2_weight.data,
                    self.w2_weight_scale.data,
                )

    def _quantize_activation_ptpc(
        self,
        moe_kernel_quantize_input,
        activation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        quantized, scale = moe_kernel_quantize_input(
            A=activation,
            A_scale=None,
            quant_dtype=self.quant_dtype,
            per_act_token_quant=True,
            block_shape=None,
            ocp_mx_scheme=None,
        )
        if scale.dim() == 1:
            scale = scale.unsqueeze(-1)
        return quantized.contiguous(), scale.contiguous()

    def _forward_routed_gemv(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        if self.is_quantized:
            raise RuntimeError(
                "MiniMax-M3 fp8 PTPC experts require vLLM's graph-safe MoE "
                "Triton helper; the local routed GEMV fallback is floating-only."
            )
        n_tokens = hidden_states.shape[0]
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        if topk_ids.dtype != torch.int32:
            topk_ids = topk_ids.to(torch.int32)
        topk_ids = topk_ids.contiguous()
        topk_weights = topk_weights.contiguous()

        top_k = topk_ids.shape[1]
        hidden_size = self.w2_weight.shape[1]
        two_intermediate_size = self.w13_weight.shape[1]
        intermediate_size = self.intermediate_size_per_partition

        gate_up = torch.empty(
            n_tokens * top_k,
            two_intermediate_size,
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        block_n = 64
        block_k = 64
        _routed_w13_kernel[
            (n_tokens * top_k, triton.cdiv(two_intermediate_size, block_n))
        ](
            hidden_states,
            self.w13_weight,
            topk_ids,
            gate_up,
            hidden_size,
            two_intermediate_size,
            top_k,
            hidden_states.stride(0),
            hidden_states.stride(1),
            self.w13_weight.stride(0),
            self.w13_weight.stride(1),
            self.w13_weight.stride(2),
            topk_ids.stride(0),
            topk_ids.stride(1),
            gate_up.stride(0),
            gate_up.stride(1),
            BLOCK_N=block_n,
            BLOCK_K=block_k,
        )

        activated = swiglu_oai_split(
            gate_up,
            alpha=self.swiglu_alpha,
            beta=self.swiglu_beta,
            limit=self.swiglu_limit,
            out_dtype=hidden_states.dtype,
        )
        output = torch.empty_like(hidden_states)
        block_h = 64
        block_i = 64
        _routed_w2_reduce_kernel[(n_tokens, triton.cdiv(hidden_size, block_h))](
            activated,
            self.w2_weight,
            topk_ids,
            topk_weights,
            output,
            hidden_size,
            intermediate_size,
            top_k,
            activated.stride(0),
            activated.stride(1),
            self.w2_weight.stride(0),
            self.w2_weight.stride(1),
            self.w2_weight.stride(2),
            topk_ids.stride(0),
            topk_ids.stride(1),
            topk_weights.stride(0),
            topk_weights.stride(1),
            output.stride(0),
            output.stride(1),
            BLOCK_H=block_h,
            BLOCK_I=block_i,
        )
        return output

    def _forward_graph_safe(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Prefer vLLM's graph-safe fused MoE path when available; it is faster
        # than the local routed GEMV fallback.
        vllm_moe_helpers = _get_vllm_moe_helpers()
        if vllm_moe_helpers is None:
            return self._forward_routed_gemv(hidden_states, topk_weights, topk_ids)
        (
            ops,
            _prepare_expert_assignment,
            invoke_fused_moe_triton_kernel,
            try_get_optimal_moe_config,
            moe_kernel_quantize_input,
        ) = vllm_moe_helpers
        if self.is_quantized and moe_kernel_quantize_input is None:
            raise RuntimeError(
                "MiniMax-M3 fp8 PTPC experts require "
                "vllm.model_executor.layers.fused_moe.utils."
                "moe_kernel_quantize_input."
            )
        n_tokens = hidden_states.shape[0]
        if not hidden_states.is_contiguous():
            hidden_states = hidden_states.contiguous()
        if topk_ids.dtype != torch.int32:
            topk_ids = topk_ids.to(torch.int32)
        topk_ids = topk_ids.contiguous()
        topk_weights = topk_weights.contiguous()

        two_intermediate_size = self.w13_weight.shape[1]
        hidden_size = self.w2_weight.shape[1]
        top_k = topk_ids.shape[1]

        config = try_get_optimal_moe_config(
            self.w13_weight.size(),
            self.w2_weight.size(),
            top_k,
            None,
            n_tokens,
            block_shape=None,
        )

        cache13 = torch.empty(
            n_tokens * top_k * max(two_intermediate_size, hidden_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        gate_up = cache13[: n_tokens * top_k * two_intermediate_size].view(
            n_tokens,
            top_k,
            two_intermediate_size,
        )
        w2_accum = cache13[: n_tokens * top_k * hidden_size].view(
            n_tokens,
            top_k,
            hidden_size,
        )
        if hidden_states.dtype == torch.bfloat16:
            compute_type = tl.bfloat16
        elif hidden_states.dtype == torch.float16:
            compute_type = tl.float16
        elif hidden_states.dtype == torch.float32:
            compute_type = tl.float32
        else:
            raise RuntimeError(
                "MiniMax-M3 fused MoE only supports bf16/fp16/fp32 inputs, "
                f"got {hidden_states.dtype}."
            )

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            _prepare_expert_assignment(
                topk_ids,
                config,
                n_tokens,
                top_k,
                self.num_experts,
                None,
                ignore_invalid_experts=True,
            )
        )

        if self.is_quantized:
            hidden_states_for_kernel, a1_scale = self._quantize_activation_ptpc(
                moe_kernel_quantize_input,
                hidden_states,
            )
            w13_scale = self.w13_weight_scale
        else:
            hidden_states_for_kernel = hidden_states
            a1_scale = None
            w13_scale = None

        invoke_fused_moe_triton_kernel(
            hidden_states_for_kernel,
            self.w13_weight,
            gate_up,
            a1_scale,
            w13_scale,
            None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,
            top_k,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=self.is_quantized,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=self.is_quantized,
            block_shape=None,
            B_bias=None,
        )

        activated = swiglu_oai_split(
            gate_up.view(-1, two_intermediate_size),
            alpha=self.swiglu_alpha,
            beta=self.swiglu_beta,
            limit=self.swiglu_limit,
            out_dtype=hidden_states.dtype,
        )

        if self.is_quantized:
            activated_for_kernel, a2_scale = self._quantize_activation_ptpc(
                moe_kernel_quantize_input,
                activated,
            )
            w2_scale = self.w2_weight_scale
        else:
            activated_for_kernel = activated
            a2_scale = None
            w2_scale = None

        invoke_fused_moe_triton_kernel(
            activated_for_kernel,
            self.w2_weight,
            w2_accum,
            a2_scale,
            w2_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            True,
            1,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=self.is_quantized,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=self.is_quantized,
            block_shape=None,
            B_bias=None,
        )

        output = torch.empty_like(hidden_states)
        ops.moe_sum(w2_accum.view(*w2_accum.size()), output)
        return output

    def forward_impl(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        e_score_correction_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        valid_weight_dtypes = (
            (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
            if self.is_quantized
            else (torch.bfloat16, torch.float16, torch.float32)
        )
        if self.w13_weight.dtype not in valid_weight_dtypes:
            raise RuntimeError(
                "MiniMax-M3 dedicated MoE received unsupported expert "
                f"weights, got {self.w13_weight.dtype}."
            )

        topk_weights, topk_ids = self._select_experts(
            router_logits,
            e_score_correction_bias,
        )
        output = self._forward_graph_safe(hidden_states, topk_weights, topk_ids)
        if self.routed_scaling_factor != 1.0:
            output = output * self.routed_scaling_factor
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        e_score_correction_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if e_score_correction_bias is None:
            e_score_correction_bias = hidden_states.new_empty(0)
        return torch.ops.aiter.minimax_m3_bf16_experts_forward(
            hidden_states,
            router_logits,
            e_score_correction_bias,
            self.layer_name,
        )
