# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from abc import abstractmethod
from functools import lru_cache
from typing import Callable, List, Optional, Tuple

from dataclasses import dataclass

from atom.model_ops.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEQuantConfig,
    fp8_w8a8_moe_quant_config,
    mxfp4_w4a16_moe_quant_config,
)
from atom.model_ops.fused_moe.modular_kernel import (
    FusedMoEModularKernel,
    FusedMoEPrepareAndFinalize,
)
from atom.model_ops.fused_moe.mori_prepare_finalize import MoriPrepareAndFinalize
from atom.utils.forward_context import ForwardContext, get_forward_context
import torch
from aiter import ActivationType, QuantType, dtypes, get_hip_quant
from aiter.dist.parallel_state import get_tp_group, get_dp_group
from aiter.fused_moe import fused_moe
from aiter.utility import fp4_utils
from aiter.ops.shuffle import (
    shuffle_scale_a16w4,
    shuffle_weight_a16w4,
)
from torch import nn
import torch.nn.functional as F
from transformers import PretrainedConfig

from atom.config import Config, QuantizationConfig, get_current_atom_config
from atom.model_loader.weight_utils import set_weight_attrs
from atom.model_ops.base_config import QuantizeMethodBase
from atom.model_ops.topK import (
    init_aiter_topK_meta_data,
    is_rocm_aiter_fuse_routed_scaling_factor,
    is_rocm_aiter_fusion_shared_expert_enabled,
)
from atom.model_ops.topK import rocm_aiter_grouped_topk as grouped_topk
from atom.model_ops.topK import rocm_aiter_topk_softmax as fused_topk
from atom.model_ops.utils import (
    normalize_e4m3fn_to_e4m3fnuz,
    per_tensor_dequantize,
    shuffle_weights,
)
from atom.utils.custom_register import direct_register_custom_op
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.jit.utils.chip_info import get_gfx
from atom.utils import envs

from atom.utils import envs, mark_spliting_op
from atom.model_ops.fused_moe.config import _has_module


@dataclass
class FusedMoEParallelConfig:
    tp_size: int
    dp_size: int
    ep_size: int
    tp_rank: int
    dp_rank: int
    ep_rank: int

    use_ep: bool  # whether to use EP or not
    local_ep_size: int

    @property
    def use_all2all_kernels(self):
        # Only use mori all2all kernels when expert parallel is enabled
        return self.dp_size > 1 and self.use_ep and _has_module("mori")

    @property
    def use_mori_kernels(self):
        return True

    @staticmethod
    def make(
        tp_size_: int, dp_size_: int, parallel_config: Config
    ) -> "FusedMoEParallelConfig":
        def flatten_tp_across_dp(dp_rank: int):
            tp_rank = 0 if tp_size_ == 1 else get_tp_group().rank_in_group
            # There are actually dp_size_ * tp_size_ devices. Update tp_size
            # and tp_rank so we shard across all devices.
            tp_size = dp_size_ * tp_size_
            tp_rank = dp_rank * tp_size_ + tp_rank
            return tp_size, tp_rank

        use_ep = dp_size_ * tp_size_ > 1 and parallel_config.enable_expert_parallel

        dp_size = dp_size_
        dp_rank = get_dp_group().rank_in_group if dp_size > 1 else 0
        tp_size, tp_rank = flatten_tp_across_dp(dp_rank)
        atom_config = get_current_atom_config()

        if not use_ep:
            return FusedMoEParallelConfig(
                tp_size=tp_size,
                tp_rank=tp_rank,
                dp_size=dp_size,
                dp_rank=dp_rank,
                ep_size=1,
                ep_rank=0,
                use_ep=False,
                local_ep_size=1,
            )
        # DP + EP / TP + EP / DP + TP + EP
        assert use_ep
        # In EP, each device owns a set of experts fully. There is no tensor
        # parallel update tp_size, tp_rank, ep_size and ep_rank to reflect that.
        ep_size = tp_size
        ep_rank = tp_rank
        return FusedMoEParallelConfig(
            tp_size=1,
            tp_rank=0,
            dp_size=dp_size,
            dp_rank=dp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            use_ep=True,
            local_ep_size=atom_config.parallel_config.data_parallel_size_local
            * tp_size_,
        )


def naive_multicast_fake(
    x: torch.Tensor, cu_tokens_across_dp_cpu: torch.Tensor
) -> torch.Tensor:
    assert len(x.shape) == 2
    # print(f"cu_tokens_across_dp_cpu: {cu_tokens_across_dp_cpu}")
    buffer = torch.empty(
        (cu_tokens_across_dp_cpu[-1], x.size(1)), device=x.device, dtype=x.dtype
    )
    return buffer


@torch_compile_guard()
def naive_multicast(
    x: torch.Tensor, cu_tokens_across_dp_cpu: torch.Tensor
) -> torch.Tensor:
    dp_rank = get_dp_group().rank_in_group
    assert len(x.shape) == 2
    # print(f"cu_tokens_across_dp_cpu: {cu_tokens_across_dp_cpu}")
    buffer = torch.empty(
        (cu_tokens_across_dp_cpu[-1], x.size(1)), device=x.device, dtype=x.dtype
    )

    start = 0 if dp_rank == 0 else cu_tokens_across_dp_cpu[dp_rank - 1]
    end = cu_tokens_across_dp_cpu[dp_rank]
    buffer[start:end, :].copy_(x)
    for idx in range(get_dp_group().world_size):
        start = 0 if idx == 0 else cu_tokens_across_dp_cpu[idx - 1]
        end = cu_tokens_across_dp_cpu[idx]
        get_dp_group().broadcast(buffer[start:end, :], idx)
    return buffer


def all_gather_with_padding(x: torch.Tensor):
    max_batch_size = get_forward_context().context.graph_bs
    dim = 0
    original_batch_size = x.shape[dim]
    padded_x = x
    if original_batch_size < max_batch_size:
        padding_size = max_batch_size - original_batch_size

        padding_shape = list(x.shape)
        padding_shape[dim] = padding_size

        padding = torch.empty(padding_shape, dtype=x.dtype, device=x.device)
        padding.zero_()
        padded_x = torch.cat([x, padding], dim=dim)

    gathered_hidden_states = get_dp_group().all_gather(padded_x, dim=dim)
    return gathered_hidden_states, original_batch_size


def reduce_scatter_with_unpadding(
    x: torch.Tensor, original_batch_size: int
) -> torch.Tensor:
    dim = 0
    dp_group = get_dp_group()

    # scattered_output = dp_group.reduce_scatter(x, dim=dim)
    scattered_output = dp_group.reduce_scatter_tensor(x)

    if scattered_output.shape[dim] > original_batch_size:
        slices = [slice(None)] * scattered_output.ndim
        slices[dim] = slice(0, original_batch_size)
        scattered_output = scattered_output[slices]

    return scattered_output


@torch_compile_guard()
def get_max_tokens_across_dispatchers(input: torch.Tensor) -> int:
    return input.item()


class FusedMoEMethodBase(QuantizeMethodBase):
    def __init__(self, moe: FusedMoEConfig):
        super().__init__()
        self.moe = moe
        self.moe_quant_config: FusedMoEQuantConfig | None = None
        self.fused_experts: FusedMoEModularKernel | None = None
        self.topk_indices_dtype = None

    @abstractmethod
    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        raise NotImplementedError

    @abstractmethod
    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        raise NotImplementedError

    @staticmethod
    def _maybe_make_prepare_finalize(
        moe: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig | None,
    ) -> FusedMoEPrepareAndFinalize | None:
        from aiter.dist.parallel_state import get_ep_group

        all2all_manager = get_ep_group().device_communicator.all2all_manager
        assert all2all_manager is not None

        prepare_finalize: FusedMoEPrepareAndFinalize | None = None

        # TODO: could allow this now
        # assert not moe.use_flashinfer_cutlass_kernels, "Must be created in modelopt.py"
        if moe.use_mori_kernels:
            assert quant_config is not None
            # For PTPC (per token per channel) quant, the scale dim for each token is 1
            # For 1x128 quant, the scale dim for each token is hidden_dim // 128
            scale_dim = 1 if quant_config.is_per_act_token else moe.hidden_dim // 128

            # Check if quant_dtype is an FP8 type
            from aiter import QuantType

            fp8_dtypes = (
                torch.float8_e4m3fn,
                torch.float8_e4m3fnuz,
                torch.float8_e5m2,
                torch.float8_e5m2fnuz,
            )
            is_fp8 = quant_config.quant_dtype in fp8_dtypes
            # For FP8: enable FP8 dispatch in Mori (quantize before communication)
            # Note: per_Tensor quant doesn't support num_local_tokens, so we use per_Token
            use_fp8_dispatch = is_fp8
            quant_type = None
            if use_fp8_dispatch:
                if quant_config.is_block_quantized:
                    quant_type = QuantType.per_1x128
                elif quant_config.is_per_act_token:
                    quant_type = QuantType.per_Token

            # For FP8: use FP8 dtype for communication
            # For FP4/no quant: use bfloat16
            mori_dtype = (
                quant_config.quant_dtype
                if is_fp8 and quant_type is not None
                else torch.bfloat16
            )
            # mori_dtype = torch.bfloat16

            all_to_all_args = dict(
                rank=all2all_manager.rank,
                num_ep_ranks=all2all_manager.world_size,
                # quant_dtype=mori_dtype,
                # We now use bfloat16 for mori
                # TODO: To support quant
                quant_dtype=moe.in_dtype,
                token_hidden_size=moe.hidden_dim,
                scale_dim=scale_dim,
                scale_type_size=torch.float32.itemsize,
                max_num_tokens_per_dp_rank=16384,
                # input_dtype=moe.in_dtype,
                input_dtype=moe.in_dtype,
                num_local_experts=moe.num_experts // all2all_manager.world_size,
                num_experts_per_token=moe.experts_per_token,
                gpu_per_node=moe.moe_parallel_config.local_ep_size,
            )
            handle = all2all_manager.get_handle(all_to_all_args)

            # We not use quant for mori now
            use_fp8_dispatch = False
            quant_type = None

            prepare_finalize = MoriPrepareAndFinalize(
                handle,
                max_tokens_per_rank=moe.max_num_tokens,
                num_dispatchers=all2all_manager.world_size,
                use_fp8_dispatch=use_fp8_dispatch,
                quant_type=quant_type,
            )

        return prepare_finalize

    def maybe_make_prepare_finalize(self) -> FusedMoEPrepareAndFinalize | None:
        # if True:
        if self.moe.moe_parallel_config.use_all2all_kernels:
            return FusedMoEMethodBase._maybe_make_prepare_finalize(
                self.moe, self.moe_quant_config
            )
        else:
            return None

    # Note: init_prepare_finalize should only be called by
    # prepare_communication_buffer_for_model.
    def init_prepare_finalize(self, layer: torch.nn.Module):
        # print("init_prepare_finalize")
        assert self.moe is not None

        # We must get the quant config here so that the layer is
        # completely initialized, i.e. all weights loaded and post
        # processed.
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)

        prepare_finalize = self.maybe_make_prepare_finalize()

        if prepare_finalize is not None:
            # logger.debug(
            #     "%s for %s(%s)", prepare_finalize.__class__.__name__, self, id(self)
            # )
            assert self.topk_indices_dtype is None
            assert (
                self.fused_experts is None
            ), f"Attempt to override experts for {id(self)}!"
            self.topk_indices_dtype = prepare_finalize.topk_indices_dtype()
            # experts = self.select_gemm_impl(prepare_finalize, layer)
            self.fused_experts = FusedMoEModularKernel(
                prepare_finalize,
                # experts,
                # layer.shared_experts,
                quant_config=self.moe_quant_config,
            )

    @property
    def using_modular_kernel(self) -> bool:
        return self.fused_experts is not None


class UnquantizedFusedMoEMethod(FusedMoEMethodBase):
    """MoE method without quantization."""

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def _maybe_pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        # Pad the weight tensor. This is an optimization on ROCm platform, which
        # can benefit from tensors located far enough from one another in memory
        # if (envs.VLLM_ROCM_MOE_PADDING and current_platform.is_rocm()
        #         and weight.stride(-1) == 1
        #         and (weight.stride(-2) * weight.element_size()) % 512 == 0):
        #     num_pad = 256 // weight.element_size()
        #     weight = F.pad(weight, (0, num_pad), "constant", 0)[..., :-num_pad]
        #     torch.cuda.empty_cache()
        return weight

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        layer.w13_weight = torch.nn.Parameter(
            self._maybe_pad_weight(layer.w13_weight.data), requires_grad=False
        )
        layer.w2_weight = torch.nn.Parameter(
            self._maybe_pad_weight(layer.w2_weight.data), requires_grad=False
        )
        # reshaping weights is required for aiter moe kernel.
        shuffled_w13, shuffled_w2 = shuffle_weights(layer.w13_weight, layer.w2_weight)

        layer.w13_weight = torch.nn.Parameter(shuffled_w13, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(shuffled_w2, requires_grad=False)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return FUSED_MOE_UNQUANTIZED_CONFIG

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: ActivationType = ActivationType.Silu,
    ) -> torch.Tensor:
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
        )
        if self.fused_experts:
            return self.fused_experts(
                hidden_states=x,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                inplace=False,
                activation=activation,
                quant_type=QuantType.No,
                global_num_experts=global_num_experts,
                expert_map=expert_map,
            )
        return fused_moe(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weight=topk_weights,
            topk_ids=topk_ids,
            expert_mask=expert_map,
            activation=activation,
        )


def rocm_aiter_fused_moe_impl(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: Optional[torch.Tensor] = None,
    activation: int = ActivationType.Silu.value,
    quant_type: int = QuantType.No.value,
    doweight_stage1: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    from aiter import ActivationType, QuantType

    activation_ = ActivationType(activation)
    quant_type_ = QuantType(quant_type)

    return fused_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        expert_mask,
        activation_,
        quant_type_,
        doweight_stage1,
        w1_scale,
        w2_scale,
        a1_scale,
        a2_scale,
    )


def rocm_aiter_fused_moe_fake(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: Optional[torch.Tensor] = None,
    activation: int = ActivationType.Silu.value,
    quant_type: int = QuantType.No.value,
    doweight_stage1: bool = False,
    w1_scale: Optional[torch.Tensor] = None,
    w2_scale: Optional[torch.Tensor] = None,
    a1_scale: Optional[torch.Tensor] = None,
    a2_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="rocm_aiter_fused_moe",
    op_func=rocm_aiter_fused_moe_impl,
    mutates_args=[],
    fake_impl=rocm_aiter_fused_moe_fake,
)


class Mxfp4MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: QuantizationConfig, moe: FusedMoEConfig):
        super().__init__(moe)
        self.quant_config = quant_config
        self.quant_type = self.quant_config["quant_type"]
        self.quant_dtype = self.quant_config["quant_dtype"]
        self.quant_method = self.quant_config["quant_method"]
        self.block_quant = (
            self.quant_type == QuantType.per_1x128
            or self.quant_type == QuantType.per_1x32
        )
        self.use_triton = get_gfx().startswith("gfx94")
        if self.use_triton:
            from atom.model_ops.fused_moe_triton import has_triton_kernels

            assert has_triton_kernels(), "triton_kernels is not installed"

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.num_experts = num_experts
        weight_dtype = params_dtype
        scale_dtype = torch.uint8

        mxfp4_block = 32
        pad_align = 256

        intermediate_size_per_partition_after_pad = (
            (intermediate_size_per_partition + pad_align - 1) // pad_align * pad_align
        )
        hidden_size = (hidden_size + pad_align - 1) // pad_align * pad_align
        self.intermediate_size = intermediate_size_per_partition_after_pad
        self.hidden_size = hidden_size
        self.hidden_pad = self.hidden_size - layer.hidden_size
        # Update moe.hidden_dim to match the padded hidden size for Mori kernels
        self.moe.hidden_dim = hidden_size
        self.intermediate_pad = (
            self.intermediate_size - layer.intermediate_size_per_partition
        )
        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        if layer.has_bias:
            w13_bias = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    2 * intermediate_size_per_partition_after_pad,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
        else:
            layer.register_parameter("w13_bias", None)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // 2,
                dtype=weight_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = torch.nn.Parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // mxfp4_block,
                dtype=scale_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        if layer.has_bias:
            w2_bias = torch.nn.Parameter(
                torch.empty(
                    num_experts,
                    hidden_size,
                    dtype=torch.bfloat16,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)
        else:
            layer.register_parameter("w2_bias", None)

    def process_weights_after_loading(self, layer):
        if layer.w13_bias is not None:
            layer.w13_bias.data = layer.w13_bias.data.to(torch.float32)
        if layer.w2_bias is not None:
            layer.w2_bias.data = layer.w2_bias.data.to(torch.float32)

        if self.use_triton:
            from atom.model_ops.fused_moe_triton import _swizzle_mxfp4
            from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig

            w13_weight, w13_flex, w13_scale = _swizzle_mxfp4(
                layer.w13_weight.view(torch.uint8),
                layer.w13_weight_scale,
            )
            w2_weight, w2_flex, w2_scale = _swizzle_mxfp4(
                layer.w2_weight.view(torch.uint8),
                layer.w2_weight_scale,
            )

            self.w13_precision_config = PrecisionConfig(
                weight_scale=w13_scale, flex_ctx=FlexCtx(rhs_data=w13_flex)
            )
            self.w2_precision_config = PrecisionConfig(
                weight_scale=w2_scale, flex_ctx=FlexCtx(rhs_data=w2_flex)
            )
            del layer.w13_weight
            del layer.w2_weight
            del layer.w13_weight_scale
            del layer.w2_weight_scale
            layer.w13_weight = w13_weight
            layer.w2_weight = w2_weight
            layer.w13_weight_scale = None
            layer.w2_weight_scale = None
            return
        elif layer.activation == ActivationType.Swiglu and layer.w13_bias is not None:
            e, n, k = layer.w13_weight.shape
            layer.w13_weight.view(torch.uint8).copy_(
                layer.w13_weight.data.view(torch.uint8)
                .view(e, n // 2, 2, k)
                .permute(0, 2, 1, 3)
                .contiguous()
                .view(e, n, k)
            )
            layer.w13_weight_scale.data = (
                layer.w13_weight_scale.data.view(e, n // 2, 2, -1)
                .permute(0, 2, 1, 3)
                .contiguous()
                .view(e, n, -1)
            )
            shuffled_w13 = shuffle_weight_a16w4(layer.w13_weight, 16, True)
            shuffled_w13_scale = shuffle_scale_a16w4(
                layer.w13_weight_scale.view(-1, layer.w13_weight_scale.shape[-1]),
                self.num_experts,
                True,
            )
            shuffled_w2 = shuffle_weight_a16w4(layer.w2_weight, 16, False)
            shuffled_w2_scale = shuffle_scale_a16w4(
                layer.w2_weight_scale.view(-1, layer.w2_weight_scale.shape[-1]),
                self.num_experts,
                False,
            )
            layer.w13_bias.data = (
                layer.w13_bias.data.view(-1, n // 2, 2)
                .permute(0, 2, 1)
                .contiguous()
                .view(-1, n)
            )
        # quark method for moe, split it out?
        elif self.quant_method == "quark":
            s0, s1, _ = layer.w13_weight_scale.shape
            w13_weight_scale = layer.w13_weight_scale.view(s0 * s1, -1)
            w13_weight_scale = fp4_utils.e8m0_shuffle(w13_weight_scale)
            layer.w13_weight_scale.data = w13_weight_scale.view(s0, s1, -1)

            s0, s1, _ = layer.w2_weight_scale.shape
            w2_weight_scale = layer.w2_weight_scale.view(s0 * s1, -1)
            w2_weight_scale = fp4_utils.e8m0_shuffle(w2_weight_scale)
            layer.w2_weight_scale.data = w2_weight_scale.view(s0, s1, -1)
            return
        else:
            shuffled_w13, shuffled_w2 = shuffle_weights(
                layer.w13_weight, layer.w2_weight
            )
            shuffled_w13_scale = fp4_utils.e8m0_shuffle(
                layer.w13_weight_scale.view(self.num_experts, -1)
            )
            shuffled_w2_scale = fp4_utils.e8m0_shuffle(
                layer.w2_weight_scale.view(self.num_experts, -1)
            )

        layer.w13_weight = torch.nn.Parameter(shuffled_w13, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(shuffled_w2, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(
            shuffled_w13_scale, requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            shuffled_w2_scale, requires_grad=False
        )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return mxfp4_w4a16_moe_quant_config(
            w1_bias=layer.w13_bias,
            w2_bias=layer.w2_bias,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: ActivationType = ActivationType.Silu,
    ) -> torch.Tensor:
        if self.use_triton:
            from atom.model_ops.fused_moe_triton import triton_kernel_moe_forward

            return triton_kernel_moe_forward(
                x,
                layer.w13_weight,
                layer.w2_weight,
                router_logits,
                topk=top_k,
                renormalize=renormalize,
                activation=activation,
                w13_precision_config=self.w13_precision_config,
                w2_precision_config=self.w2_precision_config,
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                expert_map=expert_map,
                apply_router_weight_on_input=layer.apply_router_weight_on_input,
                global_num_experts=global_num_experts,
            )

        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            num_fused_shared_experts=layer.num_fused_shared_experts,
            routed_scaling_factor=layer.routed_scaling_factor,
        )
        if self.fused_experts is None:
            return fused_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                expert_mask=expert_map,
                activation=activation,
                quant_type=self.quant_type,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                doweight_stage1=apply_router_weight_on_input,
                hidden_pad=self.hidden_pad,
                intermediate_pad=self.intermediate_pad,
                bias1=layer.w13_bias,
                bias2=layer.w2_bias,
            )
        return self.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation=activation,
            quant_type=self.quant_type,
            apply_router_weight_on_input=apply_router_weight_on_input,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=None,
            a2_scale=None,
            bias1=layer.w13_bias,
            bias2=layer.w2_bias,
            hidden_pad=self.hidden_pad,
            intermediate_pad=self.intermediate_pad,
        )


class Fp8MoEMethod(FusedMoEMethodBase):
    """MoE method for FP8.
    Supports loading FP8 checkpoints with static weight scale and
    dynamic/static activation scale.

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config.
    """

    def __init__(self, quant_config: QuantizationConfig, moe: FusedMoEConfig):
        super().__init__(moe)
        self.quant_config = quant_config
        self.quant_type = self.quant_config["quant_type"]
        self.quant_dtype = self.quant_config["quant_dtype"]
        self.block_quant = (
            self.quant_type == QuantType.per_1x128
            or self.quant_type == QuantType.per_1x32
        )
        self.need_normalize_e4m3fn_to_e4m3fnuz = (
            self.quant_dtype == torch.float8_e4m3fnuz
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):

        # TODO hard code for now
        params_dtype = torch.float8_e4m3fn

        if self.block_quant:
            if self.quant_type == QuantType.per_1x128:
                block_n = 128
                block_k = 128
            elif self.quant_type == QuantType.per_1x32:
                block_n = 1
                block_k = 32
            tp_size = get_tp_group().world_size
            # NOTE: To ensure proper alignment of the block-wise quantization
            # scales, the output_size of the weights for both the gate and up
            # layers must be divisible by block_n.
            # Required by column parallel or enabling merged weights
            if intermediate_size_per_partition % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1 and intermediate_size_per_partition % block_k != 0:
                # Required by row parallel
                raise ValueError(
                    f"The input_size of down's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}."
                )

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if not self.block_quant:
            # Allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
        else:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    2 * ((intermediate_size_per_partition + block_n - 1) // block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size_per_partition + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            assert self.quant_config["is_dynamic"] == True

        # Add the quantization method used (per tensor/grouped/channel)
        # to ensure the weight scales are loaded in properly
        # extra_weight_attrs.update(
        #     {"quant_method": FusedMoeWeightScaleSupported.BLOCK.
        #      value} if self.block_quant else
        #     {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
        # If loading fp8 checkpoint, pass the weight loaders.
        # If loading an fp16 checkpoint, do not (we will quantize in
        #   process_weights_after_loading()
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if not self.quant_config["is_dynamic"]:
            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)
            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        # Lazy import to avoid importing triton too early.
        # from vllm.model_executor.layers.fused_moe.rocm_aiter_fused_moe import (
        #     is_rocm_aiter_moe_enabled, shuffle_weights)

        # self.rocm_aiter_moe_enabled = is_rocm_aiter_moe_enabled()
        # self.rocm_aiter_use_asm = (self.rocm_aiter_moe_enabled
        #                            and envs.VLLM_ROCM_USE_AITER_ASMMOE)

        # TODO (rob): refactor block quant into separate class.
        if self.block_quant:
            assert self.quant_config["is_dynamic"] == True
            if self.need_normalize_e4m3fn_to_e4m3fnuz:
                w13_weight, w13_weight_scale, w13_input_scale = (
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w13_weight, layer.w13_weight_scale, layer.w13_input_scale
                    )
                )
                w2_weight, w2_weight_scale, w2_input_scale = (
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w2_weight, layer.w2_weight_scale, layer.w2_input_scale
                    )
                )
            else:
                w13_weight = layer.w13_weight.data
                w13_weight_scale = layer.w13_weight_scale.data
                w2_weight = layer.w2_weight
                w2_weight_scale = layer.w2_weight_scale

            # torch.compile() cannot use Parameter subclasses.
            layer.w13_weight = nn.Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale = nn.Parameter(w13_weight_scale, requires_grad=False)
            layer.w2_weight = nn.Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale = nn.Parameter(w2_weight_scale, requires_grad=False)

            shuffled_w13, shuffled_w2 = shuffle_weights(
                layer.w13_weight, layer.w2_weight
            )

            layer.w13_weight = torch.nn.Parameter(shuffled_w13, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(shuffled_w2, requires_grad=False)
            return
        else:
            # Fp8 moe kernels require a single activation scale.
            # We take the max of all the scales in case they differ.
            if not self.quant_config["is_dynamic"]:
                if layer.w13_input_scale is None or layer.w2_input_scale is None:
                    raise ValueError(
                        "QuantConfig has static quantization, but found "
                        "activation scales are None."
                    )
                # if (not all_close_1d(layer.w13_input_scale)
                #         or not all_close_1d(layer.w2_input_scale)):
                #     print(
                #         "Found input_scales that are not equal for "
                #         "fp8 MoE layer. Using the maximum across experts "
                #         "for each layer.")
                layer.w13_input_scale = torch.nn.Parameter(
                    layer.w13_input_scale.max(), requires_grad=False
                )
                layer.w2_input_scale = torch.nn.Parameter(
                    layer.w2_input_scale.max(), requires_grad=False
                )
            if self.need_normalize_e4m3fn_to_e4m3fnuz:
                # Normalize the weights and scales
                w13_weight, w13_weight_scale, w13_input_scale = (
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w13_weight, layer.w13_weight_scale, layer.w13_input_scale
                    )
                )
                w2_weight, w2_weight_scale, w2_input_scale = (
                    normalize_e4m3fn_to_e4m3fnuz(
                        layer.w2_weight, layer.w2_weight_scale, layer.w2_input_scale
                    )
                )
                # Reset the parameter
                layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
                layer.w13_weight_scale = torch.nn.Parameter(
                    w13_weight_scale, requires_grad=False
                )
                if w13_input_scale is not None:
                    layer.w13_input_scale = torch.nn.Parameter(
                        w13_input_scale, requires_grad=False
                    )
                layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
                layer.w2_weight_scale = torch.nn.Parameter(
                    w2_weight_scale, requires_grad=False
                )
                if w2_input_scale is not None:
                    layer.w2_input_scale = torch.nn.Parameter(
                        w2_input_scale, requires_grad=False
                    )

            # Fp8 moe kernel needs single weight scale for w13 per expert.
            # We take the max then dequant and requant each expert.
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values
            for expert_id in range(layer.local_num_experts):
                start = 0
                for shard_id in range(2):
                    dq_weight = per_tensor_dequantize(
                        layer.w13_weight[expert_id][start : start + shard_size, :],
                        layer.w13_weight_scale[expert_id][shard_id],
                    )
                    quant_func = get_hip_quant(self.quant_type)
                    layer.w13_weight[expert_id][start : start + shard_size, :], _ = (
                        quant_func(dq_weight, max_w13_scales[expert_id])
                    )
                    start += shard_size

            shuffled_w13, shuffled_w2 = shuffle_weights(
                layer.w13_weight, layer.w2_weight
            )

            layer.w13_weight = torch.nn.Parameter(shuffled_w13, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(shuffled_w2, requires_grad=False)

            layer.w13_weight_scale = torch.nn.Parameter(
                max_w13_scales, requires_grad=False
            )
            return

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return fp8_w8a8_moe_quant_config(
            w1_scale=(layer.w13_weight_scale),
            w2_scale=(layer.w2_weight_scale),
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            block_shape=None,
        )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: ActivationType = ActivationType.Silu,
    ) -> torch.Tensor:
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=x,
            router_logits=router_logits,
            use_grouped_topk=use_grouped_topk,
            top_k=top_k,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            num_fused_shared_experts=layer.num_fused_shared_experts,
            routed_scaling_factor=layer.routed_scaling_factor,
        )
        # per_Tensor not support num_local_tokens so not use mori
        if self.quant_type == QuantType.per_Tensor or self.fused_experts is None:
            return torch.ops.aiter.rocm_aiter_fused_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                expert_mask=expert_map,
                activation=activation.value,
                quant_type=self.quant_type.value,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                doweight_stage1=apply_router_weight_on_input,
            )
        return self.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=False,
            activation=activation,
            quant_type=self.quant_type,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            apply_router_weight_on_input=apply_router_weight_on_input,
        )


def determine_expert_map(
    ep_size: int, ep_rank: int, global_num_experts: int
) -> Tuple[int, Optional[torch.Tensor]]:
    """
    Calculates how many experts should be assigned to each rank for EP and
    creates a mapping from global to local expert index. Experts are
    distributed evenly across ranks. Any remaining are assigned to the
    last rank.

    Args:
        ep_size (int): The size of the expert parallel group
        global_num_experts (int): The total number of experts in the model.

    Returns:
        Tuple[int, Optional[torch.Tensor]]: A tuple containing:
            - local_num_experts (int): The number of experts assigned
                to the current rank.
            - expert_map (Optional[torch.Tensor]): A tensor of shape
                (global_num_experts,) mapping from global to local index.
                Contains -1 for experts not assigned to the current rank.
                Returns None if ep_size is 1.
    """
    assert ep_size > 0
    if ep_size == 1:
        return (global_num_experts, None)

    local_num_experts = global_num_experts // ep_size

    # Create a tensor of size num_experts filled with -1
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)
    # Create a expert map for the local experts
    if ep_rank < (ep_size - 1):
        # Each non-last rank gets local_num_experts experts.
        expert_map[ep_rank * local_num_experts : (ep_rank + 1) * local_num_experts] = (
            torch.arange(0, local_num_experts, dtype=torch.int32)
        )
    else:
        # All remaining experts are assigned to the last rank.
        local_num_experts = global_num_experts - ep_rank * local_num_experts

        expert_map[-local_num_experts:] = torch.arange(
            0, local_num_experts, dtype=torch.int32
        )
    return (local_num_experts, expert_map)


def moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    atom_config = get_current_atom_config()
    self = atom_config.compilation_config.static_forward_context[layer_name]
    return self.forward_impl(hidden_states, router_logits)


def moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="moe_forward",
    op_func=moe_forward,
    mutates_args=["hidden_states"],
    fake_impl=moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


class FusedMoE(torch.nn.Module):
    """FusedMoE layer for MoE models.

    This layer contains both MergedColumnParallel weights (gate_up_proj /
    w13) and RowParallelLinear weights (down_proj/ w2).

    Note: Mixtral uses w1, w2, and w3 for gate, up, and down_proj. We
    copy that naming convention here and handle any remapping in the
    load_weights function in each model implementation.

    Args:
        num_experts: Number of experts in the model
        top_k: Number of experts selected for each token
        hidden_size: Input hidden state size of the transformer
        intermediate_size: Intermediate size of the experts
        params_dtype: Data type for the parameters.
        reduce_results: Whether to all all_reduce on the output of the layer
        renomalize: Whether to renormalize the logits in the fused_moe kernel
        quant_config: Quantization configure.
    """

    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        prefix: str = "",
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        has_bias: bool = True,
        activation: ActivationType = ActivationType.Silu,
        config: Optional[PretrainedConfig] = None,
    ):
        super().__init__()

        self.params_dtype = (
            quant_config["quant_dtype"] if quant_config else torch.get_default_dtype()
        )
        self.quant_config = quant_config
        self.has_bias = has_bias
        # Note: here we guard against accessing the TP and DP groups when
        # uninitialized (this happens when testing)
        # self.tp_size = 1
        tp_size = tp_size if tp_size is not None else get_tp_group().world_size
        dp_size = dp_size if dp_size is not None else get_dp_group().world_size

        atom_config = get_current_atom_config()
        self.moe_parallel_config = FusedMoEParallelConfig.make(
            tp_size, dp_size, atom_config
        )
        tp_rank = 0 if self.tp_size == 1 else get_tp_group().rank_in_group
        self.global_num_experts = num_experts
        if self.use_ep:
            self.local_num_experts, self.expert_map = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts,
            )
        else:
            self.local_num_experts = self.global_num_experts
            self.expert_map = None
        self.top_k = top_k
        self.global_num_experts = num_experts

        self.num_fused_shared_experts = (
            config.n_shared_experts
            if config is not None
            and hasattr(config, "n_shared_experts")
            and is_rocm_aiter_fusion_shared_expert_enabled()
            else 0
        )
        self.routed_scaling_factor = (
            getattr(config, "routed_scaling_factor", 1.0)
            if config is not None and config.torch_dtype != torch.float16
            else 1.0
        )
        self.expert_mask = None
        if self.use_ep:
            expert_mask = torch.ones(
                (self.global_num_experts + self.num_fused_shared_experts + 1,),
                dtype=torch.int32,
            )
            expert_mask[-1] = 0
            expert_mask[: self.global_num_experts] = self.expert_map > -1
            self.expert_mask = expert_mask
            self.expert_map = torch.cat(
                (
                    self.expert_map,
                    torch.tensor(
                        [
                            self.local_num_experts + i
                            for i in range(self.num_fused_shared_experts)
                        ],
                        dtype=torch.int32,
                    ),
                ),
                dim=0,
            )
        if (
            is_rocm_aiter_fusion_shared_expert_enabled()
            and self.num_fused_shared_experts > 0
        ):
            init_aiter_topK_meta_data(
                n_routed_experts=self.global_num_experts,
                n_shared_experts=self.num_fused_shared_experts,
                top_k=self.top_k,
                tp_rank=self.ep_rank if self.use_ep else self.tp_rank,
                tp_size=self.ep_size if self.use_ep else self.tp_size,
                shared_experts_score=(
                    1.0
                    if is_rocm_aiter_fuse_routed_scaling_factor()
                    else 1 / self.routed_scaling_factor
                ),
                max_num_tokens=atom_config.max_num_batched_tokens,
                is_EP=self.use_ep,
            )
        if is_rocm_aiter_fusion_shared_expert_enabled():
            self.local_num_experts += self.num_fused_shared_experts
        assert intermediate_size % self.tp_size == 0
        self.hidden_size = hidden_size
        self.intermediate_size_per_partition = intermediate_size // self.tp_size
        self.reduce_results = reduce_results
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        if self.use_grouped_topk:
            assert num_expert_group is not None and topk_group is not None
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.custom_routing_function = custom_routing_function
        self.scoring_func = scoring_func
        self.e_score_correction_bias = e_score_correction_bias
        self.activation = activation

        self.use_chunked = get_dp_group().world_size > 1

        if self.scoring_func != "softmax" and not self.use_grouped_topk:
            raise ValueError(
                "Only softmax scoring function is supported for " "non-grouped topk."
            )
        moe = FusedMoEConfig(
            num_experts=self.global_num_experts,
            experts_per_token=self.top_k,
            hidden_dim=hidden_size,
            num_local_experts=self.local_num_experts,
            moe_parallel_config=self.moe_parallel_config,
            in_dtype=atom_config.torch_dtype,
            max_num_tokens=atom_config.max_num_batched_tokens,
            has_bias=self.has_bias,
            # is_act_and_mul=True,
            is_lora_enabled=False,
        )
        self.moe_config = moe

        # Note: get_quant_method will look at the layer's local_num_experts
        # for heuristic purposes, so it must be initialized first.
        if quant_config["quant_type"] == QuantType.No:
            self.quant_method: Optional[QuantizeMethodBase] = UnquantizedFusedMoEMethod(
                moe
            )
        elif quant_config["quant_dtype"] == dtypes.fp8:
            self.quant_method = Fp8MoEMethod(quant_config, moe)
        elif quant_config["quant_dtype"] == dtypes.fp4x2:
            self.quant_method = Mxfp4MoEMethod(quant_config, moe)
        else:
            raise ValueError(f"Unsupported quant dtype: {quant_config['quant_dtype']}")

        assert self.quant_method is not None

        self.apply_router_weight_on_input = apply_router_weight_on_input
        moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition": self.intermediate_size_per_partition,
            "params_dtype": self.params_dtype,
            "weight_loader": self.weight_loader,
        }
        self.quant_method.create_weights(layer=self, **moe_quant_params)
        compilation_config = atom_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError("Duplicate layer name: {}".format(prefix))
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix

    @property
    def tp_size(self):
        return self.moe_parallel_config.tp_size

    @property
    def dp_size(self):
        return self.moe_parallel_config.dp_size

    @property
    def ep_size(self):
        return self.moe_parallel_config.ep_size

    @property
    def tp_rank(self):
        return self.moe_parallel_config.tp_rank

    @property
    def dp_rank(self):
        return self.moe_parallel_config.dp_rank

    @property
    def ep_rank(self):
        return self.moe_parallel_config.ep_rank

    @property
    def use_ep(self):
        return self.moe_parallel_config.use_ep

    def _load_per_tensor_weight_scale(
        self,
        shard_id: str,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        expert_id: int,
    ):
        param_data = param.data
        # for per tensor weight quantization
        if shard_id in ("w1", "w3"):
            # We have to keep the weight scales of w1 and w3 because
            # we need to re-quantize w1/w3 weights after weight loading.
            idx = 0 if shard_id == "w1" else 1
            param_data[expert_id][idx] = loaded_weight
        # If we are in the row parallel case (down_proj)
        elif shard_id == "w2":
            param_data[expert_id] = loaded_weight

    def _load_model_weight_or_group_weight_scale(
        self,
        shard_dim: int,
        expert_data: torch.Tensor,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full_w2: bool = False,
    ):
        """
        Load grouped weight scales for group quantization or model weights
            :param shard_dim: dimension to shard
            :param expert_data: parameter for a particular expert
            :param shard_id: either w1, w2, or w3
            :param loaded_weight: checkpoint weight to load into the param
            :param tp_rank: tensor parallel rank
            :param load_full_w2: whether or not the w2 loaded should be sharded.
        """
        if shard_id == "w2":
            # In the case where we have actorder/g_idx, we do not partition the
            # w2 scales, as indicated by `load_full` argument, for all tp cases
            self._load_w2(
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                load_full=load_full_w2,
            )
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_per_channel_weight_scale(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):
        # for per channel weight quantization
        if shard_id == "w2":
            expert_data.copy_(loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    def _load_w13(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):

        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        shard_size = expert_data.shape[shard_dim] // 2
        loaded_weight = loaded_weight.narrow(
            shard_dim, shard_size * tp_rank, shard_size
        )
        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        if shard_id == "w1":
            expert_data = expert_data.narrow(shard_dim, 0, shard_size)
        # w3, up_proj: Load into second logical weight of w13.
        else:
            assert shard_id == "w3"
            expert_data = expert_data.narrow(shard_dim, shard_size, shard_size)
        if expert_data.dtype != dtypes.fp4x2:
            expert_data.copy_(loaded_weight)
        else:
            expert_data.view(torch.uint8).copy_(loaded_weight.view(torch.uint8))

    def _load_w2(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full: bool = False,
    ):

        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        shard_size = expert_data.shape[shard_dim]
        if not load_full:
            loaded_weight = loaded_weight.narrow(
                shard_dim, shard_size * tp_rank, shard_size
            )
        # w2, down_proj: Load into only logical weight of w2.
        if expert_data.dtype != dtypes.fp4x2:
            expert_data.copy_(loaded_weight)
        else:
            expert_data.view(torch.uint8).copy_(loaded_weight.view(torch.uint8))

    def _load_single_value(
        self, param: torch.nn.Parameter, loaded_weight: torch.Tensor, expert_id: int
    ):
        param_data = param.data

        # Input scales can be loaded directly and should be equal.
        param_data[expert_id] = loaded_weight

    def _load_g_idx(
        self,
        shard_id: str,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        tp_rank: int,
    ):

        if shard_id == "w2":
            self._load_w2(
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )
        else:
            assert shard_id in ("w1", "w3")
            expert_data.copy_(loaded_weight)

    def _map_global_expert_id_to_local_expert_id(self, expert_id: int) -> int:
        if self.expert_map is None:
            return expert_id
        return self.expert_map[expert_id].item()

    def mxf4_merged_weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
    ):
        # (FIXME) for gpt-oss all experts are combined
        mxfp4_block = 32
        ep_rank_start = self.ep_rank * self.local_num_experts
        ep_rank_end = ep_rank_start + self.local_num_experts
        tp_rank_start = self.tp_rank * self.intermediate_size_per_partition
        tp_rank_end = tp_rank_start + self.intermediate_size_per_partition
        if param is getattr(self, "w13_bias", None):
            if self.use_ep:
                narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = loaded_weight[:, 2 * tp_rank_start : 2 * tp_rank_end]
            dim1 = narrow_weight.shape[1]
            param[:, :dim1].copy_(narrow_weight)
        elif param is getattr(self, "w2_bias", None):
            if self.use_ep:
                narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = loaded_weight
                if self.tp_rank != 0:
                    narrow_weight.zero_()
            dim1 = narrow_weight.shape[1]
            param[:, :dim1].copy_(narrow_weight)
        elif param is getattr(self, "w13_weight", None):
            loaded_weight = loaded_weight.view(*loaded_weight.shape[:2], -1)
            if self.use_ep:
                narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = loaded_weight[
                    :, 2 * tp_rank_start : 2 * tp_rank_end, ...
                ]
            dim1, dim2 = narrow_weight.shape[1:]
            param.view(torch.uint8)[:, :dim1, :dim2].copy_(
                narrow_weight.view(torch.uint8)
            )
        elif param is getattr(self, "w2_weight", None):
            loaded_weight = loaded_weight.view(*loaded_weight.shape[:2], -1)
            if self.use_ep:
                narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = loaded_weight[
                    ..., tp_rank_start // 2 : tp_rank_end // 2
                ]
            dim1, dim2 = narrow_weight.shape[1:]
            param.view(torch.uint8)[:, :dim1, :dim2].copy_(
                narrow_weight.view(torch.uint8)
            )
        elif param is getattr(self, "w13_weight_scale", None):
            if self.use_ep:
                narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = loaded_weight[
                    :, 2 * tp_rank_start : 2 * tp_rank_end, ...
                ]
            dim1, dim2 = narrow_weight.shape[1:]
            param[:, :dim1, :dim2].copy_(narrow_weight)
        elif param is getattr(self, "w2_weight_scale", None):
            if self.use_ep:
                narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = loaded_weight[
                    ..., tp_rank_start // mxfp4_block : tp_rank_end // mxfp4_block
                ]
            dim1, dim2 = narrow_weight.shape[1:]
            param[:, :dim1, :dim2].copy_(narrow_weight)

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str = "",
        shard_id: str = "",
        expert_id: int = 0,
    ) -> None:
        if self.quant_config["quant_dtype"] == dtypes.fp4x2 and weight_name == "":
            self.mxf4_merged_weight_loader(param, loaded_weight)
            return

        expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
        if expert_id == -1:
            return

        # compressed-tensors checkpoints with packed weights are stored flipped
        # TODO (mgoin): check self.quant_method.quant_config.quant_format
        # against known CompressionFormat enum values that have this quality
        if self.quant_method.__class__.__name__ in (
            "CompressedTensorsWNA16MarlinMoEMethod",
            "CompressedTensorsWNA16MoEMethod",
        ):
            loaded_weight = loaded_weight.t().contiguous()

        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(
                f"shard_id must be ['w1','w2','w3'] but " f"got {shard_id}."
            )

        # Fetch the dim to shard the parameter/loaded weight
        # based on the shard id. This will be whatever
        # dimension intermediate_size_per_partition is used.
        SHARD_ID_TO_SHARDED_DIM = {"w1": 0, "w2": 1, "w3": 0}

        # is_transposed: if the dim to shard the weight
        # should be flipped. Required by GPTQ, compressed-tensors
        # should be whatever dimension intermediate_size_per_partition is
        is_transposed = getattr(param, "is_transposed", False)
        shard_dim = SHARD_ID_TO_SHARDED_DIM[shard_id]
        if is_transposed:
            shard_dim = int(not shard_dim)

        full_load = len(loaded_weight.shape) == 3
        if full_load:
            shard_dim += 1

        expert_data = param.data if full_load else param.data[expert_id]
        # Case input scale: input_scale loading is only supported for fp8
        if "input_scale" in weight_name:
            # this is needed for compressed-tensors only
            loaded_weight = loaded_weight.to(param.data.device)

            if (
                param.data[expert_id] != 1
                and (param.data[expert_id] - loaded_weight).abs() > 1e-5
            ):
                raise ValueError(
                    "input_scales of w1 and w3 of a layer "
                    f"must be equal. But got {param.data[expert_id]} "
                    f"vs. {loaded_weight}"
                )

            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return

        # Case g_idx
        if "g_idx" in weight_name:
            self._load_g_idx(
                shard_dim=0,
                shard_id=shard_id,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank,
            )
            return

        # Case weight scales, zero_points and offset
        if "scale" in weight_name or "zero" in weight_name or "offset" in weight_name:
            # load the weight scales and zp based on the quantization scheme
            # supported weight scales/zp can be found in
            # FusedMoeWeightScaleSupported
            # TODO @dsikka: once hardened, refactor to use vLLM Parameters
            # specific to each case
            quant_method = self.quant_config["quant_type"]
            if quant_method == QuantType.per_Token:
                self._load_per_channel_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                )
            elif quant_method in [
                QuantType.per_1x128,
                QuantType.per_1x32,
            ]:
                self._load_model_weight_or_group_weight_scale(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=self.tp_rank,
                    load_full_w2=getattr(param, "load_full_w2", False),
                )
            elif quant_method == QuantType.per_Tensor:
                self._load_per_tensor_weight_scale(
                    shard_id=shard_id,
                    param=param,
                    loaded_weight=loaded_weight,
                    expert_id=expert_id,
                )
            return

        # Case weight_shape
        if "weight_shape" in weight_name:
            # only required by compressed-tensors
            self._load_single_value(
                param=param, loaded_weight=loaded_weight, expert_id=expert_id
            )
            return

        # Case model weights
        if "weight" in weight_name:
            self._load_model_weight_or_group_weight_scale(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=self.tp_rank,
            )
            return

    @staticmethod
    def select_experts(
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        use_grouped_topk: bool,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        num_fused_shared_experts: int = 0,
        routed_scaling_factor: float = 1.0,
    ):

        # DeekSeekv2 uses grouped_top_k
        if use_grouped_topk:
            assert topk_group is not None
            assert num_expert_group is not None
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias,
                routed_scaling_factor=routed_scaling_factor,
                num_fused_shared_experts=num_fused_shared_experts,
            )
        else:
            topk_weights, topk_ids = fused_topk(
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
            )

        return topk_weights, topk_ids

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        return torch.ops.aiter.moe_forward(
            hidden_states, router_logits, self.layer_name
        )

    def forward_impl_graph(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ):
        original_hidden_size = None
        # Use all_gather/reduce_scatter when DP > 1 but not using mori all2all kernels
        if self.dp_size > 1 and not self.moe_parallel_config.use_all2all_kernels:
            hidden_states, original_hidden_size = all_gather_with_padding(hidden_states)
            router_logits, _ = all_gather_with_padding(router_logits)

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_mask,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )

        # Use reduce_scatter when DP > 1 but not using mori all2all kernels
        if self.dp_size > 1 and not self.moe_parallel_config.use_all2all_kernels:
            final_hidden_states = reduce_scatter_with_unpadding(
                final_hidden_states, original_hidden_size
            )

        if self.reduce_results and (self.tp_size > 1 or self.ep_size > 1):
            # Default set to False. (May have to add shared expert outputs.)
            final_hidden_states = get_tp_group().all_reduce(
                final_hidden_states, ca_fp8_quant=False
            )

        return final_hidden_states

    def forward_impl(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        assert self.quant_method is not None
        # cuda graph not supported forward with combine and dispatch
        if self.use_chunked:
            return self.forward_impl_graph(hidden_states, router_logits)
            # return self.forward_impl_chunked(hidden_states, router_logits)

        dp_group = get_dp_group()
        if dp_group.world_size > 1:
            cu_tokens_across_dp_cpu = (
                get_forward_context().dp_metadata.cu_tokens_across_dp_cpu
            )

            hidden_states = naive_multicast(hidden_states, cu_tokens_across_dp_cpu)
            router_logits = naive_multicast(router_logits, cu_tokens_across_dp_cpu)

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_mask,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )

        dp_group = get_dp_group()
        if dp_group.world_size > 1:
            dp_rank = dp_group.rank_in_group
            start = 0 if dp_rank == 0 else cu_tokens_across_dp_cpu[dp_rank - 1]
            end = cu_tokens_across_dp_cpu[dp_rank]

            all_hidden_states = get_dp_group().all_reduce(final_hidden_states)
            final_hidden_states = all_hidden_states[start:end, :]

        if self.reduce_results and (self.tp_size > 1 or self.ep_size > 1):
            # Default set to False. (May have to add shared expert outputs.)
            final_hidden_states = get_tp_group().all_reduce(
                final_hidden_states, ca_fp8_quant=False
            )

        return final_hidden_states

    @classmethod
    def make_expert_params_mapping(
        cls,
        ckpt_gate_proj_name: str,
        ckpt_down_proj_name: str,
        ckpt_up_proj_name: str,
        num_experts: int,
        has_bias: bool = False,
    ) -> List[Tuple[str, str, int, str]]:

        return [
            # (param_name, weight_name, expert_id, shard_id)
            (
                (
                    "experts.w13_"
                    if weight_name in [ckpt_gate_proj_name, ckpt_up_proj_name]
                    else "experts.w2_"
                ),
                f"experts.{expert_id}.{weight_name}.",
                expert_id,
                shard_id,
            )
            for expert_id in range(num_experts)
            for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]

    def extra_repr(self) -> str:

        s = (
            f"global_num_experts={self.global_num_experts}, "
            f"local_num_experts={self.local_num_experts}, "
            f"top_k={self.top_k}, "
            f"intermediate_size_per_partition={self.intermediate_size_per_partition}, "  # noqa: E501
            f"tp_size={self.tp_size},\n"
            f"ep_size={self.ep_size}, "
            f"reduce_results={self.reduce_results}, "
            f"renormalize={self.renormalize}, "
            f"use_grouped_topk={self.use_grouped_topk}"
        )

        if self.use_grouped_topk:
            s += f", num_expert_group={self.num_expert_group}, topk_group={self.topk_group}"  # noqa: E501

        s += f", scoring_func='{self.scoring_func}', activation='{self.activation}'"  # noqa: E501

        return s
