# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Callable, List, Optional, Tuple

import torch
from aiter import ActivationType, QuantType, dtypes, get_hip_quant, topk_gating
from aiter.dist.parallel_state import get_dp_group, get_tp_group
from aiter.fused_moe import fused_moe
from aiter.jit.utils.chip_info import get_gfx
from aiter.jit.utils.torch_guard import torch_compile_guard
from aiter.ops.flydsl.moe_common import GateMode
from aiter.ops.shuffle import moe_shuffle_scale, shuffle_weight
from atom.config import (
    Config,
    QuantizationConfig,
    get_current_atom_config,
)
from atom.model_loader.weight_utils import set_weight_attrs
from atom.model_ops.base_config import QuantizeMethodBase
from atom.model_ops.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEQuantConfig,
    fp8_w8a8_moe_quant_config,
    mxfp4_w4a8_moe_quant_config,
    mxfp4_w4a16_moe_quant_config,
)
from atom.model_ops.fused_moe.modular_kernel import (
    FusedMoEModularKernel,
    FusedMoEPrepareAndFinalize,
)
from atom.model_ops.fused_moe.mori_prepare_finalize import MoriPrepareAndFinalize
from atom.model_ops.topK import (
    init_aiter_topK_meta_data,
    is_rocm_aiter_fuse_routed_scaling_factor,
    is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config,
)
from atom.model_ops.topK import rocm_aiter_grouped_topk as grouped_topk
from atom.model_ops.topK import rocm_aiter_topk_softmax as fused_topk
from atom.model_ops.utils import (
    _has_module,
    atom_parameter,
    normalize_e4m3fn_to_e4m3fnuz,
    per_tensor_dequantize,
    shuffle_weights,
)
from atom.plugin.vllm.moe import FusedMoEDecoratorForPluginMode
from atom.quant_spec import LayerQuantConfig, should_skip_online_quant
from atom.quantization.quark.utils import (
    quant_weight_online,
    weight_dequant_fp8,
    weight_dequant_mxfp8,
)
from atom.utils import envs
from atom.utils.custom_register import direct_register_custom_op
from atom.utils.decorators import mark_trace
from atom.utils.forward_context import get_forward_context
from torch import nn
from transformers import PretrainedConfig

logger = logging.getLogger("atom")


class MoEActivationQuant(Enum):
    BF16 = "bf16"
    FP8 = "fp8"
    FP4 = "fp4"

    @staticmethod
    def from_model_config(a_quant_dtype: str | None) -> "MoEActivationQuant":
        if a_quant_dtype is None or a_quant_dtype == "":
            return MoEActivationQuant.BF16
        prefix = a_quant_dtype.split("_")[0]
        if prefix == "fp8":
            return MoEActivationQuant.FP8
        if prefix in ("fp4", "uint8"):
            return MoEActivationQuant.FP4
        return MoEActivationQuant.BF16


class FusedMoeWeightScaleSupported(Enum):
    """Supported quantization strategies for MoE weight scales."""

    TENSOR = "tensor"
    CHANNEL = "channel"
    GROUP = "group"
    BLOCK = "block"


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

        # Only flatten DP into TP/EP when enable_dp_attention is True.
        # Otherwise, use pure DP for MoE.
        enable_dp_attention = parallel_config.enable_dp_attention

        # When EP shards across the flattened DP * TP space (vLLM plugin under
        # EP), the ep rank must be computed in that flattened group space.
        flatten_tp_across_dp_for_moe = (
            enable_dp_attention or parallel_config.moe_ep_flatten_tp_across_dp
        )

        use_ep = dp_size_ * tp_size_ > 1 and parallel_config.enable_expert_parallel

        dp_size = dp_size_
        dp_rank = get_dp_group().rank_in_group if dp_size > 1 else 0

        if flatten_tp_across_dp_for_moe:
            tp_size, tp_rank = flatten_tp_across_dp(dp_rank)
        else:
            tp_size = tp_size_
            tp_rank = 0 if tp_size_ == 1 else get_tp_group().rank_in_group

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


def pad_for_all_gather(x: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Zero-pad ``x`` along dim 0 up to the uniform all-gather batch size.

    Every DP rank must contribute the same number of rows to the uniform
    all-gather, so a short batch is padded up to ``graph_bs`` (scaled by the
    per-sequence query length when decoding with MTP > 1).

    The padding MUST be zeros, never uninitialized memory: padded rows are
    all-gathered across DP ranks and fed straight into the aiter fused-MoE
    expert GEMM, where garbage values leak into real tokens' outputs.
    Bisection traced a ~0.7pp GSM8K drop at large batch to a bare
    ``torch.empty`` pad here; explicitly zeroing the pad rows fixes it.

    Returns the (possibly padded) tensor and the original row count so the
    caller can unpad after reduce-scatter.
    """
    ctx = get_forward_context()
    max_batch_size = ctx.context.graph_bs
    if not ctx.context.is_prefill and ctx.attn_metadata is not None:
        max_batch_size *= ctx.attn_metadata.max_seqlen_q

    original_batch_size = x.shape[0]
    padding_size = max_batch_size - original_batch_size
    if padding_size <= 0:
        return x, original_batch_size

    padding_shape = list(x.shape)
    padding_shape[0] = max_batch_size
    padded_x = torch.empty(padding_shape, device=x.device, dtype=x.dtype)
    padded_x[:original_batch_size, :].copy_(x)
    # padded_x[original_batch_size:, :].zero_()
    return padded_x, original_batch_size


def all_gather_with_padding(
    x: torch.Tensor, use_cag: bool = True
) -> Tuple[torch.Tensor, int]:
    padded_x, original_batch_size = pad_for_all_gather(x)
    # use_custom=True routes through CA IPC (outplace_all_gather). Default
    # use_custom=False falls back to torch.distributed.all_gather_into_tensor
    # (NCCL), whose WorkNCCL end-event recorded inside CUDAGraph capture is
    # later queried by the watchdog thread -> hipErrorCapturedEvent crash.
    gathered_hidden_states = get_dp_group().all_gather(
        padded_x, use_custom=use_cag, dim=0
    )
    return gathered_hidden_states, original_batch_size


def reduce_scatter_with_unpadding(
    x: torch.Tensor, original_batch_size: int
) -> torch.Tensor:
    dp_group = get_dp_group()
    scattered_output = dp_group.reduce_scatter_tensor(x)

    # Drop the rows that pad_for_all_gather zero-padded (padding is on dim 0).
    if scattered_output.shape[0] > original_batch_size:
        scattered_output = scattered_output[:original_batch_size]

    return scattered_output


def all_gatherv(x: torch.Tensor, sizes: list[int], group) -> torch.Tensor:
    return group.device_communicator.all_gatherv(x, dim=0, sizes=list(sizes))


def reduce_scatterv(x: torch.Tensor, sizes: list[int], group) -> torch.Tensor:
    return group.device_communicator.reduce_scatterv(x, dim=0, sizes=list(sizes))


def dp_gather_hidden_and_router(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    dp_eager_mode: bool,
    ctx,
    dp_group,
):
    """All-gather ``hidden_states`` + ``router_logits`` across DP ranks.

    - **Eager (variable-length) mode**: per-rank token counts may differ, so
      we ``all_gatherv`` with per-rank sizes. Both tensors are concatenated
      along the last dim and gathered in a single collective to avoid the
      cost of two separate rounds (caller splits them back out).
    - **Uniform mode** (DP-decode-only / CUDAGraph path): every rank has the
      same token count, so a plain padded ``all_gather`` per tensor is
      enough.

    Returns ``(hidden_states, router_logits, original_hidden_size, sizes)``;
    ``sizes`` is non-None only in eager mode (needed later for
    ``reduce_scatterv``).
    """
    if dp_eager_mode:
        sizes = ctx.dp_metadata.get_sizes_across_dp()
        original_hidden_size = hidden_states.shape[0]
        h_dim = hidden_states.shape[-1]
        r_dim = router_logits.shape[-1]
        r_dtype = router_logits.dtype
        # Cast router_logits to hidden_states.dtype for the single fused
        # gather; cast back after split. (all_gatherv requires uniform dtype.)
        r_for_gather = (
            router_logits
            if r_dtype == hidden_states.dtype
            else router_logits.to(hidden_states.dtype)
        )
        combined = torch.cat([hidden_states, r_for_gather], dim=-1)
        combined = all_gatherv(combined, sizes, dp_group)
        hidden_states, router_logits = combined.split([h_dim, r_dim], dim=-1)
        hidden_states = hidden_states.contiguous()
        if router_logits.dtype != r_dtype:
            router_logits = router_logits.to(r_dtype)
        else:
            router_logits = router_logits.contiguous()
        return hidden_states, router_logits, original_hidden_size, sizes

    hidden_states, original_hidden_size = all_gather_with_padding(hidden_states)
    router_logits, _ = all_gather_with_padding(router_logits)
    return hidden_states, router_logits, original_hidden_size, None


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
        fused_shared_experts_scoring_func: Optional[str] = None,
        apply_router_weight_on_input: bool = False,
        activation: ActivationType = ActivationType.Silu,
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
            # mori_dtype = (
            #     quant_config.quant_dtype
            #     if is_fp8 and quant_type is not None
            #     else torch.bfloat16
            # )
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
                max_num_tokens_per_dp_rank=moe.max_num_tokens,
                # input_dtype=moe.in_dtype,
                input_dtype=moe.in_dtype,
                num_local_experts=moe.num_experts // all2all_manager.world_size,
                num_experts_per_token=moe.experts_per_token,
                gpu_per_node=moe.moe_parallel_config.local_ep_size,
            )
            from atom.utils.tbo.ubatching import tbo_enabled

            handle = all2all_manager.get_handle(all_to_all_args)
            is_async = tbo_enabled()
            atom_config = get_current_atom_config()
            low_latency = getattr(atom_config, "enable_low_latency", False)

            # We not use quant for mori now
            use_fp8_dispatch = False
            quant_type = None

            common_args = dict(
                rank=all2all_manager.rank,
                world_size=all2all_manager.world_size,
                hidden_dim=moe.hidden_dim,
                scale_dim=scale_dim,
                max_num_inp_token_per_rank=16384,
                num_local_experts=moe.num_experts // all2all_manager.world_size,
                num_experts_per_token=moe.experts_per_token,
                gpu_per_node=moe.moe_parallel_config.local_ep_size,
                data_type_itemsize=moe.in_dtype.itemsize,
                max_token_type_size=moe.in_dtype.itemsize,
            )

            tbo_mori_ops = None
            sync_handle = handle  # IntraNode handle for prefill (sync path)
            if is_async:
                from atom.model_ops.fused_moe.mori_prepare_finalize import (
                    _NUM_TBO_UBATCHES,
                    init_mori_op,
                )

                tbo_mori_ops = [
                    init_mori_op(
                        **common_args,
                        low_latency=low_latency,
                        instance_id=i,
                    )
                    for i in range(_NUM_TBO_UBATCHES)
                ]

            prepare_finalize = MoriPrepareAndFinalize(
                sync_handle,
                max_tokens_per_rank=moe.max_num_tokens,
                num_dispatchers=all2all_manager.world_size,
                use_fp8_dispatch=use_fp8_dispatch,
                quant_type=quant_type,
                is_async=is_async,
                tbo_mori_ops=tbo_mori_ops,
                low_latency=low_latency,
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
        w13_weight = atom_parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            )
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = atom_parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            )
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

    def _maybe_pad_weight(self, weight: torch.Tensor) -> torch.Tensor:
        return weight

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)

        layer.w13_weight = atom_parameter(self._maybe_pad_weight(layer.w13_weight.data))
        layer.w2_weight = atom_parameter(self._maybe_pad_weight(layer.w2_weight.data))
        # reshaping weights is required for aiter moe kernel.
        shuffle_weights(layer.w13_weight, layer.w2_weight)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return FUSED_MOE_UNQUANTIZED_CONFIG

    @mark_trace(prefix="unquantized_moe", torch_compile=False)
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
        fused_shared_experts_scoring_func: Optional[str] = None,
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
            num_routing_experts=global_num_experts,
            num_fused_shared_experts=layer.num_fused_shared_experts,
            fused_shared_experts_scoring_func=fused_shared_experts_scoring_func,
            routed_scaling_factor=layer.routed_scaling_factor,
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
                expert_mask=layer.expert_mask,
            )
        return fused_moe(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weight=topk_weights,
            topk_ids=topk_ids,
            expert_mask=layer.expert_mask,
            activation=activation,
        )


def rocm_asm_moe_impl(
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
    from aiter.fused_moe_bf16_asm import asm_moe

    activation_ = ActivationType(activation)
    quant_type_ = QuantType(quant_type)

    # - fc1_scale: [E, inter_dim*2, 1]
    # - fc2_scale: [E, model_dim, 1]
    # - fc1_smooth_scale: [E, model_dim]
    # - fc2_smooth_scale: [E, inter_dim]
    fc1_scale_fixed = w1_scale
    fc2_scale_fixed = w2_scale
    fc1_smooth_scale_fixed = a1_scale
    fc2_smooth_scale_fixed = a2_scale

    a16_mode = (
        quant_type_ in [QuantType.per_Token, QuantType.per_1x128]
        and hidden_states.dtype in [torch.float16, torch.bfloat16]
        and w1.dtype in [torch.int8, torch.uint8, torch.float8_e4m3fnuz]
        and fc1_smooth_scale_fixed is not None
        and fc2_smooth_scale_fixed is not None
    )

    return asm_moe(
        hidden_states,
        w1,
        w2,
        topk_weight,
        topk_ids,
        fc1_scale=fc1_scale_fixed,
        fc2_scale=fc2_scale_fixed,
        fc1_smooth_scale=fc1_smooth_scale_fixed,
        fc2_smooth_scale=fc2_smooth_scale_fixed,
        a16=a16_mode,
        per_tensor_quant_scale=None,
        block_shape=None,
        expert_mask=expert_mask,
        activation=activation_,
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
    swiglu_limit: float = 0.0,
    gate_mode: str = GateMode.SEPARATED.value,
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
        swiglu_limit=swiglu_limit,
        gate_mode=gate_mode,
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
    swiglu_limit: float = 0.0,
    gate_mode: str = GateMode.SEPARATED.value,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="rocm_aiter_fused_moe",
    op_func=rocm_aiter_fused_moe_impl,
    mutates_args=[],
    fake_impl=rocm_aiter_fused_moe_fake,
)


class Mxfp4MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: LayerQuantConfig, moe: FusedMoEConfig):
        super().__init__(moe)
        self.quant_config = quant_config
        self.quant_type = quant_config.quant_type
        self.quant_dtype = quant_config.quant_dtype
        self.quant_method = quant_config.quant_method or ""
        self.static_input_scales = not quant_config.is_dynamic
        self.is_guinterleave = envs.ATOM_MOE_GU_ITLV
        self.block_quant = (
            self.quant_type == QuantType.per_1x128
            or self.quant_type == QuantType.per_1x32
        )
        gfx = get_gfx()
        self.is_gfx1250 = gfx == "gfx1250"
        # gfx1250 grouped a8w4 MoE kernel only supports the non-interleaved
        # (gate|up separated) scale layout; reject is_guinterleave up front.
        if self.is_gfx1250 and self.is_guinterleave:
            raise NotImplementedError(
                "gfx1250 MoE only supports is_guinterleave=False; "
                "unset ATOM_MOE_GU_ITLV."
            )
        if envs.is_set("ATOM_USE_TRITON_MOE"):
            self.use_triton = envs.ATOM_USE_TRITON_MOE
        else:
            self.use_triton = gfx.startswith("gfx94") or (
                gfx.startswith("gfx95") and envs.ATOM_USE_TRITON_GEMM
            )
        self.act_quant = MoEActivationQuant.from_model_config(moe.a_quant_dtype)

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
        w13_weight = atom_parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,  # TP included
                hidden_size // 2,
                dtype=weight_dtype,
            )
        )
        layer.register_parameter("w13_weight", w13_weight)
        # Zero-fill padding region: FP4 dtype doesn't support torch.zeros,
        # so we zero the underlying bytes to avoid garbage in padded rows.
        w13_weight.data.view(torch.uint8).zero_()
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w13_weight_scale = atom_parameter(
            torch.zeros(
                num_experts,
                2 * intermediate_size_per_partition_after_pad,
                hidden_size // mxfp4_block,
                dtype=scale_dtype,
            )
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)

        if layer.has_bias:
            w13_bias = atom_parameter(
                torch.zeros(
                    num_experts,
                    2 * intermediate_size_per_partition_after_pad,
                    dtype=torch.bfloat16,
                )
            )
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
        else:
            layer.register_parameter("w13_bias", None)

        # down_proj (row parallel)
        w2_weight = atom_parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // 2,  # TP included
                dtype=weight_dtype,
            )
        )
        layer.register_parameter("w2_weight", w2_weight)
        w2_weight.data.view(torch.uint8).zero_()
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_weight_scale = atom_parameter(
            torch.zeros(
                num_experts,
                hidden_size,
                intermediate_size_per_partition_after_pad // mxfp4_block,
                dtype=scale_dtype,
            )
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        if layer.has_bias:
            w2_bias = atom_parameter(
                torch.zeros(
                    num_experts,
                    hidden_size,
                    dtype=torch.bfloat16,
                )
            )
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)
        else:
            layer.register_parameter("w2_bias", None)

        layer.w13_swizzle_layout = None
        layer.w2_swizzle_layout = None

        if self.static_input_scales:
            w13_input_scale = atom_parameter(
                torch.ones(num_experts, dtype=torch.float32)
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = atom_parameter(
                torch.ones(num_experts, dtype=torch.float32)
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer):
        if layer.w13_bias is not None:
            layer.w13_bias.data = layer.w13_bias.data.to(torch.float32)
        if layer.w2_bias is not None:
            layer.w2_bias.data = layer.w2_bias.data.to(torch.float32)

        if self.static_input_scales:
            layer.w13_input_scale = atom_parameter(
                layer.w13_input_scale.max().to(torch.float32)
            )
            layer.w2_input_scale = atom_parameter(
                layer.w2_input_scale.max().to(torch.float32)
            )

        if self.use_triton:
            from atom.config import get_current_atom_config
            from atom.model_ops.fused_moe_triton import _swizzle_mxfp4

            atom_config = get_current_atom_config()

            # Stash dense (pre-swizzle) shared-expert weights so the always-on
            # shared expert can be evaluated by a standalone dense MXFP4 GEMM
            # (gemm_a16wfp4, see Mxfp4MoEMethod._apply_shared_experts_dense)
            # instead of being fused into grouped_topk routing. The shared
            # experts occupy the last ``num_fused_shared_experts`` slots of the
            # routed weight tensors; their raw per-expert layout
            # (N, K // 2) + scale (N, K // 32) is exactly what gemm_a16wfp4
            # consumes, whereas _swizzle_mxfp4 below reorders the scales into
            # the MoE-kernel-only CDNA4 layout.
            n_shared = layer.num_fused_shared_experts
            if n_shared > 0:
                layer.shared_w13_weight = (
                    layer.w13_weight.data[-n_shared:].view(torch.uint8).contiguous()
                )
                layer.shared_w13_weight_scale = layer.w13_weight_scale.data[
                    -n_shared:
                ].contiguous()
                layer.shared_w2_weight = (
                    layer.w2_weight.data[-n_shared:].view(torch.uint8).contiguous()
                )
                layer.shared_w2_weight_scale = layer.w2_weight_scale.data[
                    -n_shared:
                ].contiguous()
                if layer.w13_bias is not None:
                    layer.shared_w13_bias = layer.w13_bias.data[-n_shared:].contiguous()
                else:
                    layer.shared_w13_bias = None
                if layer.w2_bias is not None:
                    layer.shared_w2_bias = layer.w2_bias.data[-n_shared:].contiguous()
                else:
                    layer.shared_w2_bias = None

            (
                w13_weight,
                w13_scale,
                w13_swizzle_layout,
                w2_weight,
                w2_scale,
                w2_swizzle_layout,
            ) = _swizzle_mxfp4(
                layer.w13_weight.view(torch.uint8),
                layer.w13_weight_scale,
                layer.w2_weight.view(torch.uint8),
                layer.w2_weight_scale,
                "mx4",
                self.intermediate_size * 2,  # N_1,
                self.hidden_size,  # K_1,
                self.hidden_size,  # N_2,
                self.intermediate_size,  # K_2,
                atom_config.tensor_parallel_size,
                act_quant=self.act_quant,
            )
            del layer.w13_weight
            del layer.w2_weight
            del layer.w13_weight_scale
            del layer.w2_weight_scale
            layer.w13_weight = w13_weight
            layer.w2_weight = w2_weight
            layer.w13_weight_scale = w13_scale
            layer.w2_weight_scale = w2_scale
            layer.w13_swizzle_layout = w13_swizzle_layout
            layer.w2_swizzle_layout = w2_swizzle_layout
            return

        # shuffle weight
        layer.w13_weight.data = shuffle_weight(
            layer.w13_weight,
            is_guinterleave=self.is_guinterleave,
            gate_up=True,
        )
        layer.w2_weight.data = shuffle_weight(
            layer.w2_weight,
            is_guinterleave=self.is_guinterleave,
            gate_up=False,
        )
        layer.w13_weight.is_shuffled = True
        layer.w2_weight.is_shuffled = True

        # shuffle scale
        w13_scale_2d = layer.w13_weight_scale.reshape(
            -1, layer.w13_weight_scale.shape[-1]
        )
        w2_scale_2d = layer.w2_weight_scale.reshape(-1, layer.w2_weight_scale.shape[-1])

        shuffled_w13_scale = moe_shuffle_scale(
            w13_scale_2d,
            self.num_experts,
            is_guinterleave=self.is_guinterleave,
            gate_up=True,
        )
        shuffled_w2_scale = moe_shuffle_scale(
            w2_scale_2d,
            self.num_experts,
            is_guinterleave=self.is_guinterleave,
            gate_up=False,
        )
        layer.w13_weight_scale = atom_parameter(shuffled_w13_scale)
        layer.w2_weight_scale = atom_parameter(shuffled_w2_scale)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:

        a1_scale = getattr(layer, "w13_input_scale", None)
        a2_scale = getattr(layer, "w2_input_scale", None)

        if self.act_quant == MoEActivationQuant.FP8:
            return mxfp4_w4a8_moe_quant_config(
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
            )
        return mxfp4_w4a16_moe_quant_config(
            w1_bias=layer.w13_bias,
            w2_bias=layer.w2_bias,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
        )

    @mark_trace(prefix="mxfp4_moe", torch_compile=False)
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
        fused_shared_experts_scoring_func: Optional[str] = None,
        activation: ActivationType = ActivationType.Silu,
    ) -> torch.Tensor:
        if self.use_triton:
            from atom.model_ops.fused_moe_triton import (
                triton_kernel_fused_experts,
                triton_kernel_moe_forward,
            )

            # Check if the model needs custom routing that triton routing()
            # does not support (grouped topk, sigmoid scoring, bias correction).
            needs_custom_routing = (
                use_grouped_topk
                or scoring_func != "softmax"
                or e_score_correction_bias is not None
                or custom_routing_function is not None
            )

            if needs_custom_routing:
                # custom routing -- set for deepseek routing n expts act, for grouped topk
                n_expts_act = top_k

                # custom routing
                from aiter.ops.triton.moe.moe_routing.routing import (  # grouped topk included
                    routing,
                )

                routing_data, gather_idx, scatter_idx = routing(
                    router_logits,
                    n_expts_act,
                    score_mode=scoring_func,
                    bias=(
                        e_score_correction_bias.to(torch.float32)
                        if e_score_correction_bias is not None
                        else None
                    ),
                    renorm=renormalize,
                    routed_scaling_factor=layer.routed_scaling_factor,
                    use_grouped_topk=use_grouped_topk,
                    num_expert_group=num_expert_group,
                    topk_group=topk_group,
                )
                # Routed-only gate count (no shared-expert widening).
                n_expts_act = routing_data.n_expts_act

                # Convert to triton routing data structures
                num_tokens, n_expts_tot = router_logits.shape

                if global_num_experts > 0:
                    n_expts_tot = global_num_experts

                output = torch.empty_like(x)
                _moe_result = triton_kernel_fused_experts(
                    output,
                    x,
                    layer.w13_weight,
                    layer.w2_weight,
                    routing_data,
                    gather_idx,
                    scatter_idx,
                    topk=n_expts_act,
                    activation=activation,
                    w13_scale=layer.w13_weight_scale,
                    w2_scale=layer.w2_weight_scale,
                    w13_swizzle_layout=layer.w13_swizzle_layout,
                    w2_swizzle_layout=layer.w2_swizzle_layout,
                    a13_scale=layer.w13_input_scale,
                    a2_scale=layer.w2_input_scale,
                    w1_bias=layer.w13_bias,
                    w2_bias=layer.w2_bias,
                    swiglu_limit=getattr(layer, "swiglu_limit", 0.0),
                    apply_router_weight_on_input=apply_router_weight_on_input,
                    global_num_experts=n_expts_tot,
                    expert_map=expert_map,
                    act_quant=self.act_quant,
                )

                # Always-on shared expert(s) via a standalone dense GEMM,
                # added to the routed output before the TP all-reduce.
                if layer.num_fused_shared_experts > 0:
                    _moe_result = _moe_result + self._apply_shared_experts_dense(
                        layer, x, activation
                    )
                return _moe_result

            assert (
                fused_shared_experts_scoring_func is None
            ), "triton kernel does not support fused shared experts func"

            # Takes directly from model dtype in config.json
            return triton_kernel_moe_forward(
                x,
                layer.w13_weight,
                layer.w2_weight,
                router_logits,
                topk=top_k,
                renormalize=renormalize,
                activation=activation,
                w13_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                w13_swizzle_layout=layer.w13_swizzle_layout,
                w2_swizzle_layout=layer.w2_swizzle_layout,
                a13_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                w1_bias=layer.w13_bias,
                w2_bias=layer.w2_bias,
                swiglu_limit=getattr(layer, "swiglu_limit", 7.0),
                expert_map=expert_map,
                apply_router_weight_on_input=apply_router_weight_on_input,
                global_num_experts=global_num_experts,
                act_quant=self.act_quant,
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
            num_routing_experts=global_num_experts,
            num_fused_shared_experts=layer.num_fused_shared_experts,
            fused_shared_experts_scoring_func=fused_shared_experts_scoring_func,
            routed_scaling_factor=layer.routed_scaling_factor,
        )
        a1_scale = getattr(layer, "w13_input_scale", None)
        a2_scale = getattr(layer, "w2_input_scale", None)
        moe_extra_args = {
            "gate_mode": (
                GateMode.INTERLEAVE.value
                if self.is_guinterleave
                else GateMode.SEPARATED.value
            ),
            "swiglu_limit": getattr(layer, "swiglu_limit", 0.0),
        }
        if self.fused_experts is None:
            return fused_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                expert_mask=layer.expert_mask,
                activation=activation,
                quant_type=self.quant_type,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                doweight_stage1=apply_router_weight_on_input,
                hidden_pad=self.hidden_pad,
                intermediate_pad=self.intermediate_pad,
                bias1=layer.w13_bias,
                bias2=layer.w2_bias,
                **moe_extra_args,
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
            expert_mask=layer.expert_mask,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            bias1=layer.w13_bias,
            bias2=layer.w2_bias,
            hidden_pad=self.hidden_pad,
            intermediate_pad=self.intermediate_pad,
            moe_extra_args=moe_extra_args,
        )

    def _apply_shared_experts_dense(self, layer, x, activation):
        """Standalone dense MXFP4 GEMM for the always-on shared expert(s).

        Functionally replaces fusing the shared expert into grouped_topk
        routing. The fused path appended each shared expert as an always-on
        routed slot with a fixed weight of ``SHARED_SCORE`` (== 1.0 here),
        placed *after* the routed renorm / ``routed_scaling_factor`` and run
        through the MoE GEMM as the last expert(s) of ``w13/w2``.

        Here we instead apply the shared expert to every token via two dense
        MXFP4 GEMM calls (gate_up -> SiLU-and-mul -> down) on the pre-swizzle
        weight slices stashed in ``process_weights_after_loading``. The dense
        GEMM is ``gemm_afp4wfp4`` (a4w4) when activations are MXFP4, otherwise
        ``gemm_a16wfp4`` (a16w4), matching the routed-expert activation dtype,
        and return the result so the caller adds it (weight 1.0) to the routed
        output before the TP all-reduce. The shared-expert intermediate is
        TP-partitioned exactly like the routed experts, so both partial outputs
        reduce together.
        """
        from aiter.ops.triton.fusions.fused_clamp_act_mul import fused_clamp_act_mul
        from aiter.ops.triton.gemm.basic.gemm_a16wfp4 import gemm_a16wfp4

        # The dense shared-expert GEMM only implements the SiLU activation
        # path; SwiGLU models have no fused shared experts, so this assert
        # documents the supported scope.
        assert (
            activation != ActivationType.Swiglu
        ), "dense shared-expert GEMM only supports the SiLU activation path"

        M = x.shape[0]
        swiglu_limit = getattr(layer, "swiglu_limit", 0.0)

        use_a4w4 = self.act_quant == MoEActivationQuant.FP4
        if use_a4w4:
            from aiter.ops.triton.gemm.basic.gemm_afp4wfp4 import gemm_afp4wfp4
            from aiter.ops.triton.moe.moe_op_gemm_a4w4 import mxfp4_quant

        def _shared_expert_gemm(act, weight, weight_scale):
            if use_a4w4:
                act_fp4, act_mx_scale = mxfp4_quant(act)
                return gemm_afp4wfp4(act_fp4, weight, act_mx_scale, weight_scale)
            return gemm_a16wfp4(act, weight, weight_scale)

        shared_w13_bias = getattr(layer, "shared_w13_bias", None)
        shared_w2_bias = getattr(layer, "shared_w2_bias", None)

        shared_out = None
        for e in range(layer.num_fused_shared_experts):
            gate_up = _shared_expert_gemm(
                x,
                layer.shared_w13_weight[e],
                layer.shared_w13_weight_scale[e],
            )
            if shared_w13_bias is not None:
                gate_up = gate_up + shared_w13_bias[e]
            half_n = gate_up.shape[-1] // 2
            intermediate = torch.empty((M, half_n), device=x.device, dtype=x.dtype)
            fused_clamp_act_mul(
                gate_up,
                out=intermediate,
                swiglu_limit=swiglu_limit,
                activation="silu",
                dtype_quant=None,
            )
            out_e = _shared_expert_gemm(
                intermediate,
                layer.shared_w2_weight[e],
                layer.shared_w2_weight_scale[e],
            )
            if shared_w2_bias is not None:
                out_e = out_e + shared_w2_bias[e]
            shared_out = out_e if shared_out is None else shared_out + out_e
        return shared_out


# Refer to CompressedTensorsW8A8Fp8MoEMethod in vllm
class CompressedTensorsFp8MoEMethod(FusedMoEMethodBase):
    def __init__(self, quant_config: LayerQuantConfig, moe: FusedMoEConfig):
        super().__init__(moe)
        self.quant_config = quant_config
        self.quant_type = quant_config.quant_type
        self.quant_dtype = quant_config.quant_dtype

        # Check if we need to normalize e4m3fn to e4m3fnuz (AMD GPUs)
        self.need_normalize_e4m3fn_to_e4m3fnuz = (
            self.quant_dtype == torch.float8_e4m3fnuz
        )

        # Determine if this is block quantization
        self.block_quant = self.quant_type in [
            QuantType.per_1x128,
            QuantType.per_1x32,
        ]

        # For compressed-tensors, check if per-channel quantization
        self.per_channel = self.quant_type == QuantType.per_Token

        # Check if static input scales (activation quantization)
        self.static_input_scales = not quant_config.is_dynamic

        # Block sizes for block quantization
        if self.block_quant:
            if self.quant_type == QuantType.per_1x128:
                self.block_n = 128
                self.block_k = 128
            elif self.quant_type == QuantType.per_1x32:
                self.block_n = 1
                self.block_k = 32

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        """Create weight parameters for compressed-tensors FP8 MoE."""
        layer.num_experts = num_experts
        layer.params_dtype = params_dtype
        layer.hidden_size = hidden_size
        layer.intermediate_size_per_partition = intermediate_size_per_partition

        # Check block alignment for block quantization
        if self.block_quant:
            tp_size = get_tp_group().world_size
            if intermediate_size_per_partition % self.block_n != 0:
                raise ValueError(
                    f"intermediate_size_per_partition={intermediate_size_per_partition} "
                    f"must be divisible by block_n={self.block_n}"
                )
            if tp_size > 1 and intermediate_size_per_partition % self.block_k != 0:
                raise ValueError(
                    f"intermediate_size_per_partition={intermediate_size_per_partition} "
                    f"must be divisible by block_k={self.block_k}"
                )

        # WEIGHTS
        w13_weight = atom_parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            )
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = atom_parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            )
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES - different shapes based on quantization strategy
        if self.per_channel:
            # Per-channel quantization: shape [E, N, 1]
            # This is the key difference for compressed-tensors
            w13_weight_scale = atom_parameter(
                torch.ones(
                    num_experts,
                    2 * intermediate_size_per_partition,
                    1,  # Important: dimension is 1, not omitted
                    dtype=torch.float32,
                )
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)

            w2_weight_scale = atom_parameter(
                torch.ones(
                    num_experts,
                    hidden_size,
                    1,  # Important: dimension is 1, not omitted
                    dtype=torch.float32,
                )
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)

            # Mark as per-channel quantization for weight loader
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
            )
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        elif self.block_quant:
            # Block quantization
            w13_weight_scale = atom_parameter(
                torch.ones(
                    num_experts,
                    2
                    * (
                        (intermediate_size_per_partition + self.block_n - 1)
                        // self.block_n
                    ),
                    (hidden_size + self.block_k - 1) // self.block_k,
                    dtype=torch.float32,
                )
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)

            w2_weight_scale = atom_parameter(
                torch.ones(
                    num_experts,
                    (hidden_size + self.block_n - 1) // self.block_n,
                    (intermediate_size_per_partition + self.block_k - 1)
                    // self.block_k,
                    dtype=torch.float32,
                )
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)

            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            )
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        else:
            # Per-tensor quantization: shape [E, 2] for w13, [E] for w2
            w13_weight_scale = atom_parameter(
                torch.ones(num_experts, 2, dtype=torch.float32)
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)

            w2_weight_scale = atom_parameter(
                torch.ones(num_experts, dtype=torch.float32)
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)

            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
            )
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES (activation scales)
        if self.static_input_scales:
            w13_input_scale = atom_parameter(
                torch.ones(num_experts, dtype=torch.float32)
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = atom_parameter(
                torch.ones(num_experts, dtype=torch.float32)
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Post-process weights after loading from checkpoint."""
        # Get references to weights and scales
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w13_input_scale = getattr(layer, "w13_input_scale", None)
        w2_input_scale = getattr(layer, "w2_input_scale", None)

        if self.need_normalize_e4m3fn_to_e4m3fnuz:
            (
                w13.data,
                w13_scale.data,
                w13_input_scale_data,
            ) = normalize_e4m3fn_to_e4m3fnuz(
                w13.data,
                w13_scale.data,
                w13_input_scale.data if w13_input_scale is not None else None,
            )
            if w13_input_scale is not None and w13_input_scale_data is not None:
                w13_input_scale.data = w13_input_scale_data

            w2.data, w2_scale.data, w2_input_scale_data = normalize_e4m3fn_to_e4m3fnuz(
                w2.data,
                w2_scale.data,
                w2_input_scale.data if w2_input_scale is not None else None,
            )
            if w2_input_scale is not None and w2_input_scale_data is not None:
                w2_input_scale.data = w2_input_scale_data

        # For per-tensor quantization, combine w1 and w3 scales
        # This is necessary for kernels that expect a single scale per expert
        if not self.per_channel and not self.block_quant:
            # w13_weight_scale has shape [E, 2] for w1 and w3
            # Use the max scale and requantize both w1 and w3 with it
            max_w13_scales = w13_scale.max(dim=1).values  # Shape: [E]
            num_experts = w13.shape[0]
            shard_size = layer.intermediate_size_per_partition

            # Requantize w1 and w3 with max scale per expert
            for expert_id in range(num_experts):
                max_scale = max_w13_scales[expert_id]

                # Process w1 (first shard)
                w1_scale = w13_scale[expert_id, 0]
                if w1_scale != max_scale:
                    # Dequantize: weight_fp32 = weight_fp8 * scale
                    w1_dq = per_tensor_dequantize(
                        w13[expert_id, :shard_size, :], w1_scale
                    )
                    # Quantize: weight_fp8 = weight_fp32 / scale
                    w1_q = (w1_dq / max_scale).clamp(
                        min=torch.finfo(w13.dtype).min,
                        max=torch.finfo(w13.dtype).max,
                    )
                    w13.data[expert_id, :shard_size, :] = w1_q.to(w13.dtype)

                # Process w3 (second shard)
                w3_scale = w13_scale[expert_id, 1]
                if w3_scale != max_scale:
                    # Dequantize: weight_fp32 = weight_fp8 * scale
                    w3_dq = per_tensor_dequantize(
                        w13[expert_id, shard_size:, :], w3_scale
                    )
                    # Quantize: weight_fp8 = weight_fp32 / scale
                    w3_q = (w3_dq / max_scale).clamp(
                        min=torch.finfo(w13.dtype).min,
                        max=torch.finfo(w13.dtype).max,
                    )
                    w13.data[expert_id, shard_size:, :] = w3_q.to(w13.dtype)

            # Update scale to single max scale per expert [E]
            layer.w13_weight_scale = atom_parameter(max_w13_scales)

        # Shuffle weights for asm moe (moved from inference to load time for better performance).
        if w13.dtype in [
            torch.int8,
            torch.uint8,
            torch.float8_e4m3fnuz,
            torch.float8_e4m3fn,
        ]:
            from aiter.ops.shuffle import shuffle_weight

            w13.data = shuffle_weight(w13.data)
            w2.data = shuffle_weight(w2.data)

        # Call parent class for any additional processing
        super().process_weights_after_loading(layer)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        """Get quantization config for compressed-tensors FP8."""
        from atom.model_ops.fused_moe.config import fp8_w8a8_moe_quant_config

        w1_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        a1_scale = getattr(layer, "w13_input_scale", None)
        a2_scale = getattr(layer, "w2_input_scale", None)

        # Determine block shape based on quantization type
        if self.block_quant:
            block_shape = [self.block_n, self.block_k]
        else:
            block_shape = None

        return fp8_w8a8_moe_quant_config(
            w1_scale=w1_scale,
            w2_scale=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            block_shape=block_shape,
        )

    @mark_trace(prefix="compressed_fp8_moe", torch_compile=False)
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
        fused_shared_experts_scoring_func: Optional[str] = None,
        activation: ActivationType = ActivationType.Silu,
    ) -> torch.Tensor:
        """Apply compressed-tensors FP8 MoE computation."""
        # Select top-k experts using router logits
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
            num_routing_experts=layer.global_num_experts,
            fused_shared_experts_scoring_func=fused_shared_experts_scoring_func,
            routed_scaling_factor=layer.routed_scaling_factor,
        )

        # Get activation scales (may be None for dynamic quantization)
        a1_scale = getattr(layer, "w13_input_scale", None)
        a2_scale = getattr(layer, "w2_input_scale", None)

        # Use modular kernel if available (for EP/DP setups).
        # Otherwise route through the standard AITER fused_moe path so
        # compressed-tensors FP8 shares the same tuned-kernel dispatch.
        if self.fused_experts is not None:
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
                expert_mask=layer.expert_mask,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                apply_router_weight_on_input=apply_router_weight_on_input,
                moe_extra_args={
                    "gate_mode": GateMode.SEPARATED.value,
                    "swiglu_limit": getattr(layer, "swiglu_limit", 0.0),
                },
            )
        else:
            return torch.ops.aiter.rocm_aiter_fused_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                expert_mask=layer.expert_mask,
                activation=activation.value,
                quant_type=self.quant_type.value,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=a1_scale,
                a2_scale=a2_scale,
                doweight_stage1=apply_router_weight_on_input,
                gate_mode=GateMode.SEPARATED.value,
                swiglu_limit=getattr(layer, "swiglu_limit", 0.0),
            )


class Fp8MoEMethod(FusedMoEMethodBase):
    """MoE method for FP8.
    Supports three quantization strategies:
      - per_Tensor:  per-tensor weight scale, static/dynamic activation scale
      - per_Token (PTPTC): per-channel weight scale, dynamic per-token activation
      - per_1x128 / per_1x32 (block): block-wise weight scale, dynamic activation

    Also supports loading quantized FP16/BF16 model checkpoints with dynamic
    activation scaling. The weight scaling factor will be initialized after
    the model weights are loaded.

    Args:
        quant_config: The quantization config (LayerQuantConfig).
    """

    def __init__(self, quant_config: LayerQuantConfig, moe: FusedMoEConfig):
        super().__init__(moe)
        self.quant_config = quant_config
        self.quant_type = quant_config.quant_type
        self.quant_dtype = quant_config.quant_dtype
        self.block_quant = (
            self.quant_type == QuantType.per_1x128
            or self.quant_type == QuantType.per_1x32
        )
        self.channel_quant = self.quant_type == QuantType.per_Token
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
        self.num_experts = num_experts
        intermediate_size_for_weight = intermediate_size_per_partition

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
            if self.quant_type == QuantType.per_1x32:
                # aiter's GU-interleaved MXFP8 scale shuffle packs 8 scale
                # columns, i.e. 256 weight columns for 1x32 scales. TP8 on
                # MiniMax-M3 has local intermediate=384, so pad to 512.
                scale_pack_k = block_k * 8
                intermediate_size_for_weight = (
                    (intermediate_size_per_partition + scale_pack_k - 1)
                    // scale_pack_k
                    * scale_pack_k
                )

        # WEIGHTS
        w13_weight = atom_parameter(
            torch.empty(
                num_experts,
                2 * intermediate_size_for_weight,
                hidden_size,
                dtype=params_dtype,
            )
        )
        layer.register_parameter("w13_weight", w13_weight)
        if self.quant_type == QuantType.per_1x32:
            w13_weight.data.view(torch.uint8).zero_()
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = atom_parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_for_weight,
                dtype=params_dtype,
            )
        )
        layer.register_parameter("w2_weight", w2_weight)
        if self.quant_type == QuantType.per_1x32:
            w2_weight.data.view(torch.uint8).zero_()
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if self.channel_quant:
            # Per-channel (PTPTC): one scale per output channel per expert.
            # w13: [E, 2*N], w2: [E, hidden_size]
            w13_weight_scale = atom_parameter(
                torch.ones(
                    num_experts,
                    2 * intermediate_size_for_weight,
                    dtype=torch.float32,
                )
            )
            w2_weight_scale = atom_parameter(
                torch.ones(num_experts, hidden_size, dtype=torch.float32)
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
        elif self.block_quant:
            scale_shape_w13 = (
                num_experts,
                2 * ((intermediate_size_for_weight + block_n - 1) // block_n),
                (hidden_size + block_k - 1) // block_k,
            )
            scale_shape_w2 = (
                num_experts,
                (hidden_size + block_n - 1) // block_n,
                (intermediate_size_for_weight + block_k - 1) // block_k,
            )
            # MXFP8 checkpoints store 1x32 scales as e8m0 bytes. Keep the
            # existing float32 initialization for the original 1x128 path.
            if self.quant_type == QuantType.per_1x32:
                w13_scale_data = torch.empty(scale_shape_w13, dtype=dtypes.fp8_e8m0)
                w2_scale_data = torch.empty(scale_shape_w2, dtype=dtypes.fp8_e8m0)
                w13_scale_data.view(torch.uint8).zero_()
                w2_scale_data.view(torch.uint8).zero_()
            else:
                w13_scale_data = torch.ones(scale_shape_w13, dtype=torch.float32)
                w2_scale_data = torch.ones(scale_shape_w2, dtype=torch.float32)
            w13_weight_scale = atom_parameter(w13_scale_data)
            w2_weight_scale = atom_parameter(w2_scale_data)
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            assert self.quant_config.is_dynamic
        else:
            # Per-tensor
            w13_weight_scale = atom_parameter(
                torch.ones(num_experts, 2, dtype=torch.float32)
            )
            w2_weight_scale = atom_parameter(
                torch.ones(num_experts, dtype=torch.float32)
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)

        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        # Per-channel uses dynamic per-token activation, no static input scales.
        if self.channel_quant or self.quant_config.is_dynamic:
            layer.w13_input_scale = None
            layer.w2_input_scale = None
        else:
            w13_input_scale = atom_parameter(
                torch.ones(num_experts, dtype=torch.float32)
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)
            w2_input_scale = atom_parameter(
                torch.ones(num_experts, dtype=torch.float32)
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)

    def _normalize_weights_and_scales(self, layer: nn.Module):
        if not self.need_normalize_e4m3fn_to_e4m3fnuz:
            return
        w13_weight, w13_weight_scale, w13_input_scale = normalize_e4m3fn_to_e4m3fnuz(
            layer.w13_weight, layer.w13_weight_scale, layer.w13_input_scale
        )
        w2_weight, w2_weight_scale, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz(
            layer.w2_weight, layer.w2_weight_scale, layer.w2_input_scale
        )
        layer.w13_weight = atom_parameter(w13_weight)
        layer.w13_weight_scale = atom_parameter(w13_weight_scale)
        layer.w2_weight = atom_parameter(w2_weight)
        layer.w2_weight_scale = atom_parameter(w2_weight_scale)
        if w13_input_scale is not None:
            layer.w13_input_scale = atom_parameter(w13_input_scale)
        if w2_input_scale is not None:
            layer.w2_input_scale = atom_parameter(w2_input_scale)

    def process_weights_after_loading(self, layer: nn.Module) -> None:
        if self.block_quant:
            self._process_block_quant(layer)
        elif self.channel_quant:
            self._process_channel_quant(layer)
        else:
            self._process_tensor_quant(layer)

    def _process_block_quant(self, layer: nn.Module) -> None:
        assert self.quant_config.is_dynamic
        self._normalize_weights_and_scales(layer)

        if not self.need_normalize_e4m3fn_to_e4m3fnuz:
            layer.w13_weight = atom_parameter(layer.w13_weight.data)
            layer.w13_weight_scale = atom_parameter(layer.w13_weight_scale.data)
            layer.w2_weight = atom_parameter(layer.w2_weight.data)
            layer.w2_weight_scale = atom_parameter(layer.w2_weight_scale.data)

        if self.quant_type == QuantType.per_1x32:
            # aiter's MXFP8 MoE kernels consume the same gate/up interleaved
            # layout used by their 1x32 shuffle helpers. Keep this branch
            # isolated so the existing 1x128 FP8 path still uses shuffle_weights.
            layer.w13_weight.data = shuffle_weight(
                layer.w13_weight,
                is_guinterleave=True,
                gate_up=True,
            )
            layer.w2_weight.data = shuffle_weight(
                layer.w2_weight,
                is_guinterleave=True,
                gate_up=False,
            )
            layer.w13_weight.is_shuffled = True
            layer.w2_weight.is_shuffled = True

            w13_scale_2d = layer.w13_weight_scale.reshape(
                -1, layer.w13_weight_scale.shape[-1]
            )
            w2_scale_2d = layer.w2_weight_scale.reshape(
                -1, layer.w2_weight_scale.shape[-1]
            )
            layer.w13_weight_scale = atom_parameter(
                moe_shuffle_scale(
                    w13_scale_2d,
                    self.num_experts,
                    is_guinterleave=True,
                    gate_up=True,
                )
            )
            layer.w2_weight_scale = atom_parameter(
                moe_shuffle_scale(
                    w2_scale_2d,
                    self.num_experts,
                    is_guinterleave=True,
                    gate_up=False,
                )
            )
            return

        shuffle_weights(layer.w13_weight, layer.w2_weight)

    def _process_channel_quant(self, layer: nn.Module) -> None:
        """PTPTC"""
        self._normalize_weights_and_scales(layer)

        if layer.w13_weight.data.dtype in (torch.bfloat16, torch.float16):
            quant_func = get_hip_quant(QuantType.per_Token)
            for expert_id in range(layer.local_num_experts):
                w13_q, w13_s = quant_func(
                    layer.w13_weight.data[expert_id], quant_dtype=dtypes.fp8
                )
                layer.w13_weight.data[expert_id] = w13_q
                layer.w13_weight_scale.data[expert_id] = w13_s.squeeze(-1)

                w2_q, w2_s = quant_func(
                    layer.w2_weight.data[expert_id], quant_dtype=dtypes.fp8
                )
                layer.w2_weight.data[expert_id] = w2_q
                layer.w2_weight_scale.data[expert_id] = w2_s.squeeze(-1)

        shuffle_weights(layer.w13_weight, layer.w2_weight)

    def _process_tensor_quant(self, layer: nn.Module) -> None:
        if not self.quant_config.is_dynamic:
            if layer.w13_input_scale is None or layer.w2_input_scale is None:
                raise ValueError(
                    "QuantConfig has static quantization, but found "
                    "activation scales are None."
                )
            layer.w13_input_scale = atom_parameter(layer.w13_input_scale.max())
            layer.w2_input_scale = atom_parameter(layer.w2_input_scale.max())

        self._normalize_weights_and_scales(layer)

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
                (
                    layer.w13_weight[expert_id][start : start + shard_size, :],
                    _,
                ) = quant_func(dq_weight, max_w13_scales[expert_id])
                start += shard_size

        shuffle_weights(layer.w13_weight, layer.w2_weight)

        layer.w13_weight_scale = atom_parameter(max_w13_scales)

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        if self.channel_quant:
            return fp8_w8a8_moe_quant_config(
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                per_act_token_quant=True,
            )
        else:
            # block_quant (per_1x128 / per_1x32) MUST hand the kernel its block
            # shape — otherwise fp8_w8a8 broadcasts the [N/128, K/128] scale as
            # if it were per-tensor / per-channel and produces garbage.
            # V4-Flash-Base on gfx942 hits this: routed experts are FP8 e4m3
            # per_1x128 + UE8M0 block-scale.
            if self.block_quant:
                if self.quant_type == QuantType.per_1x128:
                    block_shape = [128, 128]
                elif self.quant_type == QuantType.per_1x32:
                    block_shape = [1, 32]
                else:
                    block_shape = None
            else:
                block_shape = None
            return fp8_w8a8_moe_quant_config(
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                block_shape=block_shape,
            )

    @mark_trace(prefix="fp8_moe", torch_compile=False)
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
        fused_shared_experts_scoring_func: Optional[str] = None,
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
            fused_shared_experts_scoring_func=fused_shared_experts_scoring_func,
            num_routing_experts=global_num_experts,
            num_fused_shared_experts=layer.num_fused_shared_experts,
            routed_scaling_factor=layer.routed_scaling_factor,
        )
        # Match the 1x32 preshuffled layout above; other FP8 quant modes keep
        # the historical separated gate/up layout.
        gate_mode = (
            GateMode.INTERLEAVE.value
            if self.quant_type == QuantType.per_1x32
            else GateMode.SEPARATED.value
        )
        moe_extra_args = {
            "gate_mode": gate_mode,
            "swiglu_limit": getattr(layer, "swiglu_limit", 0.0),
        }
        if self.quant_type == QuantType.per_Tensor or self.fused_experts is None:
            return torch.ops.aiter.rocm_aiter_fused_moe(
                x,
                layer.w13_weight,
                layer.w2_weight,
                topk_weights,
                topk_ids,
                expert_mask=layer.expert_mask,
                activation=activation.value,
                quant_type=self.quant_type.value,
                w1_scale=layer.w13_weight_scale,
                w2_scale=layer.w2_weight_scale,
                a1_scale=layer.w13_input_scale,
                a2_scale=layer.w2_input_scale,
                doweight_stage1=apply_router_weight_on_input,
                **moe_extra_args,
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
            expert_mask=layer.expert_mask,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            apply_router_weight_on_input=apply_router_weight_on_input,
            moe_extra_args=moe_extra_args,
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


@FusedMoEDecoratorForPluginMode
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
        has_bias: bool = False,
        activation: ActivationType = ActivationType.Silu,
        shared_expert_scoring_func: Optional[str] = None,
        config: Optional[PretrainedConfig] = None,
        shared_expert_prefix: Optional[str] = None,
    ):
        super().__init__()
        self.prefix = prefix
        layer_quant_config = (
            quant_config.get_layer_quant_config(prefix, check_children=True)
            if quant_config
            else None
        )
        self.params_dtype = (
            layer_quant_config.quant_dtype
            if layer_quant_config
            else torch.get_default_dtype()
        )
        self.layer_quant_config = layer_quant_config
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
        self.global_num_experts = num_experts
        self.register_buffer("expert_map", None, persistent=False)
        self.register_buffer("expert_mask", None, persistent=False)
        if self.use_ep:
            self.local_num_experts, self.expert_map = determine_expert_map(
                ep_size=self.ep_size,
                ep_rank=self.ep_rank,
                global_num_experts=self.global_num_experts,
            )
        else:
            self.local_num_experts = self.global_num_experts
        self.top_k = top_k
        self.global_num_experts = num_experts
        self.shared_expert_scoring_func = shared_expert_scoring_func

        if shared_expert_prefix is None and prefix.endswith(".experts"):
            shared_expert_prefix = prefix[: -len(".experts")] + ".shared_experts"

        fuse_shared_experts = (
            is_rocm_aiter_fusion_shared_expert_enabled_for_quant_config(
                quant_config,
                shared_expert_prefix=shared_expert_prefix,
                routed_expert_prefix=prefix,
            )
        )
        self.num_fused_shared_experts = (
            config.n_shared_experts
            if config is not None
            and hasattr(config, "n_shared_experts")
            and fuse_shared_experts
            else 0
        )
        self.routed_scaling_factor = (
            getattr(config, "routed_scaling_factor", 1.0)
            if config is not None and atom_config.torch_dtype != torch.float16
            else 1.0
        )
        if self.use_ep:
            expert_mask = torch.ones(
                (self.global_num_experts + self.num_fused_shared_experts + 1,),
                dtype=torch.int32,
                device=self.expert_map.device,
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
                    # Sentinel entry for the fake expert ID
                    # (global_num_experts + num_fused_shared_experts) used by
                    # aiter topK to mark non-local tokens when EP is active.
                    # Must map to -1 so that EP remapping zeros their weights.
                    torch.tensor([-1], dtype=torch.int32),
                ),
                dim=0,
            )
        if fuse_shared_experts and self.num_fused_shared_experts > 0:
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
        if fuse_shared_experts:
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

        try:
            a_quant_dtype = (
                config.quantization_config.get("global_quant_config", "")
                .get("input_tensors", "")
                .get("dtype", "")
            )
        except AttributeError:
            # global quant config does not exist, no activation loaded
            a_quant_dtype = None

        moe = FusedMoEConfig(
            num_experts=self.global_num_experts,
            experts_per_token=self.top_k,
            hidden_dim=hidden_size,
            num_local_experts=self.local_num_experts,
            moe_parallel_config=self.moe_parallel_config,
            in_dtype=atom_config.torch_dtype,
            a_quant_dtype=a_quant_dtype,
            max_num_tokens=atom_config.max_num_batched_tokens,
            has_bias=self.has_bias,
            # is_act_and_mul=True,
            is_lora_enabled=False,
        )
        self.moe_config = moe
        self.quant_config = quant_config
        self.online_quant = quant_config is not None and quant_config.online_quant

        quant_method_str = (
            layer_quant_config.quant_method if layer_quant_config else None
        )
        if layer_quant_config is None or layer_quant_config.quant_type == QuantType.No:
            self.quant_method: Optional[QuantizeMethodBase] = UnquantizedFusedMoEMethod(
                moe
            )
        elif (
            quant_method_str == "compressed-tensors"
            and layer_quant_config.quant_dtype == dtypes.fp8
        ):
            # Use CompressedTensorsFp8MoEMethod for compressed-tensors format
            self.quant_method = CompressedTensorsFp8MoEMethod(layer_quant_config, moe)
        elif layer_quant_config.quant_dtype == dtypes.fp8:
            self.quant_method = Fp8MoEMethod(layer_quant_config, moe)
        elif layer_quant_config.quant_dtype == dtypes.fp4x2:
            self.quant_method = Mxfp4MoEMethod(layer_quant_config, moe)
        else:
            raise ValueError(
                f"Unsupported quant dtype: {layer_quant_config.quant_dtype}"
            )

        assert self.quant_method is not None

        self.apply_router_weight_on_input = apply_router_weight_on_input
        self.moe_quant_params = {
            "num_experts": self.local_num_experts,
            "hidden_size": hidden_size,
            "intermediate_size_per_partition": self.intermediate_size_per_partition,
            "params_dtype": self.params_dtype,
            "weight_loader": self.weight_loader,
        }
        self.quant_method.create_weights(layer=self, **self.moe_quant_params)
        compilation_config = atom_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError("Duplicate layer name: {}".format(prefix))
        compilation_config.static_forward_context[prefix] = self
        self.layer_name = prefix

    def process_weights_after_loading(self):
        self._online_quant()

    def _online_quant(self):
        """Handle online quantization: (optionally dequant →) quantize weights,
        then switch quant_method.

        Called by the loader BEFORE quant_method.process_weights_after_loading().
        Flow:
          1. If source is already quantized (e.g. per_1x128 FP8), dequant → bf16
          2. Switch quant_method and allocate target quantized buffers
          3. Per-expert: quantize bf16 → write into buffers via
             _load_model_weight_or_group_weight_scale (reuses TP-shard + padding)
          4. Loader then calls target method's process_weights_after_loading
             which does fn→fnuz normalization and shuffle on the already-FP8 weights.
        """
        if not self.online_quant:
            return

        online_quant_config = self.quant_config.get_layer_quant_config(
            self.layer_name, use_online_quant=True
        )
        online_quant_type = online_quant_config.quant_type
        online_quant_dtype = online_quant_config.quant_dtype
        source_quant_type = self.layer_quant_config.quant_type
        if should_skip_online_quant(
            source_quant_type, self.params_dtype, online_quant_config
        ):
            return

        assert source_quant_type in (
            QuantType.No,
            QuantType.per_1x128,
            QuantType.per_1x32,
        ), (
            f"Unsupported source quant_type for MoE online quantization: "
            f"{source_quant_type} (layer={self.layer_name})"
        )
        need_dequant = source_quant_type in (
            QuantType.per_1x128,
            QuantType.per_1x32,
        )

        def _dequant_func(w: torch.Tensor, sc: torch.Tensor) -> torch.Tensor:
            # per_1x128 -> deepseek-style square-block FP8; per_1x32 -> MXFP8.
            if source_quant_type == QuantType.per_1x32:
                return weight_dequant_mxfp8(w.contiguous(), sc.contiguous())
            return weight_dequant_fp8(w.contiguous(), sc.contiguous())

        # Online weight quant dispatch (MXFP4 vs FP8), shared with the Linear
        # path via a single helper under quark so both stay in sync.
        def _quant_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return quant_weight_online(
                w,
                online_quant_type=online_quant_type,
                online_quant_dtype=online_quant_dtype,
            )

        # Determine whether each weight needs all_gather to match offline quantization.
        # w13 (column parallel): (E, (2*intermediate/tp, hidden)) — TP dim 0
        # w2  (row parallel):    (E, (hidden, intermediate/tp)) — TP dim 1
        # w13 [e, m, n]->[e, m//tp, n//2]->[e, m//tp, n//32]
        def check_need_allgather():
            if self.use_ep:
                assert self.tp_size == 1, "EP MoE should not TP-shard expert weights"
                return False

            need_gather_w2 = False
            if self.tp_size > 1:
                # self.intermediate_size_per_partition = intermediate_size // self.tp_size
                w2_in = self.intermediate_size_per_partition
                if online_quant_type == QuantType.per_Token:
                    need_gather_w2 = True
                elif online_quant_type == QuantType.per_1x32:
                    need_gather_w2 = w2_in % 32 != 0
            return need_gather_w2

        need_gather_w2 = check_need_allgather()
        tp_group = get_tp_group() if need_gather_w2 else None
        load_full_w2 = not need_gather_w2

        # Save references to old weights before create_weights overwrites them.
        # For per_1x128 source we also need the old scales for dequantization.
        old_w13_data = self.w13_weight.data
        old_w2_data = self.w2_weight.data
        old_w13_scale = self.w13_weight_scale.data if need_dequant else None
        old_w2_scale = self.w2_weight_scale.data if need_dequant else None
        device = old_w13_data.device

        # Switch quant_method and allocate target quantized-type buffers.
        if online_quant_dtype == dtypes.fp8:
            self.quant_method = Fp8MoEMethod(online_quant_config, self.moe_config)
        elif online_quant_dtype == dtypes.fp4x2:
            self.quant_method = Mxfp4MoEMethod(online_quant_config, self.moe_config)
        else:
            raise ValueError(
                f"Unsupported online quant_dtype for MoE: {online_quant_dtype}"
            )
        self.moe_quant_params["params_dtype"] = online_quant_dtype
        with torch.device(device):
            self.quant_method.create_weights(layer=self, **self.moe_quant_params)

        self.w13_input_scale = None
        self.w2_input_scale = None

        for expert_id in range(self.local_num_experts):
            # --- w13 column-parallel ---
            w13_local = old_w13_data[expert_id]
            w1_size = w13_local.shape[0] // 2

            if need_dequant:
                w13_scale = old_w13_scale[expert_id]
                s1_size = w13_scale.shape[0] // 2
                w1_bf16 = _dequant_func(
                    w13_local[:w1_size],
                    w13_scale[:s1_size],
                )
                w3_bf16 = _dequant_func(
                    w13_local[w1_size:],
                    w13_scale[s1_size:],
                )
            else:
                w1_bf16 = w13_local[:w1_size]
                w3_bf16 = w13_local[w1_size:]

            w1_q, w1_s = _quant_weight(w1_bf16)
            w3_q, w3_s = _quant_weight(w3_bf16)
            del w1_bf16, w3_bf16

            w13_expert = self.w13_weight.data[expert_id]
            w13_scale_expert = self.w13_weight_scale.data[expert_id]
            for shard_id, wq, ws in (("w1", w1_q, w1_s), ("w3", w3_q, w3_s)):
                self._load_model_weight_or_group_weight_scale(
                    shard_dim=0,
                    expert_data=w13_expert,
                    shard_id=shard_id,
                    loaded_weight=wq,
                    tp_rank=self.tp_rank,
                    load_full=True,
                )
                self._load_quant_weight_scale(
                    expert_data=w13_scale_expert,
                    shard_dim=0,
                    shard_id=shard_id,
                    loaded_weight=ws,
                    tp_rank=self.tp_rank,
                    quant_type=online_quant_type,
                    load_full=True,
                )
            del w1_q, w3_q, w1_s, w3_s

            # w2 row-parallel: optionally gather before quantization
            # w2 mxfp4    [e, m, n]->[e, m, n//2//tp]->[e, m, n//32//tp]
            # w2 ptpc_fp8 [e, m, n]->[e, m, n//tp]->[e, m, 1]
            w2_local = old_w2_data[expert_id]
            if need_dequant:
                w2_local = _dequant_func(
                    w2_local,
                    old_w2_scale[expert_id],
                )
            if need_gather_w2:
                w2_full = tp_group.all_gather(w2_local, dim=1)
                w2_q, w2_s = _quant_weight(w2_full)
                del w2_full
            else:
                w2_q, w2_s = _quant_weight(w2_local)

            self._load_model_weight_or_group_weight_scale(
                shard_dim=1,
                expert_data=self.w2_weight.data[expert_id],
                shard_id="w2",
                loaded_weight=w2_q,
                tp_rank=self.tp_rank,
                load_full=load_full_w2,
            )
            # per_Token scale is along output dim (not TP-split), never needs shard
            w2_scale_load_full = (
                False if online_quant_type == QuantType.per_Token else load_full_w2
            )
            self._load_quant_weight_scale(
                expert_data=self.w2_weight_scale.data[expert_id],
                shard_dim=1,
                shard_id="w2",
                loaded_weight=w2_s,
                tp_rank=self.tp_rank,
                quant_type=online_quant_type,
                load_full=w2_scale_load_full,
            )
            del w2_q, w2_s

        del old_w13_data, old_w2_data
        if need_dequant:
            del old_w13_scale, old_w2_scale

        self._online_quant_info = {
            "layer": self.layer_name,
            "quant_type": online_quant_type.name,
            "quant_dtype": str(online_quant_dtype),
        }

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
        load_full: bool = False,
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
                load_full=load_full,
            )
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
                load_full=load_full,
            )

    def _load_quant_weight_scale(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        quant_type,
        load_full: bool = False,
    ):
        """Dispatch weight-scale loading by quant_type."""
        if quant_type == QuantType.per_Token:
            self._load_per_channel_weight_scale(
                expert_data=expert_data,
                shard_dim=shard_dim,
                shard_id=shard_id,
                loaded_weight=loaded_weight.squeeze(-1),
                tp_rank=tp_rank,
                load_full=load_full,
            )
        else:
            # The Aiter FP4 quantization function returns a value of type FP4*2
            self._load_model_weight_or_group_weight_scale(
                shard_dim=shard_dim,
                expert_data=expert_data,
                shard_id=shard_id,
                loaded_weight=loaded_weight.view(torch.uint8),
                tp_rank=tp_rank,
                load_full=load_full,
            )

    def _load_per_channel_weight_scale(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full: bool = False,
    ):
        # for per channel weight quantization
        if load_full:
            if shard_id == "w2":
                load_size = loaded_weight.shape[shard_dim]
                if load_size != expert_data.shape[shard_dim]:
                    expert_data = expert_data.narrow(shard_dim, 0, load_size)
                self._copy_quant_storage(expert_data, loaded_weight)
            elif shard_id in ("w1", "w3"):
                self._load_w13(
                    shard_id=shard_id,
                    shard_dim=shard_dim,
                    loaded_weight=loaded_weight,
                    expert_data=expert_data,
                    tp_rank=tp_rank,
                    load_full=True,
                )
            return
        if shard_id == "w2":
            self._copy_quant_storage(expert_data, loaded_weight)
        elif shard_id in ("w1", "w3"):
            self._load_w13(
                shard_id=shard_id,
                shard_dim=shard_dim,
                loaded_weight=loaded_weight,
                expert_data=expert_data,
                tp_rank=tp_rank,
            )

    @staticmethod
    def _copy_quant_storage(dst: torch.Tensor, src: torch.Tensor) -> None:
        """Copy quantized tensors without numeric conversion of byte formats."""
        if dst.dtype == dtypes.fp4x2:
            dst.view(torch.uint8).copy_(src.view(torch.uint8))
            return
        fp8_storage_dtypes = (
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e8m0fnu,
            dtypes.fp8,
            dtypes.fp8_e8m0,
        )
        if dst.dtype in fp8_storage_dtypes and src.dtype in fp8_storage_dtypes:
            # Offline FP8 checkpoints encode raw FP8 bytes. Avoid dtype-to-dtype
            # numeric conversion when destination storage uses a different FP8
            # variant; later scale fixups expect the original bytes.
            dst.view(torch.uint8).copy_(src.view(torch.uint8))
            return
        if dst.dtype == dtypes.fp8_e8m0 and src.dtype == torch.uint8:
            # e8m0 microscale tensors are byte-encoded; copy_ would convert the
            # uint8 values numerically instead of preserving the scale bits.
            dst.view(torch.uint8).copy_(src)
            return
        if dst.dtype == torch.uint8 and src.dtype in (
            torch.float8_e8m0fnu,
            torch.float8_e4m3fn,
        ):
            src = src.view(torch.uint8)
        dst.copy_(src)

    def _load_w13(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        shard_id: str,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full: bool = False,
    ):
        # for online local quantizaiton
        if load_full:
            expert_shard_size = expert_data.shape[shard_dim] // 2
            if shard_id == "w1":
                expert_data = expert_data.narrow(shard_dim, 0, expert_shard_size)
            else:
                assert shard_id == "w3"
                expert_data = expert_data.narrow(
                    shard_dim, expert_shard_size, expert_shard_size
                )
            load_size = loaded_weight.shape[shard_dim]
            if load_size != expert_shard_size:
                expert_data = expert_data.narrow(shard_dim, 0, load_size)
            self._copy_quant_storage(expert_data, loaded_weight)
            return

        # Index the loaded weight for tp sharding.
        # gate_up_proj: "MergedColumnParallel", so tp sharding on output_dim
        expert_shard_size = expert_data.shape[shard_dim] // 2
        # Derive shard size from loaded_weight (unpadded checkpoint) to avoid
        # out-of-bounds when expert_data is padded (e.g. MXFP4 alignment).
        load_shard_size = loaded_weight.shape[shard_dim] // self.tp_size
        loaded_weight = loaded_weight.narrow(
            shard_dim, load_shard_size * tp_rank, load_shard_size
        )
        # Narrow parameter and load.
        # w1, gate_proj: Load into first logical weight of w13.
        if shard_id == "w1":
            expert_data = expert_data.narrow(shard_dim, 0, expert_shard_size)
        # w3, up_proj: Load into second logical weight of w13.
        else:
            assert shard_id == "w3"
            expert_data = expert_data.narrow(
                shard_dim, expert_shard_size, expert_shard_size
            )
        # When expert_data is padded beyond the actual weight size, narrow to
        # the loaded weight size so the copy shape matches.
        if load_shard_size != expert_shard_size:
            expert_data = expert_data.narrow(shard_dim, 0, load_shard_size)
        self._copy_quant_storage(expert_data, loaded_weight)

    def _load_w2(
        self,
        expert_data: torch.Tensor,
        shard_dim: int,
        loaded_weight: torch.Tensor,
        tp_rank: int,
        load_full: bool = False,
    ):
        # # for online local quantizaiton
        if load_full:
            shard_size = expert_data.shape[shard_dim]
            load_size = loaded_weight.shape[shard_dim]
            if load_size != shard_size:
                expert_data = expert_data.narrow(shard_dim, 0, load_size)
            self._copy_quant_storage(expert_data, loaded_weight)
            return

        # Index the loaded weight for tp sharding.
        # down_proj: "RowParallel" so tp sharding on input_dim
        # Narrow parameter and load.
        shard_size = expert_data.shape[shard_dim]
        load_shard_size = loaded_weight.shape[shard_dim] // self.tp_size
        loaded_weight = loaded_weight.narrow(
            shard_dim, load_shard_size * tp_rank, load_shard_size
        )
        if load_shard_size != shard_size:
            expert_data = expert_data.narrow(shard_dim, 0, load_shard_size)
        # w2, down_proj: Load into only logical weight of w2.
        self._copy_quant_storage(expert_data, loaded_weight)

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
        expert_id: Optional[int] = None,
    ):
        target_param = param
        # single_expert means gate_up_proj.shape=[2880*2, 1440] from quark
        maybe_single_expert_input = loaded_weight.dim() == param.dim() - 1
        if expert_id is not None and maybe_single_expert_input:
            local_expert_id = self._map_global_expert_id_to_local_expert_id(expert_id)
            if local_expert_id == -1:
                return
            # Support loading a split/single expert tensor while reusing the
            # original merged loading logic.
            if loaded_weight.dim() == param.dim() - 1:
                loaded_weight = loaded_weight.unsqueeze(0)
                target_param = param[local_expert_id : local_expert_id + 1]
        # (FIXME) for gpt-oss all experts are combined
        mxfp4_block = 32
        ep_rank_start = self.ep_rank * self.local_num_experts
        ep_rank_end = ep_rank_start + self.local_num_experts
        tp_rank_start = self.tp_rank * self.intermediate_size_per_partition
        tp_rank_end = tp_rank_start + self.intermediate_size_per_partition
        if param is getattr(self, "w13_bias", None):
            if self.use_ep:
                if loaded_weight.shape[0] == target_param.shape[0]:
                    narrow_weight = loaded_weight
                else:
                    narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = loaded_weight[:, 2 * tp_rank_start : 2 * tp_rank_end]
            dim1 = narrow_weight.shape[1]
            target_param[:, :dim1].copy_(narrow_weight)
        elif param is getattr(self, "w2_bias", None):
            if self.use_ep:
                if loaded_weight.shape[0] == target_param.shape[0]:
                    narrow_weight = loaded_weight
                else:
                    narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = loaded_weight
                if self.tp_rank != 0:
                    narrow_weight.zero_()
            dim1 = narrow_weight.shape[1]
            target_param[:, :dim1].copy_(narrow_weight)
        elif param is getattr(self, "w13_weight", None):
            loaded_weight = loaded_weight.view(*loaded_weight.shape[:2], -1)
            if self.use_ep:
                if loaded_weight.shape[0] == target_param.shape[0]:
                    narrow_weight = loaded_weight
                else:
                    narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = loaded_weight[
                    :, 2 * tp_rank_start : 2 * tp_rank_end, ...
                ]
            dim1, dim2 = narrow_weight.shape[1:]
            target_param.view(torch.uint8)[:, :dim1, :dim2].copy_(
                narrow_weight.view(torch.uint8)
            )
        elif param is getattr(self, "w2_weight", None):
            loaded_weight = loaded_weight.view(*loaded_weight.shape[:2], -1)
            if self.use_ep:
                if loaded_weight.shape[0] == target_param.shape[0]:
                    narrow_weight = loaded_weight
                else:
                    narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = loaded_weight[
                    ..., tp_rank_start // 2 : tp_rank_end // 2
                ]
            dim1, dim2 = narrow_weight.shape[1:]
            target_param.view(torch.uint8)[:, :dim1, :dim2].copy_(
                narrow_weight.view(torch.uint8)
            )
        elif param is getattr(self, "w13_weight_scale", None):
            if self.use_ep:
                if loaded_weight.shape[0] == target_param.shape[0]:
                    narrow_weight = loaded_weight
                else:
                    narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = loaded_weight[
                    :, 2 * tp_rank_start : 2 * tp_rank_end, ...
                ]
            dim1, dim2 = narrow_weight.shape[1:]
            target_param[:, :dim1, :dim2].copy_(narrow_weight)
        elif param is getattr(self, "w2_weight_scale", None):
            if self.use_ep:
                if loaded_weight.shape[0] == target_param.shape[0]:
                    narrow_weight = loaded_weight
                else:
                    narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = loaded_weight[
                    ..., tp_rank_start // mxfp4_block : tp_rank_end // mxfp4_block
                ]
            dim1, dim2 = narrow_weight.shape[1:]
            target_param[:, :dim1, :dim2].copy_(narrow_weight)
        elif param is getattr(self, "w13_input_scale", None) or param is getattr(
            self, "w2_input_scale", None
        ):
            # input_scale is scalar per expert.
            if loaded_weight.dim() == 0:
                loaded_weight = loaded_weight.unsqueeze(0)
            if self.use_ep and loaded_weight.shape[0] != target_param.shape[0]:
                narrow_weight = loaded_weight[ep_rank_start:ep_rank_end, ...]
            else:
                narrow_weight = loaded_weight
            target_param[: narrow_weight.shape[0]].copy_(narrow_weight)

    def weight_loader(
        self,
        param: torch.nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str = "",
        shard_id: str = "",
        expert_id: int = 0,
    ) -> None:
        if self.layer_quant_config.quant_dtype == dtypes.fp4x2 and weight_name == "":
            self.mxf4_merged_weight_loader(param, loaded_weight, expert_id)
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
            quant_method = self.layer_quant_config.quant_type
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
                    load_full=getattr(param, "load_full_w2", False),
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
        num_routing_experts: int = 0,
        num_fused_shared_experts: int = 0,
        fused_shared_experts_scoring_func: Optional[str] = None,
        routed_scaling_factor: float = 1.0,
    ):

        # custom_routing_function takes precedence (e.g. DeepSeek-V4 hash routing
        # in the first 3 layers, where topk_ids are looked up from a per-token
        # hash table instead of computed from gate logits).
        if custom_routing_function is not None:
            topk_weights, topk_ids = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
            )
            return topk_weights, topk_ids

        # DeekSeekv2 uses grouped_top_k
        if use_grouped_topk:
            assert topk_group is not None
            assert num_expert_group is not None
            assert fused_shared_experts_scoring_func is None
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
            if scoring_func == "softmax":
                topk_weights, topk_ids = fused_topk(
                    gating_output=router_logits,
                    topk=top_k,
                    renormalize=renormalize,
                    num_fused_shared_experts=num_fused_shared_experts,
                    num_routing_experts=num_routing_experts,
                    fused_shared_experts_scoring_func=fused_shared_experts_scoring_func,
                )
            elif scoring_func in ("sigmoid", "sqrtsoftplus"):
                tokens_num = router_logits.shape[0]
                fuse_shared = num_fused_shared_experts > 0
                if fuse_shared:
                    import atom.model_ops.topK as _topK_mod

                    assert _topK_mod.aiter_topK_meta_data is not None, (
                        "AITER topK meta data is not initialized. "
                        "Please ensure that init_aiter_topK_meta_data is called "
                        "before this function."
                    )
                    total_topk_weights, total_topk_ids = _topK_mod.aiter_topK_meta_data
                    assert total_topk_weights.shape[0] >= tokens_num
                    n_extra = total_topk_ids.shape[1] - top_k
                    topk_ids, _ = torch.split(
                        total_topk_ids[:tokens_num],
                        [top_k, n_extra],
                        dim=1,
                    )
                    topk_weights, _ = torch.split(
                        total_topk_weights[:tokens_num],
                        [top_k, n_extra],
                        dim=1,
                    )
                else:
                    topk_ids = torch.empty(
                        tokens_num,
                        top_k,
                        dtype=torch.int32,
                        device=router_logits.device,
                    )
                    topk_weights = torch.empty(
                        tokens_num,
                        top_k,
                        dtype=torch.float32,
                        device=router_logits.device,
                    )

                # MiniMax-M3 applies the routed scale outside MoE when shared
                # experts are not fused; DeepSeek-V4 folds it into routing.
                route_scale = (
                    routed_scaling_factor
                    if fuse_shared or scoring_func == "sqrtsoftplus"
                    else 1.0
                )
                topk_gating(
                    topk_weights,
                    topk_ids,
                    router_logits,
                    e_score_correction_bias,
                    renormalize,
                    route_scale,
                    score_func=scoring_func,
                )
                if fuse_shared:
                    # Switch from the routed view back to the full buffer
                    # (routed + shared cols) for the fused MoE kernel.
                    topk_weights = total_topk_weights[:tokens_num]
                    topk_ids = total_topk_ids[:tokens_num]
            else:
                raise ValueError(
                    f"Unsupported scoring function for non-grouped topk: {scoring_func}"
                )

        return topk_weights, topk_ids

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        return torch.ops.aiter.moe_forward(
            hidden_states, router_logits, self.layer_name
        )

    def forward_impl_graph(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ):
        # There are three mode
        # 1. Pure DP mode: only DP is used
        # 2. DP attention + EP mori Moe
        # 3. DP attention + TP All_gahter/reduce Moe
        original_hidden_size = None
        sizes = None
        # Use all_gather/reduce_scatter when DP > 1 but not using mori all2all kernels
        use_dp_gather_scatter = (
            self.dp_size > 1
            and not self.moe_parallel_config.use_all2all_kernels
            and get_current_atom_config().enable_dp_attention
        )
        if use_dp_gather_scatter:
            ctx = get_forward_context()
            dp_group = get_dp_group()
            dp_eager_mode = not ctx.context.dp_uniform_decode

            from atom.utils.tbo.ubatching import tbo_active

            _tbo = tbo_active()
            if _tbo:
                from atom.utils.tbo.ubatching import (
                    tbo_switch_to_compute_sync,
                    tbo_yield_and_switch_from_compute_to_comm,
                )

                tbo_yield_and_switch_from_compute_to_comm()

            (
                hidden_states,
                router_logits,
                original_hidden_size,
                sizes,
            ) = dp_gather_hidden_and_router(
                hidden_states, router_logits, dp_eager_mode, ctx, dp_group
            )

            if _tbo:
                tbo_switch_to_compute_sync()

        # Matrix multiply.
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            fused_shared_experts_scoring_func=self.shared_expert_scoring_func,
            activation=self.activation,
            apply_router_weight_on_input=self.apply_router_weight_on_input,
        )

        # Use reduce_scatter when DP > 1 but not using mori all2all kernels
        if use_dp_gather_scatter:
            if _tbo:
                tbo_yield_and_switch_from_compute_to_comm()
            if dp_eager_mode:
                final_hidden_states = reduce_scatterv(
                    final_hidden_states, sizes, dp_group
                )
            else:
                final_hidden_states = reduce_scatter_with_unpadding(
                    final_hidden_states, original_hidden_size
                )
            if _tbo:
                tbo_switch_to_compute_sync()

        if self.reduce_results and (self.tp_size > 1 or self.ep_size > 1):
            # Default set to False. (May have to add shared expert outputs.)
            final_hidden_states = get_tp_group().all_reduce(
                final_hidden_states, ca_fp8_quant=False
            )

        return final_hidden_states

    def forward_impl(self, hidden_states: torch.Tensor, router_logits: torch.Tensor):
        assert self.quant_method is not None

        if get_dp_group().world_size > 1:
            return self.forward_impl_graph(hidden_states, router_logits)

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
            expert_map=self.expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            fused_shared_experts_scoring_func=self.shared_expert_scoring_func,
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
