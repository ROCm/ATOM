from abc import ABC, abstractmethod

import os
from dataclasses import dataclass
from atom.model_ops.fused_moe.config import FusedMoEQuantConfig
from atom.model_ops.fused_moe.utils import disable_inplace
from atom.utils.tbo.ubatching import tbo_overlap_enabled
from atom.utils.forward_context import get_forward_context
import torch
import torch.nn.functional as F
from typing import Callable, Optional, final
from enum import Enum
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe
from aiter.dist.parallel_state import get_dp_group


class FusedMoEActivationFormat(Enum):
    """
    The standard activation format (num_tokens, hidden dim).
    """

    Standard = ("standard",)
    """
    The batched experts format (num experts, max tokens per expert, hidden dim)
    """
    BatchedExperts = ("batched_experts",)


@dataclass
class ExpertTokensMetadata:
    """
    Metadata regarding expert-token routing.
    """

    expert_num_tokens: torch.Tensor
    expert_num_tokens_cpu: torch.Tensor | None

    @staticmethod
    def make_from_list(
        expert_num_tokens_list: list[int], device: str
    ) -> "ExpertTokensMetadata":
        expert_num_tokens_cpu = torch.tensor(
            expert_num_tokens_list, device="cpu", dtype=torch.int32
        )
        return ExpertTokensMetadata(
            expert_num_tokens=expert_num_tokens_cpu.to(device, non_blocking=True),
            expert_num_tokens_cpu=expert_num_tokens_cpu,
        )


PrepareResultType = tuple[
    torch.Tensor,
    torch.Tensor | None,
    ExpertTokensMetadata | None,
    torch.Tensor | None,
    torch.Tensor | None,
]

ReceiverType = Callable[[], PrepareResultType]


_E2M1_TO_FLOAT = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
)


def _env_flag(name: str) -> bool:
    return os.getenv(name, "0").lower() in ("1", "true", "yes", "on")


def _inverse_shuffle_weight(weight: torch.Tensor) -> torch.Tensor:
    """Undo aiter.ops.shuffle.shuffle_weight(..., layout=(16, 16))."""
    n = weight.shape[-2]
    packed_k = weight.shape[-1]
    bn = 16
    bk = 32
    k_lane = 16
    if n % bn != 0 or packed_k % bk != 0:
        raise ValueError(f"cannot unshuffle weight with shape {tuple(weight.shape)}")

    return (
        weight.reshape(-1, n // bn, packed_k // bk, bk // k_lane, bn, k_lane)
        .permute(0, 1, 4, 2, 3, 5)
        .contiguous()
        .reshape_as(weight)
    )


def _inverse_shuffle_scale(scale: torch.Tensor) -> torch.Tensor:
    """Undo aiter.utility.fp4_utils.e8m0_shuffle for 2D scale tensors."""
    if scale.ndim != 2:
        raise ValueError(f"scale must be 2D, got shape {tuple(scale.shape)}")

    rows, groups = scale.shape
    if rows % 32 != 0 or groups % 8 != 0:
        raise ValueError(f"cannot unshuffle scale with shape {tuple(scale.shape)}")

    return (
        scale.reshape(rows // 32, groups // 8, 4, 16, 2, 2)
        .permute(0, 5, 3, 1, 4, 2)
        .contiguous()
        .reshape_as(scale)
    )


def _decode_fp4_e2m1fn_x2(packed: torch.Tensor) -> torch.Tensor:
    if packed.dtype != torch.uint8:
        packed = packed.contiguous().view(torch.uint8)
    lut = _E2M1_TO_FLOAT.to(device=packed.device)

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).flatten(start_dim=-2)

    signs = (codes & 0x08).to(torch.bool)
    magnitudes = (codes & 0x07).to(torch.long)
    values = lut[magnitudes]
    return torch.where(signs, -values, values)


def _decode_e8m0fnu(scale: torch.Tensor) -> torch.Tensor:
    if scale.dtype != torch.uint8:
        scale = scale.contiguous().view(torch.uint8)
    exponent = scale.to(torch.int32).clamp(max=254)
    values = torch.exp2(exponent.to(torch.float32) - 127.0)
    return torch.where(exponent == 0, torch.zeros_like(values), values)


def _dequant_mxfp4_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
    *,
    unshuffle: bool,
) -> torch.Tensor:
    weight = weight.contiguous().view(torch.uint8)
    scale = scale.contiguous().view(torch.uint8)
    if unshuffle:
        weight = _inverse_shuffle_weight(weight)
        scale = _inverse_shuffle_scale(scale)

    dequant = _decode_fp4_e2m1fn_x2(weight)
    out_features, in_features = dequant.shape
    if scale.shape != (out_features, in_features // 32):
        raise ValueError(
            f"scale shape {tuple(scale.shape)} does not match weight shape "
            f"{tuple(weight.shape)} after FP4 unpack"
        )

    scale_f32 = _decode_e8m0fnu(scale)
    return (
        dequant.reshape(out_features, in_features // 32, 32) * scale_f32[:, :, None]
    ).reshape(out_features, in_features)


def _local_expert_hash(expert_mask: torch.Tensor) -> torch.Tensor:
    expert_mask = expert_mask.to(torch.int32)
    hashed = expert_mask.cumsum(0, dtype=torch.int32) - 1
    hashed[expert_mask == 0] = -1
    return hashed


@torch.no_grad()
def _torch_native_mxfp4_fused_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    expert_mask: torch.Tensor | None,
    num_local_tokens: torch.Tensor | None,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    *,
    activation: ActivationType,
) -> torch.Tensor:
    if activation != ActivationType.Silu:
        raise NotImplementedError(
            f"torch native MXFP4 MoE only supports Silu, got {activation}"
        )
    if w1_scale is None or w2_scale is None:
        raise ValueError("torch native MXFP4 MoE requires w1_scale and w2_scale")

    num_experts, w1_out_features, _ = w1.shape
    _, w2_out_features, _ = w2.shape
    w1_scale = w1_scale.contiguous().view(torch.uint8).reshape(
        num_experts, w1_out_features, -1
    )
    w2_scale = w2_scale.contiguous().view(torch.uint8).reshape(
        num_experts, w2_out_features, -1
    )

    if expert_mask is not None:
        hashed = _local_expert_hash(expert_mask)
        valid = (topk_ids >= 0) & (topk_ids < hashed.numel())
        mapped_ids = torch.full_like(topk_ids, -1)
        mapped_ids[valid] = hashed[topk_ids[valid]]
    else:
        mapped_ids = topk_ids

    if num_local_tokens is not None:
        num_rows = min(hidden_states.shape[0], int(num_local_tokens.sum().item()))
    else:
        num_rows = hidden_states.shape[0]

    output = torch.zeros(
        hidden_states.shape,
        dtype=torch.float32,
        device=hidden_states.device,
    )
    if num_rows <= 0:
        return output.to(hidden_states.dtype)

    mapped_ids = mapped_ids[:num_rows]
    topk_weights = topk_weights[:num_rows]
    hidden_rows = hidden_states[:num_rows]

    unshuffle = os.getenv("ATOM_MOE_TORCH_NATIVE_UNSHUFFLE", "1") != "0"
    local_experts = torch.unique(mapped_ids[mapped_ids >= 0]).sort().values.tolist()
    for local_expert in local_experts:
        token_idx, topk_idx = (mapped_ids == local_expert).nonzero(as_tuple=True)
        if token_idx.numel() == 0:
            continue

        w13_f32 = _dequant_mxfp4_weight(
            w1[local_expert],
            w1_scale[local_expert],
            unshuffle=unshuffle,
        )
        w2_f32 = _dequant_mxfp4_weight(
            w2[local_expert],
            w2_scale[local_expert],
            unshuffle=unshuffle,
        )

        x = hidden_rows[token_idx].to(torch.float32)
        gate_up = x @ w13_f32.T
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = F.silu(gate) * up
        expert_out = intermediate @ w2_f32.T
        expert_out = expert_out * topk_weights[token_idx, topk_idx].to(torch.float32)[
            :, None
        ]
        output[:num_rows].index_add_(0, token_idx, expert_out)

        del w13_f32, w2_f32, gate_up, intermediate, expert_out

    return output.to(hidden_states.dtype)


def _report_native_diff(
    name: str, native_out: torch.Tensor, fused_out: torch.Tensor
) -> None:
    native_f32 = native_out.float()
    fused_f32 = fused_out.float()
    diff = native_f32 - fused_f32
    native_absmax = float(native_f32.abs().max().item()) if native_f32.numel() else 0.0
    fused_absmax = float(fused_f32.abs().max().item()) if fused_f32.numel() else 0.0
    diff_absmax = float(diff.abs().max().item()) if diff.numel() else 0.0
    rmse = float(torch.sqrt(torch.mean(diff * diff)).item()) if diff.numel() else 0.0
    denom = max(fused_absmax, 1e-12)
    print(
        f"[ATOM_MOE_TORCH_NATIVE] {name}: "
        f"native_absmax={native_absmax} fused_absmax={fused_absmax} "
        f"diff_absmax={diff_absmax} rmse={rmse} rel_absmax={diff_absmax / denom}"
    )


class FusedMoEPrepareAndFinalize(ABC):
    """
    An abstract base class for the [Quantize-Prepare] and [Finalize] steps
    described above.
    """

    @abstractmethod
    def prepare(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_type: QuantType = QuantType.No,
    ) -> PrepareResultType:
        raise NotImplementedError

    def supports_async(self) -> bool:
        return False

    def prepare_async(
        self,
        a1: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
    ) -> tuple[Callable, ReceiverType] | ReceiverType:
        raise NotImplementedError

    @abstractmethod
    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor:
        raise NotImplementedError

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> tuple[Callable, Callable] | Callable:
        raise NotImplementedError

    @abstractmethod
    def topk_indices_dtype(self) -> torch.dtype | None:
        raise NotImplementedError

    @abstractmethod
    def max_num_tokens_per_rank(self) -> int | None:
        raise NotImplementedError

    @abstractmethod
    def num_dispatchers(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def output_is_reduced(self) -> bool:
        """
        Indicates whether or not the output of finalize is reduced across all
        ranks.
        """
        raise NotImplementedError


@final
class FusedMoEModularKernel(torch.nn.Module):

    def __init__(
        self,
        prepare_finalize: FusedMoEPrepareAndFinalize,
        shared_experts: torch.nn.Module | None = None,
        quant_config: FusedMoEQuantConfig = None,
    ):
        super().__init__()
        self.prepare_finalize = prepare_finalize
        # self.fused_experts = fused_experts
        self.shared_experts = shared_experts
        self.quant_config = quant_config

    def output_is_reduced(self) -> bool:
        """
        Indicates whether or not the output of fused MoE kernel
        is reduced across all ranks.
        """
        return self.prepare_finalize.output_is_reduced()

    def _prepare(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        apply_router_weight_on_input: bool,
        quant_type: QuantType = QuantType.No,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        ExpertTokensMetadata | None,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        The _prepare method is a wrapper around self.prepare_finalize.prepare
        that handles TBO and async.
        """
        if not self.prepare_finalize.supports_async():
            assert not tbo_overlap_enabled()

            (
                a1q,
                a1q_scale,
                expert_tokens_meta,
                _expert_topk_ids,
                _expert_topk_weights,
            ) = self.prepare_finalize.prepare(
                hidden_states,
                topk_weights,
                topk_ids,
                global_num_experts,
                expert_map,
                apply_router_weight_on_input,
                self.quant_config,
                quant_type,
            )
        else:
            from atom.utils.tbo.ubatching import (
                tbo_maybe_run_recv_hook,
                tbo_register_recv_hook,
                tbo_yield,
            )

            tbo_maybe_run_recv_hook()

            result = self.prepare_finalize.prepare_async(
                hidden_states,
                topk_weights,
                topk_ids,
                global_num_experts,
                expert_map,
                apply_router_weight_on_input,
            )
            if isinstance(result, tuple):
                hook, receiver = result
                tbo_register_recv_hook(hook)
                tbo_yield()
            else:
                receiver = result
            (
                a1q,
                a1q_scale,
                expert_tokens_meta,
                _expert_topk_ids,
                _expert_topk_weights,
            ) = receiver()

        # Maybe prepare gathered topk_ids and topk_weights from other EP ranks.
        topk_ids = topk_ids if _expert_topk_ids is None else _expert_topk_ids
        topk_weights = (
            topk_weights if _expert_topk_weights is None else _expert_topk_weights
        )

        return a1q, a1q_scale, expert_tokens_meta, topk_ids, topk_weights

    def _finalize(
        self,
        output: torch.Tensor,
        fused_out: torch.Tensor,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        The _finalize method is a wrapper around self.prepare_finalize.finalize
        that handles TBO, async and shared expert overlap.
        """

        if not self.prepare_finalize.supports_async():
            assert not tbo_overlap_enabled()

            output = self.prepare_finalize.finalize(
                output,
                fused_out,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
            )
        else:
            from atom.utils.tbo.ubatching import (
                tbo_maybe_run_recv_hook,
                tbo_register_recv_hook,
                tbo_yield,
            )

            tbo_maybe_run_recv_hook()

            result = self.prepare_finalize.finalize_async(
                output,
                fused_out,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
            )
            if isinstance(result, tuple):
                hook, receiver = result
                tbo_register_recv_hook(hook)
                tbo_yield()
                output = receiver()
            else:
                output = result()
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        inplace: bool = False,
        activation: ActivationType = ActivationType.Silu,
        quant_type: QuantType = QuantType.No,
        global_num_experts: int = -1,
        expert_map: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        w1_scale: Optional[torch.Tensor] = None,
        w2_scale: Optional[torch.Tensor] = None,
        a1_scale: Optional[torch.Tensor] = None,
        a2_scale: Optional[torch.Tensor] = None,
        bias1: Optional[torch.Tensor] = None,
        bias2: Optional[torch.Tensor] = None,
        hidden_pad: Optional[int] = 0,
        intermediate_pad: Optional[int] = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        if inplace and self.shared_experts is None and not disable_inplace():
            output = hidden_states
        else:
            output = None

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts
        (
            dispatch_a1,
            dispatch_scale,
            expert_tokens_meta,
            dispatch_ids,
            dispatch_weights,
        ) = self._prepare(
            hidden_states,
            topk_weights,
            topk_ids,
            global_num_experts,
            expert_map,
            apply_router_weight_on_input,
            quant_type,
        )

        # optimize fused_moe hidden_states
        # mori dispatch expands buffer to (max_tokens * world_size, hidden_dim)
        # but actual valid tokens = graph_bs * topk * dp_size
        context = get_forward_context().context
        dp_size = get_dp_group().world_size
        topk = topk_ids.shape[1]
        # Use graph_bs for cudagraph compatibility (consistent shape during capture/replay)
        total_valid_tokens = context.graph_bs * topk * dp_size
        if total_valid_tokens < dispatch_a1.shape[0] and not context.is_prefill:
            dispatch_a1 = dispatch_a1[:total_valid_tokens]
            dispatch_ids = dispatch_ids[:total_valid_tokens]
            dispatch_weights = dispatch_weights[:total_valid_tokens]
            if dispatch_scale is not None:
                dispatch_scale = dispatch_scale[:total_valid_tokens]

        use_native_mxfp4 = _env_flag("ATOM_MOE_USE_TORCH_NATIVE_MXFP4")
        compare_native_mxfp4 = _env_flag("ATOM_MOE_COMPARE_TORCH_NATIVE_MXFP4")
        native_out = None
        if use_native_mxfp4 or compare_native_mxfp4:
            if quant_type != QuantType.per_1x32:
                raise NotImplementedError(
                    "ATOM_MOE_USE_TORCH_NATIVE_MXFP4 only supports QuantType.per_1x32"
                )
            if dispatch_scale is not None or a1_scale is not None or a2_scale is not None:
                raise NotImplementedError(
                    "ATOM_MOE_USE_TORCH_NATIVE_MXFP4 does not support activation scales"
                )
            if bias1 is not None or bias2 is not None:
                raise NotImplementedError(
                    "ATOM_MOE_USE_TORCH_NATIVE_MXFP4 does not support MoE bias"
                )
            if apply_router_weight_on_input:
                raise NotImplementedError(
                    "ATOM_MOE_USE_TORCH_NATIVE_MXFP4 expects router weights on output"
                )

            native_out = _torch_native_mxfp4_fused_moe(
                dispatch_a1,
                w1,
                w2,
                dispatch_weights,
                dispatch_ids,
                expert_map,
                expert_tokens_meta.expert_num_tokens,
                w1_scale,
                w2_scale,
                activation=ActivationType(activation),
            )

        if use_native_mxfp4:
            fused_out = native_out
        else:
            fused_out = fused_moe(
                dispatch_a1,
                w1,
                w2,
                dispatch_weights,
                dispatch_ids,
                expert_map,
                activation,
                quant_type=quant_type,
                num_local_tokens=expert_tokens_meta.expert_num_tokens,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                a1_scale=dispatch_scale if dispatch_scale is not None else a1_scale,
                a2_scale=a2_scale,
                doweight_stage1=apply_router_weight_on_input,
                hidden_pad=hidden_pad,
                intermediate_pad=intermediate_pad,
                bias1=bias1,
                bias2=bias2,
                dtype=hidden_states.dtype,
            )
            if compare_native_mxfp4:
                _report_native_diff("dispatch fused_out", native_out, fused_out)
        return self._finalize(
            output,
            fused_out,
            hidden_states,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
        )
