from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import final

import torch
from aiter import ActivationType, QuantType
from aiter.dist.parallel_state import get_dp_group
from aiter.fused_moe import fused_moe

from atom.model_ops.fused_moe.config import FusedMoEQuantConfig
from atom.model_ops.fused_moe.utils import disable_inplace
from atom.utils import envs
from atom.utils.forward_context import get_forward_context
from atom.utils.tbo.ubatching import tbo_overlap_enabled


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

    def _maybe_trim_dispatch_output(
        self,
        dispatch_a1: torch.Tensor,
        dispatch_scale: torch.Tensor | None,
        dispatch_ids: torch.Tensor,
        dispatch_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        expert_tokens_meta,
    ):
        """Trim the mori dispatch buffer's dead tail before fused_moe.

        Default (native/sglang/rtp) policy: under a uniform all-ranks-decode
        batch, trim to the static graph_bs*topk*dp bound so the shape is
        consistent across cudagraph capture/replay. atom-vllm needs a different,
        exact received-token trim for DP+EP mixed batches and overrides this
        method via a plugin patch -- keep this body frontend-agnostic.
        """
        context = get_forward_context().context
        if context is None:
            return dispatch_a1, dispatch_scale, dispatch_ids, dispatch_weights

        dp_size = get_dp_group().world_size
        topk = topk_ids.shape[1]
        # graph_bs keeps the trimmed shape consistent during capture/replay.
        total_valid_tokens = context.graph_bs * topk * dp_size
        all_ranks_decode = getattr(context, "dp_uniform_decode", not context.is_prefill)
        if total_valid_tokens < dispatch_a1.shape[0] and all_ranks_decode:
            dispatch_a1 = dispatch_a1[:total_valid_tokens]
            dispatch_ids = dispatch_ids[:total_valid_tokens]
            dispatch_weights = dispatch_weights[:total_valid_tokens]
            if dispatch_scale is not None:
                dispatch_scale = dispatch_scale[:total_valid_tokens]
        elif envs.ATOM_EP_TRIM_PREFILL and not all_ranks_decode:
            # Prefill keeps the whole (mbt * ep_size) buffer above, since
            # graph_bs is meaningless here. But rows are per-token and
            # de-duplicated per destination rank, so at most ONE row can arrive
            # per cluster token -- making the cross-rank token total an exact
            # upper bound on received rows.
            #
            # cu_tokens_across_dp_cpu is already on the host, so reading it costs
            # no sync. Using this rank's own count instead would be WRONG: under
            # non-uniform prefill another rank can hold more tokens and send more
            # rows here, and under-sizing drops received tokens silently.
            #
            # Prefill is not cudagraph-captured, so a per-pass dynamic bound is
            # fine (the static graph_bs bound above is what capture needs).
            dp_meta = getattr(get_forward_context(), "dp_metadata", None)
            cu = getattr(dp_meta, "cu_tokens_across_dp_cpu", None)
            if cu is not None:
                cluster_tokens = int(cu[-1])
                if 0 < cluster_tokens < dispatch_a1.shape[0]:
                    dispatch_a1 = dispatch_a1[:cluster_tokens]
                    dispatch_ids = dispatch_ids[:cluster_tokens]
                    dispatch_weights = dispatch_weights[:cluster_tokens]
                    if dispatch_scale is not None:
                        dispatch_scale = dispatch_scale[:cluster_tokens]
        return dispatch_a1, dispatch_scale, dispatch_ids, dispatch_weights

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
        expert_mask: torch.Tensor | None = None,
        apply_router_weight_on_input: bool = False,
        w1_scale: torch.Tensor | None = None,
        w2_scale: torch.Tensor | None = None,
        a1_scale: torch.Tensor | None = None,
        a2_scale: torch.Tensor | None = None,
        bias1: torch.Tensor | None = None,
        bias2: torch.Tensor | None = None,
        hidden_pad: int | None = 0,
        intermediate_pad: int | None = 0,
        moe_extra_args: dict | None = None,
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

        # mori dispatch expands the receive buffer to
        # (max_tokens * world_size, hidden_dim); only the first
        # `expert_num_tokens` rows are valid and fused_moe is driven by that
        # count via num_local_tokens, so the buffer must never be trimmed below
        # it. Trimming the dead tail keeps fused_moe off uninitialized rows; the
        # exact policy is frontend-specific (atom-vllm overrides this method),
        # so it is isolated in a hookable helper.
        (
            dispatch_a1,
            dispatch_scale,
            dispatch_ids,
            dispatch_weights,
        ) = self._maybe_trim_dispatch_output(
            dispatch_a1,
            dispatch_scale,
            dispatch_ids,
            dispatch_weights,
            topk_ids,
            expert_tokens_meta,
        )

        # TEMPORARY: capture for _test/bench_ep_moe.py. No-op unless
        # ATOM_EP_MOE_DUMP_DIR is set; delete with ep_moe_dump.py.
        from atom.model_ops.fused_moe.ep_moe_dump import maybe_dump_dispatched

        maybe_dump_dispatched(
            dispatch_a1,
            dispatch_scale,
            dispatch_ids,
            dispatch_weights,
            expert_tokens_meta.expert_num_tokens,
            expert_map,
            expert_mask,
            local_num_experts,
            w1,
            w2,
            {
                "activation": activation,
                "quant_type": quant_type,
                "global_num_experts": global_num_experts,
                "hidden_pad": hidden_pad,
                "intermediate_pad": intermediate_pad,
                "apply_router_weight_on_input": apply_router_weight_on_input,
                "gate_mode": (moe_extra_args or {}).get("gate_mode"),
                "swiglu_limit": (moe_extra_args or {}).get("swiglu_limit"),
                "w1_scale": w1_scale,
                "w2_scale": w2_scale,
            },
        )

        # aiter fused_moe expects a *binary* (0/1) expert_mask in this slot, not
        # the index-style expert_map (which carries -1 sentinels for non-local
        # experts). Passing expert_map here makes moe_sorting mis-classify
        # routing and compute out-of-range expert ids -> illegal memory access.
        # See PR #887 which fixed the same bug on the non-modular path.
        # Extra, model-/method-specific kwargs (e.g. DeepSeek-V4 MXFP4 needs
        # gate_mode=INTERLEAVE + swiglu_limit) are forwarded verbatim from the
        # quant method's apply() via `moe_extra_args`.
        extra_kwargs = dict(moe_extra_args or {})

        # Triton backend for the routed-expert GEMMs, in place of flydsl
        # fused_moe. Sits between dispatch and combine, so `_prepare`/`_finalize`
        # (and therefore the mori all-to-all) are untouched. The a8w4-specific
        # weights live on the layer, so the quant method forwards them through
        # `moe_extra_args` -- the modular kernel holds no layer reference.
        triton_experts = extra_kwargs.pop("triton_experts", None)
        if triton_experts is not None:
            from atom.model_ops.fused_moe_triton import (
                routing_from_dispatched,
                triton_kernel_fused_experts_a4w4_silu_gugu,
                triton_kernel_fused_experts_a8w4_silu_gugu,
            )

            # a4w4 takes the same signature and the same weights (both are w4);
            # only the activation quant differs, so the switch is just this.
            fused_experts_fn = (
                triton_kernel_fused_experts_a4w4_silu_gugu
                if envs.ATOM_USE_TRITON_MOE_EP_A4W4
                else triton_kernel_fused_experts_a8w4_silu_gugu
            )

            routing_data, gather_idx, scatter_idx, gate_valid = routing_from_dispatched(
                dispatch_weights,
                dispatch_ids,
                expert_map,
                local_num_experts,
                expert_tokens_meta.expert_num_tokens,
            )
            # gate_scal carries the dispatched router weights, and
            # apply_router_weight_on_input is False (mori asserts it), so GEMM2
            # applies them -- same split as flydsl's doweight_stage1=False.
            fused_out = fused_experts_fn(
                dispatch_a1,
                triton_experts["w13_weight"],
                triton_experts["w2_weight"],
                routing_data,
                gather_idx,
                scatter_idx,
                w13_scale=triton_experts["w13_scale"],
                w2_scale=triton_experts["w2_scale"],
                w13_swizzle_layout=triton_experts["w13_swizzle_layout"],
                w2_swizzle_layout=triton_experts["w2_swizzle_layout"],
                a13_scale=triton_experts.get("a13_scale"),
                a2_scale=triton_experts.get("a2_scale"),
                w1_bias=triton_experts.get("w1_bias"),
                w2_bias=triton_experts.get("w2_bias"),
                swiglu_limit=triton_experts.get("swiglu_limit", 10.0),
                apply_router_weight_on_input=apply_router_weight_on_input,
                gate_valid=gate_valid,
            )
            return self._finalize(
                output,
                fused_out,
                hidden_states,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
            )

        fused_out = fused_moe(
            dispatch_a1,
            w1,
            w2,
            dispatch_weights,
            dispatch_ids,
            expert_mask,
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
            **extra_kwargs,
        )
        return self._finalize(
            output,
            fused_out,
            hidden_states,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
        )
