from abc import ABC, abstractmethod

from dataclasses import dataclass
from atom.model_ops.fused_moe.config import FusedMoEQuantConfig
from atom.model_ops.fused_moe.utils import cdiv, count_expert_num_tokens, _resize_cache, _slice_scales, disable_inplace
from atom.utils import envs
from atom.utils.dbo.ubatching import dbo_current_ubatch_id, dbo_enabled, dbo_maybe_run_recv_hook, dbo_register_recv_hook, dbo_yield
import torch
from math import prod
from typing import Callable, Optional, final
from enum import Enum
from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

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
    ) -> PrepareResultType:
        """
        Perform any quantization (and/or) dispatching needed for this kernel.
        - a1: The (unquantized) input to the MoE layer.
        - topk_ids: The topk ids.
        - topk_weights: The topk weights.
        - num_experts: The total number of experts in the global expert space.
        - expert_map: A tensor mapping expert indices from the global expert
          space to the local expert space of the expert parallel shard.
        - apply_router_weight_on_input: When True, apply the weights to the
          activations, before quantization + dispatching.
        - quant_config: Quantization info provided by the fused experts.

        Returns a tuple of:
        - quantized + dispatched a.
        - Optional quantized + dispatched a1_scales.
        - Optional ExpertTokensMetadata containing gpu/cpu tensors
          as big as the number of local experts with the information about the
          number of tokens assigned to each local expert.
        - Optional dispatched expert topk IDs
        - Optional dispatched expert topk weight
        """
        raise NotImplementedError

    def supports_async(self) -> bool:
        """
        Indicates whether or not this class implements prepare_async and
        finalize_async.
        """
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
        """
        Perform any quantization (and/or) dispatching needed for this kernel
        but do not wait for results from other workers.
        - a1: The (unquantized) input to the MoE layer.
        - a1_scale: Optional scales for a1
        - a2_scale: Optional scales for the second MoE gemm.  Required to make
          sure the quantization is consistent for both gemms.
        - topk_ids: The topk ids.
        - topk_weights: The topk weights.
        - num_experts: The total number of experts in the global expert space.
        - expert_map: A tensor mapping expert indices from the global expert
          space to the local expert space of the expert parallel shard.
        - apply_router_weight_on_input: When True, apply the weights to the
          activations, before quantization + dispatching.

        Returns a callback or a hook callback pair that when invoked waits for
        results from other workers and has the same return signature as
        `prepare`, if a hook is returned this is more lightweight check that
        the recv is complete without doing extra work (used by DBO, will be
        refactored in the very near future)

        e.g.

        ret = obj.prepare_async(...)

        if isinstance(ret, tuple):
            hook, receiver = ret
            hook()

        if hook is not None:
        a, a_scales, expert_meta, topk_ids, topk_weights = receiver()

        is equivalent to:

        a, a_scales, expert_meta, topk_ids, topk_weights = obj.prepare(...)
        """
        raise NotImplementedError

    @abstractmethod
    def finalize(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> None:
        """
        Perform any combine plus apply weights and perform a reduction on the
        fused experts output.
        - output: The output tensor, written in place.  Must be (M, K) shape.
        - fused_expert_output: The unweighted, unreduced output of the fused
          experts, it will have (M, topk, K) shape.
        - topk_weights: The weights to be applied to the fused_experts_output.
        - topk_ids: The topk_ids.
        - apply_router_weight_on_input: When False, apply the weights to
          fused_expert_output.
        - weight_and_reduce_impl: An optional TopKWeightAndReduce
          implementation.
        """
        raise NotImplementedError

    def finalize_async(
        self,
        output: torch.Tensor,
        fused_expert_output: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        apply_router_weight_on_input: bool,
    ) -> tuple[Callable, Callable] | Callable:
        """
        Perform any combine plus apply weights and perform a reduction on the
        fused experts output but do not wait for results from other workers.
        - output: The output tensor, written in place.  Must be (M, K) shape.
        - fused_expert_output: The unweighted, unreduced output of the fused
          experts, it will have (M, topk, K) shape.
        - topk_weights: The weights to be applied to the fused_experts_output.
        - topk_ids: The topk_ids.
        - apply_router_weight_on_input: When False, apply the weights to
          fused_expert_output.
        - weight_and_reduce_impl: An optional TopKWeightAndReduce
          implementation.

        Returns a callback or a hook callback pair that when invoked waits for
        results from other workers and has the same return signature as
        `finalize`, if a hook is returned this is more lightweight check that
        the recv is complete without doing extra work (used by DBO, will be
        refactored in the very near future)

        ret = obj.finalize_async(output, ...)
        ... output not valid yet ...
        if isinstance(ret, tuple):
            hook, receiver = ret
            hook()
        receiver()
        ... output valid here ...

        is equivalent to:

        obj.finalize(output, ...)
        """
        raise NotImplementedError


    @abstractmethod
    def topk_indices_dtype(self) -> torch.dtype | None:
        """
        The PrepareFinalize All2All implementations generally constrain the
        dtype of the topk_ids they support. This function returns the
        required topk indices dtype so it can be respected.
        Return None if there are no such restrictions.
        """
        raise NotImplementedError

    @abstractmethod
    def max_num_tokens_per_rank(self) -> int | None:
        """
        Some PrepareFinalize All2All implementations are batched. Meaning,
        they can process only as set of tokens at a time. This
        function returns the batch size i.e the maximum number of tokens
        the implementation can process at a time.
        Return None if there are no such restrictions.
        """
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
        # assert (
            # prepare_finalize.activation_format == fused_experts.activation_formats[0]
        # ), (
        #     f"{prepare_finalize.__class__.__name__}."
            # f"{prepare_finalize.activation_format} == "
            # f"{fused_experts.__class__.__name__}."
            # f"{fused_experts.activation_formats[0]}"
        # )

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
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
        ExpertTokensMetadata | None,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        The _prepare method is a wrapper around self.prepare_finalize.prepare
        that handles DBO and async.
        """
        if not self.prepare_finalize.supports_async():
            # We shouldn't be running an a2a kernel that doesn't
            # support async prepare/finalize
            # TODO(lucas): enable in follow-up
            assert not dbo_enabled()

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
            )
        else:
            # Overlap shared expert compute with all2all dispatch.
            dbo_maybe_run_recv_hook()
            prepare_ret = self.prepare_finalize.prepare_async(
                hidden_states,
                topk_weights,
                topk_ids,
                global_num_experts,
                expert_map,
                apply_router_weight_on_input,
                self.quant_config,
            )

            # TODO(lucas): refactor this in the alternative schedules followup
            # currently unpack if we have hook + receiver pair or just
            # receiver (see finalize_async docstring)
            hook, receiver = (
                prepare_ret if isinstance(prepare_ret, tuple) else (None, prepare_ret)
            )

            if hook is not None:
                if dbo_enabled():
                    # If DBO is being used, register the hook with the ubatch
                    # context and call it in dbo_maybe_run_recv_hook instead of
                    #  passing it to the receiver.
                    dbo_register_recv_hook(hook)
                    dbo_yield()
                else:
                    hook()

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
        that handles DBO, async and shared expert overlap.
        """
        shared_output: torch.Tensor | None = None

        if not self.prepare_finalize.supports_async():
            assert not dbo_enabled()

            self.prepare_finalize.finalize(
                output,
                fused_out,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                # self.fused_experts.finalize_weight_and_reduce_impl(),
            )
            # if self.shared_experts is not None:
            #     shared_output = self.shared_experts(hidden_states)
        else:
            finalize_ret = self.prepare_finalize.finalize_async(
                output,
                fused_out,
                topk_weights,
                topk_ids,
                apply_router_weight_on_input,
                # self.fused_experts.finalize_weight_and_reduce_impl(),
            )

            if self.shared_experts is not None:
                shared_output = self.shared_experts(hidden_states)

            # TODO(lucas): refactor this in the alternative schedules followup
            # currently unpack if we have hook + receiver pair or just
            # receiver (see finalize_async docstring)
            hook, receiver = (
                finalize_ret
                if isinstance(finalize_ret, tuple)
                else (None, finalize_ret)
            )

            if hook is not None:
                if dbo_enabled():
                    # If DBO is being used, register the hook with the ubatch
                    # context and call it in dbo_maybe_run_recv_hook instead of
                    #  passing it to the receiver.
                    dbo_register_recv_hook(hook)
                    dbo_yield()
                else:
                    hook()

            receiver()
        return output
        if self.shared_experts is None:
            return output
        else:
            assert shared_output is not None
            return shared_output, output

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
        """
        This function computes a Mixture of Experts (MoE) layer using two sets
        of weights, w1 and w2, and top-k gating mechanism.

        Parameters:
        - hidden_states: (torch.Tensor): The input tensor to the MoE layer.
        - w1 (torch.Tensor): The first set of expert weights.
        - w2 (torch.Tensor): The second set of expert weights.
        - topk_weights (torch.Tensor): The topk weights applied at the end of
          the layer.
        - topk_ids (torch.Tensor): A map of row to expert id.
        - inplace (bool): If True, perform the operation in-place.
          Defaults to False.
        - activation (str): The activation function to apply after the first
          MoE layer.
        - global_num_experts (int): The total number of experts in the global
          expert space.
        - expert_map (Optional[torch.Tensor]):  A tensor mapping expert indices
          from the global expert space to the local expert space of the expert
          parallel shard.
        - apply_router_weight_on_input (bool): When true, the topk weights are
          applied directly on the inputs. This is only applicable when topk is
          1.

        Returns:
        - torch.Tensor: The output tensor after applying the MoE layer.
        """
        if inplace and self.shared_experts is None and not disable_inplace():
            output = hidden_states
        else:
            output = torch.zeros_like(hidden_states)

        local_num_experts = w1.size(0)
        if global_num_experts == -1:
            global_num_experts = local_num_experts
        dispatch_a1, dispatch_scale, expert_tokens_meta, dispatch_ids, dispatch_weights = self._prepare(
            hidden_states,
            topk_weights,
            topk_ids,
            global_num_experts,
            expert_map,
            apply_router_weight_on_input,
        )
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
        return self._finalize(
            output,
            fused_out,
            hidden_states,
            topk_weights,
            topk_ids,
            apply_router_weight_on_input,
        )


