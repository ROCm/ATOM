# SPDX-License-Identifier: MIT
"""Optional communication-fused backend for :class:`FusedMoE`."""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable
from types import ModuleType
from typing import TYPE_CHECKING, Any

import torch
from aiter import ActivationType, QuantType, dtypes
from aiter.dist.parallel_state import get_dp_group, get_tp_group
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.flydsl.moe_common import GateMode

from atom.config import get_current_atom_config
from atom.utils import envs
from atom.utils.custom_register import direct_register_custom_op

if TYPE_CHECKING:
    from atom.model_ops.moe import FusedMoE


logger = logging.getLogger("atom")


def _load_backend() -> tuple[ModuleType, ModuleType] | None:
    try:
        host = importlib.import_module("aiter.ops.flydsl.comm_fused_moe_host")
        runtime = importlib.import_module("aiter.ops.comm_fused_moe_runtime")
    except (ImportError, OSError):
        return None

    if not all(
        hasattr(host, name)
        for name in ("ShapeKey", "winners_for", "create_flydsl_comm_fused_runners")
    ) or not hasattr(runtime, "CommFusedMoeRuntime"):
        return None
    return host, runtime


def create_comm_fused_moe_backend(
    *,
    layer_quant_config: Any,
    online_quant: bool,
    parallel_config: Any,
    model_dim: int,
    inter_dim: int,
    experts: int,
    topk: int,
    activation: ActivationType,
    apply_router_weight_on_input: bool,
) -> CommFusedMoeBackend | None:
    """Return a configured backend when this FusedMoE layout is supported."""
    config = get_current_atom_config()
    use_dp_reduce_scatter = bool(
        config.enable_dp_attention and parallel_config.dp_size > 1
    )
    if (
        config.moe_backend != "standard"
        or online_quant
        or layer_quant_config is None
        or layer_quant_config.quant_dtype != dtypes.fp4x2
        or layer_quant_config.quant_type != QuantType.per_1x32
        or config.torch_dtype != torch.bfloat16
        or activation != ActivationType.Silu
        or not envs.ATOM_MOE_GU_ITLV
        or apply_router_weight_on_input
        or os.getenv("AITER_DISABLE_COMM_FUSED_MOE") == "1"
        or config.enable_tbo
        or config.enable_rapidserve
        or config.fake_eplb
        or config.enable_expert_parallel
        or (parallel_config.dp_size != 1 and not use_dp_reduce_scatter)
        or parallel_config.use_ep
        or config.prefill_context_parallel_size != 1
        or parallel_config.tp_size == 1
    ):
        return None

    modules = _load_backend()
    if modules is None:
        return None
    host, runtime = modules
    try:
        winners = host.winners_for(
            host.ShapeKey(
                get_gfx_runtime(),
                model_dim,
                inter_dim,
                experts,
                topk,
                parallel_config.tp_size,
                add_shared=True,
                comm="rs" if use_dp_reduce_scatter else "ar",
            )
        )
    except KeyError:
        return None
    return (
        CommFusedMoeBackend(
            host,
            runtime,
            use_dp_reduce_scatter=use_dp_reduce_scatter,
        )
        if winners
        else None
    )


class CommFusedMoeBackend:
    """Communication-fused execution plugged into an ordinary FusedMoE."""

    def __init__(
        self,
        host: ModuleType,
        runtime: ModuleType,
        *,
        use_dp_reduce_scatter: bool = False,
    ) -> None:
        self.host = host
        self.runtime_module = runtime
        self.use_dp_reduce_scatter = use_dp_reduce_scatter
        # Hash-routed DSV4 layers read separately gathered input_ids from the
        # forward context. Leave them on the ordinary path until that gather is
        # wired to the same compact rank order as hidden/router.
        self.supports_custom_routing = not use_dp_reduce_scatter
        self.runtime = None
        self._logged_buckets: set[int] = set()

    def _dpa_bucket_layout(self, max_tokens: int, world_size: int) -> tuple[int, int]:
        gathered_tokens = max_tokens * world_size
        bucket = self.runtime.bucket_for(gathered_tokens)
        if bucket < gathered_tokens or bucket % world_size:
            raise ValueError(
                f"DPA comm-fused bucket M={bucket} cannot represent "
                f"{world_size} rank shards for gathered M={gathered_tokens}"
            )
        return bucket, bucket // world_size

    def initialize(self, layer: FusedMoE) -> None:
        # Importing here avoids a cycle while atom.model_ops.moe defines FusedMoE.
        from atom.model_ops.moe import Mxfp4MoEMethod

        method = layer.quant_method
        if not isinstance(method, Mxfp4MoEMethod):
            raise TypeError("Communication-fused MoE requires MXFP4 weights")
        group = get_dp_group() if self.use_dp_reduce_scatter else get_tp_group()
        world_size = int(group.world_size)
        if self.use_dp_reduce_scatter:
            valid_layout = (
                world_size > 1
                and layer.dp_size == world_size
                and not layer.use_ep
                and layer.tp_size == world_size
            )
            layout_name = "DPA reduce-scatter"
        else:
            valid_layout = (
                world_size > 1
                and layer.dp_size == 1
                and not layer.use_ep
                and layer.tp_size == world_size
            )
            layout_name = "TP all-reduce"
        if not valid_layout:
            raise ValueError(
                f"Communication-fused MoE {layout_name} layout mismatch: "
                f"group={world_size}, moe_tp={layer.tp_size}, "
                f"dp={layer.dp_size}, use_ep={layer.use_ep}"
            )

        runner_args = {
            "tp_group": group,
            "model_dim": layer.hidden_size,
            "inter_dim": layer.intermediate_size_per_partition,
            "experts": layer.global_num_experts,
            "topk": layer.top_k,
            "comm": "rs" if self.use_dp_reduce_scatter else "ar",
            "add_shared": True,
        }
        self.runtime = self.runtime_module.CommFusedMoeRuntime(
            runners=self.host.create_flydsl_comm_fused_runners(**runner_args)
        )

    def supports(self, tokens: int) -> bool:
        if self.runtime is None:
            return False
        if not self.use_dp_reduce_scatter:
            return self.runtime.supports(tokens)
        from atom.utils.forward_context import get_forward_context

        ctx = get_forward_context()
        dp_metadata = ctx.dp_metadata
        max_tokens = (
            int(dp_metadata.max_tokens_across_dp)
            if dp_metadata is not None
            else int(ctx.context.running_tokens)
        )
        world_size = int(get_dp_group().world_size)
        gathered_tokens = max_tokens * world_size
        try:
            bucket, _ = self._dpa_bucket_layout(max_tokens, world_size)
        except ValueError:
            return False
        supported = self.runtime.supports(bucket)
        if supported and bucket not in self._logged_buckets:
            config = self.runtime.runners.configs[bucket]
            logger.info(
                "comm-fused DPA rank=%d selected local_max=%d gathered=%d "
                "bucket=%d config=%s",
                int(get_dp_group().rank_in_group),
                max_tokens,
                gathered_tokens,
                bucket,
                config,
            )
            self._logged_buckets.add(bucket)
        return supported

    @staticmethod
    def _pad_rows(tensor: torch.Tensor, rows: int) -> torch.Tensor:
        if tensor.shape[0] == rows:
            return tensor
        # Padding participates in expert selection after the rank-major gather.
        # Leaving it uninitialized can create arbitrary router scores and
        # non-zero routed contributions that contaminate another rank's RS
        # shard.  Zero hidden rows are neutral regardless of which expert the
        # zero router row selects.
        padded = tensor.new_zeros((rows, *tensor.shape[1:]))
        padded[: tensor.shape[0]].copy_(tensor)
        return padded

    def _gather_dpa_inputs(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, list[int] | None]:
        """Gather DPA inputs in compact rank order when Window supports ragged M."""
        from atom.utils.forward_context import get_forward_context

        ctx = get_forward_context()
        local_tokens = int(hidden_states.shape[0])
        group = get_dp_group()
        world_size = int(group.world_size)
        dp_metadata = ctx.dp_metadata
        max_tokens = (
            int(dp_metadata.max_tokens_across_dp)
            if dp_metadata is not None
            else int(ctx.context.running_tokens)
        )
        if max_tokens < local_tokens:
            raise ValueError(
                f"DPA comm-fused padding height {max_tokens} is below "
                f"local token count {local_tokens}"
            )

        bucket, shard_rows = self._dpa_bucket_layout(max_tokens, world_size)
        use_ragged_m = dp_metadata is not None and self.runtime.supports_ragged_m(
            bucket
        )

        if use_ragged_m:
            sizes = [int(size) for size in dp_metadata.get_sizes_across_dp()]
            rank = int(group.rank_in_group)
            if len(sizes) != world_size:
                raise ValueError(
                    "DPA ragged sizes length must equal world size: "
                    f"sizes={len(sizes)}, world_size={world_size}"
                )
            if sizes[rank] != local_tokens:
                raise ValueError(
                    f"DPA rank {rank} has {local_tokens} input rows but "
                    f"metadata reports {sizes[rank]}"
                )
            hidden_states, router_logits = group.device_communicator.all_gatherv(
                [hidden_states, router_logits],
                dim=0,
                sizes=sizes,
            )
            return hidden_states, router_logits, local_tokens, sizes

        # Window RS assigns output row ``rank * (bucket / world_size) + row``
        # to each rank.  Padding only the gathered tensor's global tail would
        # leave the real rank-major boundaries at ``rank * max_tokens`` and
        # shift every rank after rank 0 whenever the bucket is larger.  Insert
        # the bucket slack inside every rank's shard before all-gather instead.
        # Keep the two ordinary gather payloads separate.  Concatenating them
        # made the large transfer slower and forced both split views through
        # full-size contiguous materialization on every MoE layer.
        hidden_states = group.all_gather(
            self._pad_rows(hidden_states, shard_rows),
            use_custom=True,
            dim=0,
        )
        router_logits = group.all_gather(
            self._pad_rows(router_logits, shard_rows),
            use_custom=True,
            dim=0,
        )
        return hidden_states, router_logits, local_tokens, None

    def forward(
        self,
        layer: FusedMoE,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_partial: torch.Tensor | None,
        before_stage2: Callable[[int], torch.Tensor] | None = None,
        before_shared_add: Callable[[], None] | None = None,
        stage2_stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        if before_stage2 is not None:
            return self.forward_impl(
                layer,
                hidden_states,
                router_logits,
                shared_partial,
                before_stage2=before_stage2,
                before_shared_add=before_shared_add,
                stage2_stream=stage2_stream,
            )
        if shared_partial is None:
            raise ValueError("Communication-fused MoE requires a shared partial")
        return torch.ops.aiter.comm_fused_moe_forward(
            hidden_states,
            router_logits,
            shared_partial,
            layer.layer_name,
        )

    def forward_impl(
        self,
        layer: FusedMoE,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_partial: torch.Tensor | None,
        before_stage2: Callable[[int], torch.Tensor] | None = None,
        before_shared_add: Callable[[], None] | None = None,
        stage2_stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        method = layer.quant_method
        reduce_scatter_sizes = None
        if self.use_dp_reduce_scatter:
            (
                hidden_states,
                router_logits,
                local_tokens,
                reduce_scatter_sizes,
            ) = self._gather_dpa_inputs(hidden_states, router_logits)
            if reduce_scatter_sizes is None:
                shard_rows = hidden_states.shape[0] // int(get_dp_group().world_size)
                if shared_partial is not None:
                    shared_partial = self._pad_rows(shared_partial, shard_rows)
        topk_weights, topk_ids = method.select_experts_with_record(
            layer=layer,
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=layer.use_grouped_topk,
            top_k=layer.top_k,
            renormalize=layer.renormalize,
            topk_group=layer.topk_group,
            num_expert_group=layer.num_expert_group,
            global_num_experts=layer.global_num_experts,
            custom_routing_function=layer.custom_routing_function,
            scoring_func=layer.scoring_func,
            e_score_correction_bias=layer.e_score_correction_bias,
            fused_shared_experts_scoring_func=layer.shared_expert_scoring_func,
        )
        output = self.runtime.run(
            hidden_states=hidden_states,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weight=topk_weights,
            topk_ids=topk_ids,
            expert_mask=layer.expert_mask,
            activation=layer.activation,
            quant_type=method.quant_type,
            doweight_stage1=layer.apply_router_weight_on_input,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            hidden_pad=method.hidden_pad,
            intermediate_pad=method.intermediate_pad,
            bias1=layer.w13_bias,
            bias2=layer.w2_bias,
            swiglu_limit=float(layer.swiglu_limit),
            gate_mode=(
                GateMode.INTERLEAVE.value
                if method.is_guinterleave
                else GateMode.SEPARATED.value
            ),
            shared_partial=shared_partial,
            before_stage2=before_stage2,
            before_shared_add=before_shared_add,
            stage2_stream=stage2_stream,
            reduce_scatter_sizes=reduce_scatter_sizes,
            # The DPA input-gather phase completes only after every rank has
            # entered this invocation. It therefore also proves that every peer
            # finished reading the previous Direct partial before Stage2 reuses
            # it, so Direct can omit its standalone reuse wait.
            reuse_is_synchronized=self.use_dp_reduce_scatter,
        )
        if not self.use_dp_reduce_scatter:
            return output
        if reduce_scatter_sizes is not None:
            return output
        if output.shape[0] == hidden_states.shape[0]:
            shard_begin = int(get_dp_group().rank_in_group) * shard_rows
            return output[shard_begin : shard_begin + local_tokens]
        return output[:local_tokens]


def comm_fused_moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_partial: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    layer = get_current_atom_config().compilation_config.static_forward_context[
        layer_name
    ]
    return layer._comm_fused_moe.forward_impl(
        layer, hidden_states, router_logits, shared_partial
    )


def _comm_fused_moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    shared_partial: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="comm_fused_moe_forward",
    op_func=comm_fused_moe_forward,
    mutates_args=[],
    fake_impl=_comm_fused_moe_forward_fake,
    tags=(torch.Tag.needs_fixed_stride_order,),
)


__all__ = ["CommFusedMoeBackend", "create_comm_fused_moe_backend"]
