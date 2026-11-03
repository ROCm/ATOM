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
from atom.model_ops.fused_moe.routed_rows import gather_rows
from atom.model_ops.topK import reassign_gathered_shared_ids
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

    gfx = get_gfx_runtime()
    modules = _load_backend()
    if modules is None:
        return None
    host, runtime = modules
    try:
        winners = host.winners_for(
            host.ShapeKey(
                gfx,
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
        self.runtime = None
        self._logged_buckets: set[int] = set()

    @staticmethod
    def configure_quant_method(method: Any) -> None:
        """Select the FlyDSL weight layout before weights are prepared."""
        # Importing here avoids a cycle while atom.model_ops.moe defines FusedMoE.
        from atom.model_ops.moe import Mxfp4MoEMethod

        if not isinstance(method, Mxfp4MoEMethod):
            raise TypeError("Communication-fused MoE requires MXFP4 weights")
        # Mega, Direct, and Window consume the ordinary Aiter/FlyDSL layout.
        # Triton rewrites the same parameters into a different layout during
        # process_weights_after_loading(), so make the ownership decision before
        # create_weights and weight post-processing can observe these flags.
        method.use_triton = False
        method.use_triton_decode = False

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
        if method.use_triton or method.use_triton_decode:
            raise RuntimeError(
                "Communication-fused MoE requires the Aiter/FlyDSL weight layout; "
                "configure_quant_method() must run before weight preparation"
            )
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

    @staticmethod
    def _dpa_max_tokens(ctx: Any) -> int | None:
        """Return a DP-uniform dispatch height, or None when it is unproven."""
        dp_metadata = ctx.dp_metadata
        if dp_metadata is not None:
            return int(dp_metadata.max_tokens_across_dp)
        if not bool(getattr(ctx.context, "running_tokens_are_unified", False)):
            return None
        return int(ctx.context.running_tokens)

    def supports(self, tokens: int) -> bool:
        if self.runtime is None:
            return False
        if not self.use_dp_reduce_scatter:
            return self.runtime.supports(tokens)
        from atom.utils.forward_context import get_forward_context

        ctx = get_forward_context()
        max_tokens = self._dpa_max_tokens(ctx)
        if max_tokens is None:
            return False
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
    def _pad_rows(tensor: torch.Tensor, rows: int, fill: int = 0) -> torch.Tensor:
        if tensor.shape[0] == rows:
            return tensor
        # The rows are already routed, so a pad row must say it goes nowhere
        # (ids -1, weight 0) and add nothing (hidden and shared partial 0).
        # Left uninitialized, its weight and ids would route garbage into
        # another rank's RS shard.
        padded = tensor.new_full((rows, *tensor.shape[1:]), fill)
        padded[: tensor.shape[0]].copy_(tensor)
        return padded

    def _gather_dpa_inputs(
        self, rows: tuple[torch.Tensor, ...]
    ) -> tuple[tuple[torch.Tensor, ...], int, list[int] | None]:
        """Gather DPA inputs -- a rank's hidden states with their top-k weights
        and ids -- in compact rank order when Window supports ragged M."""
        from atom.utils.forward_context import get_forward_context

        ctx = get_forward_context()
        local_tokens = int(rows[0].shape[0])
        group = get_dp_group()
        world_size = int(group.world_size)
        dp_metadata = ctx.dp_metadata
        max_tokens = self._dpa_max_tokens(ctx)
        if max_tokens is None:
            raise RuntimeError(
                "Communication-fused DPA requires dp_metadata or DP-uniform "
                "running_tokens"
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
        else:
            # Window RS assigns output row ``rank * (bucket / world_size) + row``
            # to each rank.  Padding only the gathered tensor's global tail would
            # leave the real rank-major boundaries at ``rank * max_tokens`` and
            # shift every rank after rank 0 whenever the bucket is larger.
            # Insert the bucket slack inside every rank's shard before the
            # gather instead.
            fills = (0, 0, -1)  # hidden, weights, ids
            rows = tuple(self._pad_rows(t, shard_rows, f) for t, f in zip(rows, fills))
            sizes = [shard_rows] * world_size
        # One grouped collective for every payload, each kept separate.
        rows = gather_rows(
            lambda ts: group.device_communicator.all_gatherv(ts, dim=0, sizes=sizes),
            rows,
        )
        return rows, local_tokens, sizes if use_ragged_m else None

    def forward(
        self,
        layer: FusedMoE,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        shared_partial: torch.Tensor | None,
        before_stage2_for_rows: Callable[[int], torch.Tensor] | None = None,
        before_shared_add: Callable[[], None] | None = None,
        stage2_stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        if before_stage2_for_rows is not None:
            return self.forward_impl(
                layer,
                hidden_states,
                router_logits,
                shared_partial,
                before_stage2_for_rows=before_stage2_for_rows,
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
        before_stage2_for_rows: Callable[[int], torch.Tensor] | None = None,
        before_shared_add: Callable[[], None] | None = None,
        stage2_stream: torch.cuda.Stream | None = None,
    ) -> torch.Tensor:
        method = layer.quant_method
        reduce_scatter_sizes = None
        # Routed on the rank that owns the rows, before the gather (as in
        # FusedMoE.forward_impl_graph).
        topk_weights, topk_ids = method.select_experts(
            layer, hidden_states, router_logits
        )
        if self.use_dp_reduce_scatter:
            (
                (hidden_states, topk_weights, topk_ids),
                local_tokens,
                reduce_scatter_sizes,
            ) = self._gather_dpa_inputs((hidden_states, topk_weights, topk_ids))
            reassign_gathered_shared_ids(layer, topk_ids)
            if reduce_scatter_sizes is None:
                shard_rows = hidden_states.shape[0] // int(get_dp_group().world_size)
                if shared_partial is not None:
                    shared_partial = self._pad_rows(shared_partial, shard_rows)
        # On the physical experts after the gather, as FusedMoE._run_experts.
        topk_weights, topk_ids = method.place_experts(
            layer, topk_weights, topk_ids, self.use_dp_reduce_scatter
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
            before_stage2_for_rows=before_stage2_for_rows,
            before_shared_add=before_shared_add,
            stage2_stream=stage2_stream,
            reduce_scatter_sizes=reduce_scatter_sizes,
            # Routing sends the rows past `scheduled_tokens_gpu` to expert -1.
            routes_may_be_missing=True,
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
