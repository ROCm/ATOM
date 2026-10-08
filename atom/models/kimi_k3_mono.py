# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Kimi-K3 C1 fused MoE tail adapter."""

from __future__ import annotations

import logging

import torch
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)

from atom.model_ops.monokernel.config import (
    KIMI_K3_CONFIG,
    Mxfp4ScaleLayout,
    Mxfp4WeightLayout,
)
from atom.model_ops.monokernel.dispatch import (
    MonoUnsupported,
    tp_uniform_local_validation,
)
from atom.model_ops.monokernel.k3.prepared import (
    KimiK3PreparedTailWeights,
    prepare_kimi_k3_tail_weights,
)
from atom.model_ops.monokernel.weights import (
    LayerWeights,
    atom_mxfp4_storage_view,
    linear_bf16,
)
from atom.plugin.prepare import is_plugin_mode
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")
_C1_ROWS = 8


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


def _tail_weights(layer, rank: int, npes: int) -> LayerWeights:
    moe = layer.block_sparse_moe
    experts = moe.experts
    cfg = KIMI_K3_CONFIG
    _need(not experts.quant_method.is_guinterleave, "ATOM_MOE_GU_ITLV must be 0")
    _need(experts.global_num_experts == cfg.n_experts, "expert count")
    _need(experts.intermediate_size_per_partition == cfg.inter, "expert width")

    w_ug = experts.w13_weight
    s_ug = experts.w13_weight_scale
    w_dn = experts.w2_weight
    s_dn = experts.w2_weight_scale
    atom_mxfp4_storage_view(
        w_ug,
        name="w_ug",
        logical_rows=cfg.n_experts * 2 * cfg.inter,
        logical_k=cfg.routed_hidden,
        scale=False,
    )
    atom_mxfp4_storage_view(
        s_ug,
        name="s_ug",
        logical_rows=cfg.n_experts * 2 * cfg.inter,
        logical_k=cfg.routed_hidden,
        scale=True,
    )
    atom_mxfp4_storage_view(
        w_dn,
        name="w_dn",
        logical_rows=cfg.n_experts * cfg.routed_hidden,
        logical_k=cfg.inter,
        scale=False,
    )
    atom_mxfp4_storage_view(
        s_dn,
        name="s_dn",
        logical_rows=cfg.n_experts * cfg.routed_hidden,
        logical_k=cfg.inter,
        scale=True,
    )
    shard = cfg.hidden // npes
    tensors = {
        "w_r": linear_bf16(
            moe.gate,
            name="gate",
            logical_rows=cfg.n_experts,
            logical_cols=cfg.hidden,
        ),
        "bias": moe.gate.e_score_correction_bias,
        "w_latent_down": linear_bf16(
            moe.routed_expert_down_proj,
            name="routed_expert_down_proj",
            logical_rows=cfg.routed_hidden,
            logical_cols=cfg.hidden,
        ),
        "g_latent": moe.routed_expert_norm.weight,
        "w_latent_up": linear_bf16(
            moe.routed_expert_up_proj,
            name="routed_expert_up_proj",
            logical_rows=cfg.hidden,
            logical_cols=cfg.routed_hidden,
            row_start=rank * shard,
            row_count=shard,
        ),
        "w_shared_ug": linear_bf16(
            moe.shared_experts.gate_up_proj,
            name="shared_experts.gate_up_proj",
            logical_rows=2 * cfg.shared_inter,
            logical_cols=cfg.hidden,
        ),
        "w_shared_dn": linear_bf16(
            moe.shared_experts.down_proj,
            name="shared_experts.down_proj",
            logical_rows=cfg.hidden,
            logical_cols=cfg.shared_inter,
        ),
        "w_ug": w_ug,
        "s_ug": s_ug,
        "w_dn": w_dn,
        "s_dn": s_dn,
    }
    return LayerWeights(
        cfg.local_heads,
        tensors,
        cfg,
        rank,
        npes,
        mxfp4_weight_layout=Mxfp4WeightLayout.ATOM,
        mxfp4_scale_layout=Mxfp4ScaleLayout.ATOM,
    )


def _run_layer_hooks(layer, args: tuple, kwargs: dict, output: tuple) -> tuple:
    """Publish fused-tail results through the layer-local DSpark hook ABI."""

    hooks_with_kwargs = getattr(layer, "_forward_hooks_with_kwargs", {})
    for hook_id, hook in tuple(getattr(layer, "_forward_hooks", {}).items()):
        if hook_id in hooks_with_kwargs:
            result = hook(layer, args, kwargs, output)
        else:
            result = hook(layer, args, output)
        if result is not None:
            output = result
    if not isinstance(output, tuple) or len(output) != 4:
        raise RuntimeError("a Kimi decoder forward hook changed the layer output ABI")
    return output


class _KimiLayerOp:
    def __init__(
        self,
        layer,
        weights: LayerWeights,
        prepared_weights: KimiK3PreparedTailWeights,
    ) -> None:
        from atom.model_ops.monokernel.k3.staged import _KimiK3FusedTail

        rank = get_tensor_model_parallel_rank()
        npes = get_tensor_model_parallel_world_size()
        self.op = _KimiK3FusedTail(
            weights,
            _C1_ROWS,
            layer_idx=layer.layer_idx,
            rank=rank,
            npes=npes,
            group=get_tp_group().cpu_group,
            prepared_weights=prepared_weights,
        )

    def close(self) -> None:
        self.op.close()


def validate_kimi_c1_config(atom_config) -> None:
    speculative = atom_config.speculative_config
    if (
        speculative is None
        or getattr(speculative, "method", None) != "dspark"
        or getattr(speculative, "num_speculative_tokens", None) != _C1_ROWS - 1
    ):
        raise MonoUnsupported("Kimi staged_c1 requires DSpark7")
    if atom_config.decode_context_parallel_size != 1:
        raise MonoUnsupported("Kimi staged_c1 requires DCP1")
    if envs.ATOM_ENABLE_REPLAYSSM is not False:
        raise MonoUnsupported("Kimi staged_c1 requires ATOM_ENABLE_REPLAYSSM=0")


class KimiStagedC1Decode:
    """Replace only the measured Kimi-K3 C1 MoE tail."""

    def __init__(self, causal_lm, atom_config, mode: str) -> None:
        if mode not in {"off", "staged_c1"}:
            raise ValueError(
                f"Kimi fused-tail mode must be off or staged_c1, got {mode!r}"
            )
        self._lm = causal_lm
        self._ops: dict[int, _KimiLayerOp] = {}
        self._weights: dict[int, LayerWeights] = {}
        self._prepared: dict[int, KimiK3PreparedTailWeights] = {}
        self._outputs: dict[int, torch.Tensor] = {}
        self._refused: set[int] = set()
        self._prebuilt = False
        self._enabled = mode == "staged_c1"
        self._announced = False
        if not self._enabled:
            return

        checks = (
            (atom_config.tensor_parallel_size == 8, "not TP8"),
            (atom_config.parallel_config.data_parallel_size == 1, "DP"),
            (not atom_config.enable_dp_attention, "DPA"),
            (atom_config.pipeline_parallel_size == 1, "PP"),
            (not is_plugin_mode(), "plugin mode"),
            (atom_config.kv_cache_dtype == "fp8", "not FP8 KV"),
        )
        for ok, why in checks:
            if not ok:
                logger.info("Kimi-K3 staged_c1 off: %s", why)
                self._enabled = False
                return
        try:
            validate_kimi_c1_config(atom_config)
        except MonoUnsupported as error:
            logger.info("Kimi-K3 staged_c1 off: %s", error)
            self._enabled = False

    @staticmethod
    def _metadata(fwd):
        metadata = getattr(fwd.attn_metadata, "kda_metadata", None)
        if metadata is None:
            metadata = getattr(fwd.attn_metadata, "gdn_metadata", None)
        return metadata

    @staticmethod
    def _metadata_is_c1(metadata) -> bool:
        if (
            metadata is None
            or metadata.num_prefills != 0
            or metadata.num_spec_decodes != 1
            or not 0 < metadata.num_actual_tokens <= _C1_ROWS
            or getattr(metadata, "replayssm", False)
        ):
            return False
        slots = metadata.spec_state_indices_tensor
        accepted = metadata.num_accepted_tokens
        return (
            slots is not None
            and slots.dtype is torch.int32
            and slots.is_contiguous()
            and slots.ndim == 2
            and slots.shape[0] >= 1
            and slots.shape[1] == _C1_ROWS
            and accepted is not None
            and accepted.dtype is torch.int32
            and accepted.is_contiguous()
            and accepted.numel() >= 1
        )

    def supports(
        self, input_ids, positions, intermediate_tensors, inputs_embeds
    ) -> bool:
        if (
            not self._enabled
            or intermediate_tensors is not None
            or input_ids.numel() != _C1_ROWS
            or positions.numel() != _C1_ROWS
        ):
            return False
        if inputs_embeds is not None and (
            inputs_embeds.shape != (_C1_ROWS, KIMI_K3_CONFIG.hidden)
            or inputs_embeds.dtype != torch.bfloat16
            or not inputs_embeds.is_contiguous()
        ):
            return False
        fwd = get_forward_context()
        return (
            fwd.context is not None
            and not fwd.context.is_prefill
            and fwd.ubatch_slices is None
            and self._metadata_is_c1(self._metadata(fwd))
        )

    def _op(self, layer) -> _KimiLayerOp:
        layer_idx = layer.layer_idx
        if layer_idx in self._refused:
            raise RuntimeError(f"layer {layer_idx} fused-tail construction was refused")
        if layer_idx in self._ops:
            return self._ops[layer_idx]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("cannot construct Kimi fused tails during graph capture")

        rank = get_tensor_model_parallel_rank()
        npes = get_tensor_model_parallel_world_size()
        tp = get_tp_group()
        weights = self._weights.get(layer_idx)
        validation_error = None
        if weights is None:
            try:
                weights = _tail_weights(layer, rank, npes)
            except (MonoUnsupported, ValueError) as error:
                validation_error = error
        try:
            tp_uniform_local_validation(
                validation_error,
                group=tp.cpu_group,
                world_size=npes,
                context=f"layer {layer_idx} tail weight mapping failed",
            )
            assert weights is not None
            self._weights[layer_idx] = weights
            prepared = self._prepared.get(layer_idx)
            preparation_error = None
            if prepared is None:
                try:
                    prepared = prepare_kimi_k3_tail_weights(weights)
                except (MonoUnsupported, ValueError) as error:
                    preparation_error = error
                tp_uniform_local_validation(
                    preparation_error,
                    group=tp.cpu_group,
                    world_size=npes,
                    context=f"layer {layer_idx} tail weight preparation failed",
                )
                assert prepared is not None
                self._prepared[layer_idx] = prepared
            else:
                prepared.validate_source(weights, "tail")
            owned = _KimiLayerOp(layer, weights, prepared)
            self._ops[layer_idx] = owned
            if rank == 0 and not self._announced:
                logger.info("Kimi-K3 staged_c1 fused MoE tail on")
                self._announced = True
            return owned
        except (MonoUnsupported, ValueError) as error:
            self._refused.add(layer_idx)
            raise RuntimeError(
                f"Kimi layer {layer_idx} cannot use staged_c1: {error}"
            ) from error

    def prepare(self) -> None:
        """Construct static tail weights before KV memory is budgeted."""

        if not self._enabled or self._prebuilt:
            return
        model = self._lm.model
        for layer in model.layers[model.start_layer : model.end_layer]:
            if hasattr(layer, "block_sparse_moe"):
                self._op(layer)
        self._prebuilt = True

    @staticmethod
    def _require_blocks(blocks: torch.Tensor | None) -> torch.Tensor:
        if blocks is None:
            raise RuntimeError("Kimi staged_c1 requires attention-residual blocks")
        return blocks

    def _output(self, layer_idx: int, hidden: torch.Tensor) -> torch.Tensor:
        output = self._outputs.get(layer_idx)
        if output is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("cannot allocate Kimi output during graph capture")
            output = torch.empty_like(hidden)
            self._outputs[layer_idx] = output
        return output

    def _run_tail(
        self,
        layer,
        updated_prefix: torch.Tensor,
        moe_input: torch.Tensor,
    ) -> torch.Tensor:
        output = self._output(layer.layer_idx, moe_input)
        self._op(layer).op.forward_from_moe_input(
            updated_prefix,
            moe_input,
            x_out=output,
            epoch_layer=layer.layer_idx,
        )
        return output

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model = self._lm.model
        hidden = (
            model.get_input_embeddings(input_ids)
            if inputs_embeds is None
            else inputs_embeds
        )
        blocks = hidden.new_zeros(_C1_ROWS, 0, hidden.shape[-1])
        pending = pending2 = None

        for layer in model.layers[model.start_layer : model.end_layer]:
            if not hasattr(layer, "block_sparse_moe"):
                hidden, pending, pending2, blocks = layer(
                    positions,
                    hidden,
                    blocks,
                    pending_add=pending,
                    pending_add2=pending2,
                )
                continue

            hook_input = hidden
            hook_kwargs = {"pending_add": pending, "pending_add2": pending2}
            attention_input, prefix_sum = layer.self_attention_attn_res(
                hidden, blocks, pending, pending2
            )
            blocks, prefix_sum = layer.self_attention_attn_res.maybe_close_block(
                prefix_sum, blocks
            )
            blocks = self._require_blocks(blocks)
            attention_delta = (
                layer.self_attn(attention_input)
                if layer.is_linear_attn
                else layer.self_attn(positions, attention_input)
            )
            if attention_delta.shape != hidden.shape:
                raise RuntimeError(
                    f"attention layer {layer.layer_idx} changed graph rows from "
                    f"{hidden.shape[0]} to {attention_delta.shape[0]}"
                )
            moe_input, updated_prefix = layer.mlp_attn_res(
                prefix_sum, blocks, attention_delta
            )
            hidden = self._run_tail(layer, updated_prefix, moe_input)
            pending = pending2 = None
            hidden, pending, pending2, blocks = _run_layer_hooks(
                layer,
                (positions, hook_input, blocks),
                hook_kwargs,
                (hidden, pending, pending2, blocks),
            )

        hidden, _ = model.output_attn_res(hidden, blocks, pending, pending2)
        return hidden

    def close(self) -> None:
        for owned in self._ops.values():
            owned.close()
        self._ops.clear()
        self._prepared.clear()
        self._weights.clear()
        self._outputs.clear()
        self._refused.clear()
        self._announced = False
        self._prebuilt = False
