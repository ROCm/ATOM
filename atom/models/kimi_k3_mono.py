# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Native Kimi-K3 AgentX decode using ATOM-owned MonoKernel sources."""

from __future__ import annotations

import logging

import torch
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)

from atom.model_ops.attention_residual_contract import resolve_attn_res_prefix
from atom.model_ops.monokernel.config import (
    KIMI_K3_CONFIG,
    ConvStateLayout,
    KimiDecodeGeometry,
    Mxfp4ScaleLayout,
    Mxfp4WeightLayout,
)
from atom.model_ops.monokernel.dispatch import (
    MonoUnsupported,
    select_backend,
    tp_uniform_local_validation,
)
from atom.model_ops.monokernel.k3.prepared import (
    KimiK3PreparedTailWeights,
    KimiK3PreparedWeights,
    prepare_kimi_k3_tail_weights,
    prepare_kimi_k3_weights,
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
_MAX_LAUNCH_WIDTH = 8


def _need(ok: bool, what: str) -> None:
    if not ok:
        raise MonoUnsupported(what)


def fold_kimi_pending(
    prefix_sum: torch.Tensor | None,
    pending_add: torch.Tensor | None,
    pending_add2: torch.Tensor | None,
) -> torch.Tensor:
    """Materialize the baseline layer return protocol before a native layer."""

    prefix_sum, pending_add, pending_add2 = resolve_attn_res_prefix(
        prefix_sum, pending_add, pending_add2
    )
    if pending_add is not None:
        prefix_sum = prefix_sum + pending_add
        if pending_add2 is not None:
            prefix_sum = prefix_sum + pending_add2
    return prefix_sum


def build_kimi_state_chains(
    output: torch.Tensor,
    resume_columns: torch.Tensor,
    spec_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor,
) -> torch.Tensor:
    """Build ``resume + q outputs`` for every request without a host read."""

    groups, q = spec_state_indices.shape
    if output.shape != (groups, q + 1) or output.dtype is not torch.int32:
        raise ValueError(f"output must be int32 [{groups}, {q + 1}]")
    if resume_columns.shape != (groups,) or resume_columns.dtype is not torch.int64:
        raise ValueError(f"resume_columns must be int64 [{groups}]")
    if (
        spec_state_indices.dtype is not torch.int32
        or not spec_state_indices.is_contiguous()
    ):
        raise ValueError("spec_state_indices must be contiguous int32")
    if (
        num_accepted_tokens.shape != (groups,)
        or num_accepted_tokens.dtype is not torch.int32
        or not num_accepted_tokens.is_contiguous()
    ):
        raise ValueError(f"num_accepted_tokens must be contiguous int32 [{groups}]")
    if any(
        tensor.device != spec_state_indices.device
        for tensor in (output, resume_columns, num_accepted_tokens)
    ):
        raise ValueError("state-chain tensors must share one device")

    output[:, 1:].copy_(spec_state_indices)
    resume_columns.copy_(num_accepted_tokens)
    resume_columns.sub_(1)
    torch.gather(
        spec_state_indices,
        1,
        resume_columns.view(groups, 1),
        out=output[:, :1],
    )
    return output


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
        "g_in": layer.input_layernorm.weight,
        "g_post": layer.post_attention_layernorm.weight,
        "g_self_res": layer.self_attention_res_norm.weight,
        "w_self_res": linear_bf16(
            layer.self_attention_res_proj,
            name="self_attention_res_proj",
            logical_rows=1,
            logical_cols=cfg.hidden,
        ).view(-1),
        "g_mlp_res": layer.mlp_res_norm.weight,
        "w_mlp_res": linear_bf16(
            layer.mlp_res_proj,
            name="mlp_res_proj",
            logical_rows=1,
            logical_cols=cfg.hidden,
        ).view(-1),
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


def _layer_weights(layer, rank: int, npes: int) -> LayerWeights:
    _need(layer.is_linear_attn, f"layer {layer.layer_idx}: not KDA")
    weights = _tail_weights(layer, rank, npes)
    attn = layer.self_attn
    cfg = KIMI_K3_CONFIG
    weights.t.update(
        {
            "w_kda_in": linear_bf16(
                attn.in_proj,
                name="in_proj",
                logical_rows=(
                    4 * cfg.local_heads * cfg.v_dim + cfg.local_heads + cfg.v_dim
                ),
                logical_cols=cfg.hidden,
            ),
            "w_kda_fb": linear_bf16(
                attn.f_b_proj,
                name="f_b_proj",
                logical_rows=cfg.local_heads * cfg.v_dim,
                logical_cols=cfg.v_dim,
            ),
            "w_kda_conv": attn.conv_weight,
            "kda_a_log": attn.A_log.float().contiguous(),
            "kda_dt_bias": attn.dt_bias.view(cfg.local_heads, cfg.v_dim),
            "g_kda_out": attn.o_norm.weight,
            "w_kda_o": linear_bf16(
                attn.o_proj,
                name="o_proj",
                logical_rows=cfg.hidden,
                logical_cols=cfg.local_heads * cfg.v_dim,
            ),
        }
    )
    return weights


def _run_layer_hooks(layer, args: tuple, kwargs: dict, output: tuple) -> tuple:
    """Publish native results through the layer-local hook ABI used by DSpark."""

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
        geometry: KimiDecodeGeometry,
        backend: str,
        kind: str,
        launch_width: int,
    ) -> None:
        rank = get_tensor_model_parallel_rank()
        npes = get_tensor_model_parallel_world_size()
        tp = get_tp_group()
        common = {
            "layer_idx": layer.layer_idx,
            "rank": rank,
            "npes": npes,
            "group": tp.cpu_group,
            "reduce_group": tp.device_group,
            "mtp": geometry.q > 1,
            "conv_state_layout": ConvStateLayout.TIME_MAJOR,
            "conv_state_rows": geometry.conv_state_rows,
            "state_dtype": torch.float16,
        }
        if kind == "kda":
            if not isinstance(prepared_weights, KimiK3PreparedWeights):
                raise TypeError("KDA launches require KDA prepared weights")
            if backend == "mono":
                from atom.model_ops.monokernel.k3.op import KimiK3MonoKernel

                op_type = KimiK3MonoKernel
            else:
                from atom.model_ops.monokernel.k3.staged import _KimiK3KdaStagedPath

                op_type = _KimiK3KdaStagedPath
            self.op = op_type(
                weights,
                launch_width,
                prepared_weights=prepared_weights,
                **common,
            )
        else:
            from atom.model_ops.monokernel.k3.staged import _KimiK3FusedTail

            self.op = _KimiK3FusedTail(
                weights,
                launch_width,
                prepared_weights=prepared_weights,
                prepared_backend="tail",
                **common,
            )
        self.launch_width = launch_width
        self.kind = kind

    def close(self) -> None:
        self.op.close()


def resolve_kimi_decode_geometry(atom_config) -> KimiDecodeGeometry:
    """Resolve one construction-time geometry from the published recipe flags."""

    speculative = atom_config.speculative_config
    if speculative is None:
        q = 1
    else:
        if getattr(speculative, "method", None) != "dspark":
            raise MonoUnsupported("speculative method is not DSpark")
        num_spec = getattr(speculative, "num_speculative_tokens", None)
        if num_spec not in {3, 7}:
            raise MonoUnsupported(f"unsupported DSpark width {num_spec}")
        q = num_spec + 1
    replay_override = envs.ATOM_ENABLE_REPLAYSSM
    replay_mode = (q > 1) if replay_override is None else replay_override
    dcp = atom_config.decode_context_parallel_size
    supported = q == 8 and dcp == 1 and not replay_mode
    if not supported:
        raise MonoUnsupported(
            f"unsupported Kimi geometry q={q}, DCP={dcp}, ReplaySSM={int(replay_mode)}"
        )
    return KimiDecodeGeometry(
        groups=1,
        q=q,
        conv_state_rows=q + 2,
        state_dtype="fp16",
        replay_mode=replay_mode,
    )


def kimi_launch_slices(geometry: KimiDecodeGeometry) -> tuple[tuple[int, int], ...]:
    """Partition flattened rows into q-chains or independent decode chunks."""

    if geometry.q > 1:
        return tuple(
            (group * geometry.q, (group + 1) * geometry.q)
            for group in range(geometry.groups)
        )
    slices = []
    start = 0
    remaining = geometry.groups
    for width in (_MAX_LAUNCH_WIDTH, 4, 2, 1):
        while remaining >= width:
            slices.append((start, start + width))
            start += width
            remaining -= width
    return tuple(slices)


class KimiMonoDecode:
    """Parameterize native Kimi decode across the published AgentX bands."""

    def __init__(self, causal_lm, atom_config, mode: str) -> None:
        self._lm = causal_lm
        self._atom_config = atom_config
        self._mode = mode
        self._ops: dict[tuple[int, int, str, str, bool], _KimiLayerOp] = {}
        self._weights: dict[tuple[int, str], LayerWeights] = {}
        self._prepared: dict[tuple[int, str, bool], KimiK3PreparedTailWeights] = {}
        self._outputs: dict[tuple[int, int, str], torch.Tensor] = {}
        self._chains: dict[tuple[int, int], tuple[torch.Tensor, torch.Tensor]] = {}
        self._refused: set[tuple[int, int, str, str, bool]] = set()
        self._announced: set[tuple[str, str, int]] = set()
        self._prebuilt = False
        self._geometry_template: KimiDecodeGeometry | None = None
        self._enabled = mode != "off"
        self._staged_c1_enabled = mode == "staged_c1"
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
                logger.info("Kimi-K3 MonoKernel off: %s", why)
                self._enabled = False
                return
        try:
            self._geometry_template = resolve_kimi_decode_geometry(atom_config)
        except MonoUnsupported as error:
            logger.info("Kimi-K3 MonoKernel off: %s", error)
            self._enabled = False

    @staticmethod
    def _metadata(fwd):
        metadata = getattr(fwd.attn_metadata, "kda_metadata", None)
        if metadata is None:
            metadata = getattr(fwd.attn_metadata, "gdn_metadata", None)
        return metadata

    def _geometry(self, samples: int, metadata) -> KimiDecodeGeometry | None:
        template = self._geometry_template
        if (
            template is None
            or metadata is None
            or metadata.num_prefills != 0
            or getattr(metadata, "replayssm", False) != template.replay_mode
            or not 0 < metadata.num_actual_tokens <= samples
        ):
            return None
        if template.q > 1:
            slots = metadata.spec_state_indices_tensor
            accepted = metadata.num_accepted_tokens
            if (
                metadata.num_spec_decodes <= 0
                or slots is None
                or slots.ndim != 2
                or slots.shape[1] != template.q
            ):
                raise RuntimeError(
                    f"Kimi q={template.q} requires a matching DSpark state table"
                )
            if samples % template.q:
                raise RuntimeError(
                    f"Kimi token rows {samples} are not divisible by q={template.q}"
                )
            groups = samples // template.q
            if slots.shape[0] < groups or accepted is None or accepted.numel() < groups:
                raise RuntimeError(
                    "Kimi speculative metadata does not cover the graph batch"
                )
        else:
            slots = metadata.non_spec_state_indices_tensor
            if metadata.num_decodes <= 0 or slots is None or slots.numel() < samples:
                raise RuntimeError("Kimi decode slots do not cover the graph batch")
            groups = samples
        return KimiDecodeGeometry(
            groups=groups,
            q=template.q,
            conv_state_rows=template.conv_state_rows,
            state_dtype=template.state_dtype,
            replay_mode=template.replay_mode,
        )

    def supports(
        self, input_ids, positions, intermediate_tensors, inputs_embeds
    ) -> bool:
        if not self._enabled or intermediate_tensors is not None:
            return False
        samples = input_ids.numel()
        if inputs_embeds is not None and (
            inputs_embeds.shape != (samples, KIMI_K3_CONFIG.hidden)
            or inputs_embeds.dtype != torch.bfloat16
            or not inputs_embeds.is_contiguous()
        ):
            return False
        fwd = get_forward_context()
        if (
            fwd.context is None
            or fwd.context.is_prefill
            or fwd.ubatch_slices is not None
        ):
            return False
        geometry = self._geometry(samples, self._metadata(fwd))
        return geometry is not None and positions.numel() == samples

    def _spec_chains(self, geometry: KimiDecodeGeometry, metadata) -> torch.Tensor:
        key = (geometry.groups, geometry.q)
        buffers = self._chains.get(key)
        if buffers is None:
            if torch.cuda.is_current_stream_capturing():
                raise MonoUnsupported(
                    "cannot allocate state chains during graph capture"
                )
            slots = metadata.spec_state_indices_tensor
            buffers = (
                torch.empty(
                    geometry.groups,
                    geometry.q + 1,
                    dtype=torch.int32,
                    device=slots.device,
                ),
                torch.empty(
                    geometry.groups,
                    dtype=torch.int64,
                    device=slots.device,
                ),
            )
            self._chains[key] = buffers
        chains, resume_columns = buffers
        return build_kimi_state_chains(
            chains,
            resume_columns,
            metadata.spec_state_indices_tensor[: geometry.groups],
            metadata.num_accepted_tokens[: geometry.groups],
        )

    def _op(
        self,
        layer,
        geometry: KimiDecodeGeometry,
        kind: str,
        launch_width: int,
    ) -> _KimiLayerOp:
        backend = "tail"
        if kind == "kda":
            selected = select_backend(
                "kimi_k3",
                self._mode,
                samples=launch_width,
                tp_size=8,
                kv_cache_dtype=self._atom_config.kv_cache_dtype,
                mtp=geometry.q > 1,
                q=geometry.q,
                is_kda=True,
                has_moe=True,
            )
            if selected is None:
                raise RuntimeError("expected Kimi KDA backend is unavailable")
            backend = selected
        key = (layer.layer_idx, launch_width, backend, kind, geometry.q > 1)
        if key in self._refused:
            raise RuntimeError(
                f"layer {layer.layer_idx} native construction was refused"
            )
        if key in self._ops:
            return self._ops[key]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "cannot construct Kimi native layers during graph capture"
            )

        rank = get_tensor_model_parallel_rank()
        npes = get_tensor_model_parallel_world_size()
        tp = get_tp_group()
        weight_key = (layer.layer_idx, kind)
        weights = self._weights.get(weight_key)
        validation_error = None
        if weights is None:
            try:
                mapper = _layer_weights if kind == "kda" else _tail_weights
                weights = mapper(layer, rank, npes)
            except (MonoUnsupported, ValueError) as error:
                validation_error = error
        try:
            tp_uniform_local_validation(
                validation_error,
                group=tp.cpu_group,
                world_size=npes,
                context=f"layer {layer.layer_idx} weight mapping failed",
            )
            assert weights is not None
            self._weights[weight_key] = weights
            prepared_key = (layer.layer_idx, backend, geometry.q > 1)
            prepared = self._prepared.get(prepared_key)
            preparation_error = None
            if prepared is None:
                try:
                    if kind == "kda":
                        prepared = prepare_kimi_k3_weights(
                            weights,
                            backend,
                            mtp=geometry.q > 1,
                        )
                    else:
                        prepared = prepare_kimi_k3_tail_weights(weights)
                except (MonoUnsupported, ValueError) as error:
                    preparation_error = error
                tp_uniform_local_validation(
                    preparation_error,
                    group=tp.cpu_group,
                    world_size=npes,
                    context=f"layer {layer.layer_idx} weight preparation failed",
                )
                assert prepared is not None
                self._prepared[prepared_key] = prepared
            else:
                prepared.validate_source(weights, backend)
            owned = _KimiLayerOp(
                layer,
                weights,
                prepared,
                geometry,
                backend,
                kind,
                launch_width,
            )
            self._ops[key] = owned
            announcement = (kind, backend, geometry.q)
            if rank == 0 and announcement not in self._announced:
                logger.info(
                    "Kimi-K3 MonoKernel on: path=%s backend=%s q=%d",
                    kind,
                    backend,
                    geometry.q,
                )
                self._announced.add(announcement)
            return owned
        except (MonoUnsupported, ValueError) as error:
            self._refused.add(key)
            raise RuntimeError(
                f"Kimi layer {layer.layer_idx} cannot use {kind}: {error}"
            ) from error

    @staticmethod
    def _launch_widths(geometry: KimiDecodeGeometry) -> tuple[int, ...]:
        return (geometry.q,) if geometry.q > 1 else (_MAX_LAUNCH_WIDTH, 4, 2, 1)

    def prepare(self) -> None:
        """Construct static native weights before KV memory is budgeted."""

        if not self._enabled or self._prebuilt:
            return
        geometry = self._geometry_template
        assert geometry is not None
        model = self._lm.model
        for layer in model.layers[model.start_layer : model.end_layer]:
            if not hasattr(layer, "block_sparse_moe"):
                continue
            if getattr(self, "_staged_c1_enabled", False):
                kind = "tail"
            else:
                kind = (
                    "tail"
                    if (not layer.is_linear_attn or geometry.replay_mode)
                    else "kda"
                )
            for launch_width in self._launch_widths(geometry):
                self._op(layer, geometry, kind, launch_width)
        self._prebuilt = True

    @staticmethod
    def _require_blocks(blocks: torch.Tensor | None) -> torch.Tensor:
        if blocks is None:
            raise RuntimeError("Kimi native layers require attention-residual blocks")
        return blocks

    @staticmethod
    def _ensure_block_capacity(
        hidden: torch.Tensor,
        blocks: torch.Tensor,
        block_index: int,
    ) -> torch.Tensor:
        if block_index < blocks.shape[1]:
            return blocks
        extra = hidden.new_zeros(
            hidden.shape[0],
            block_index + 1 - blocks.shape[1],
            hidden.shape[1],
        )
        return torch.cat((blocks, extra), dim=1)

    def _output(
        self,
        layer_idx: int,
        geometry: KimiDecodeGeometry,
        kind: str,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        key = (layer_idx, geometry.tokens, kind)
        output = self._outputs.get(key)
        if output is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "cannot allocate grouped output during graph capture"
                )
            output = torch.empty_like(hidden)
            self._outputs[key] = output
        return output

    def _run_kda(
        self,
        layer,
        geometry: KimiDecodeGeometry,
        hidden: torch.Tensor,
        blocks: torch.Tensor,
        cache,
        state_indices: torch.Tensor,
        accepted: torch.Tensor | None,
    ) -> torch.Tensor:
        output = self._output(layer.layer_idx, geometry, "kda", hidden)
        for start, end in kimi_launch_slices(geometry):
            launch_width = end - start
            owned = self._op(layer, geometry, "kda", launch_width)
            if geometry.q > 1:
                group = start // geometry.q
                indices = state_indices[group]
                accepted_row = accepted[group : group + 1]
            else:
                indices = state_indices[start:end]
                accepted_row = None
            owned.op.forward(
                hidden[start:end],
                blocks[start:end],
                indices,
                cache.k_cache,
                cache.v_cache,
                num_accepted_tokens=accepted_row,
                x_out=output[start:end],
                epoch_layer=layer.layer_idx,
            )
        return output

    def _run_tail(
        self,
        layer,
        geometry: KimiDecodeGeometry,
        prefix_sum: torch.Tensor | None,
        blocks: torch.Tensor,
        attention_delta: torch.Tensor,
    ) -> torch.Tensor:
        output = self._output(layer.layer_idx, geometry, "tail", attention_delta)
        for start, end in kimi_launch_slices(geometry):
            launch_width = end - start
            owned = self._op(layer, geometry, "tail", launch_width)
            prefix = None if prefix_sum is None else prefix_sum[start:end]
            owned.op.forward_from_attention(
                prefix,
                blocks[start:end],
                attention_delta[start:end],
                x_out=output[start:end],
                epoch_layer=layer.layer_idx,
            )
        return output

    def _run_staged_c1_tail(
        self,
        layer,
        geometry: KimiDecodeGeometry,
        updated_prefix: torch.Tensor,
        moe_input: torch.Tensor,
    ) -> torch.Tensor:
        output = self._output(layer.layer_idx, geometry, "tail", moe_input)
        for start, end in kimi_launch_slices(geometry):
            owned = self._op(layer, geometry, "tail", end - start)
            owned.op.forward_from_moe_input(
                updated_prefix[start:end],
                moe_input[start:end],
                x_out=output[start:end],
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
        fwd = get_forward_context()
        metadata = self._metadata(fwd)
        samples = input_ids.numel()
        geometry = self._geometry(samples, metadata)
        if geometry is None:
            raise RuntimeError("Kimi native forward lost its decode geometry")
        hidden = (
            model.get_input_embeddings(input_ids)
            if inputs_embeds is None
            else inputs_embeds
        )
        staged_c1_enabled = getattr(self, "_staged_c1_enabled", False)
        blocks = hidden.new_zeros(samples, 0, hidden.shape[-1])
        pending = pending2 = None
        state_indices = None
        accepted = None
        if not geometry.replay_mode:
            state_indices = (
                self._spec_chains(geometry, metadata)
                if geometry.q > 1
                else metadata.non_spec_state_indices_tensor[:samples]
            )
            if geometry.q > 1:
                accepted = metadata.num_accepted_tokens[: geometry.groups]

        for layer in model.layers[model.start_layer : model.end_layer]:
            has_moe = hasattr(layer, "block_sparse_moe")
            if not has_moe:
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
            baseline_attention = (
                not layer.is_linear_attn or geometry.replay_mode or staged_c1_enabled
            )
            if baseline_attention:
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
                if staged_c1_enabled:
                    moe_input, updated_prefix = layer.mlp_attn_res(
                        prefix_sum, blocks, attention_delta
                    )
                    hidden = self._run_staged_c1_tail(
                        layer, geometry, updated_prefix, moe_input
                    )
                else:
                    hidden = self._run_tail(
                        layer,
                        geometry,
                        prefix_sum,
                        blocks,
                        attention_delta,
                    )
            else:
                hidden = fold_kimi_pending(hidden, pending, pending2)
                pending = pending2 = None
                blocks = self._require_blocks(blocks)
                first_start, first_end = kimi_launch_slices(geometry)[0]
                first = self._op(
                    layer,
                    geometry,
                    "kda",
                    first_end - first_start,
                )
                blocks = self._ensure_block_capacity(
                    hidden, blocks, first.op.block_write_idx
                )
                cache = fwd.kv_cache_data[f"layer_{layer.layer_idx}"]
                assert state_indices is not None
                hidden = self._run_kda(
                    layer,
                    geometry,
                    hidden,
                    blocks,
                    cache,
                    state_indices,
                    accepted,
                )
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
        self._chains.clear()
        self._refused.clear()
        self._announced.clear()
        self._prebuilt = False
