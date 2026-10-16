# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""S=1 Qwen decode runner backed by persistent GDN and QSA layers."""

import logging
from dataclasses import dataclass

import torch
from aiter.dist.communication_op import tensor_model_parallel_all_reduce
from aiter.dist.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_tp_group,
)

from atom.model_ops.qwen4_exp.qsa_attention import Qwen4ExpQSAPrepared
from atom.models.qwen4_exp_mono import contract
from atom.models.qwen4_exp_mono.check import (
    check_finish,
    check_gdn,
    check_gdn_reduce,
    check_qsa_post,
    check_qsa_pre,
)
from atom.models.qwen4_exp_mono.config import (
    ENABLED_WIDTHS,
    QSA_CONTEXT_BUCKETS,
    QSA_DIRECT_CONTEXT,
    QSA_MAX_MODEL_LEN,
    SUPPORTED_TP,
    SUPPORTED_WIDTHS,
)
from atom.models.qwen4_exp_mono.kernels import (
    collective,
    finish,
    gdn_layer,
    gdn_tp8,
    qsa_post,
    qsa_pre,
    qsa_tp8,
)
from atom.models.qwen4_exp_mono.plan import (
    ExpertLayout,
    LayerKind,
    MonoBuild,
    make_plan,
)
from atom.models.qwen4_exp_mono.state import MonoState, StateView
from atom.models.qwen4_exp_mono.step_meta import (
    GdnStepView,
    QsaStepView,
    gdn_step_view,
    qsa_step_view,
)
from atom.models.qwen4_exp_mono.weights import LayerWeights, bind_model
from atom.mono.plan.execution import BLOCKS
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.mono.runtime.widths import WidthBuilds
from atom.utils import envs
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")


@dataclass(frozen=True)
class WidthBuild:
    gdn_layer: dict[LayerKind, object]
    qsa_pre: object | None
    qsa_post: dict[int, object]
    tp8_finish: object | None = None
    tp8_pull: object | None = None
    tp8_gdn_k1: object | None = None
    tp8_gdn_k2: object | None = None
    tp8_qsa_k1: object | None = None
    tp8_qsa_k2: dict[int, object] | None = None


class QwenMonoRunner:
    """Run persistent GDN or two-launch QSA layers for an accepted width."""

    def __init__(self, model) -> None:
        self.model = model
        self.rank = get_tensor_model_parallel_rank()
        self.tp = get_tensor_model_parallel_world_size()
        self.group = get_tp_group().cpu_group
        self.device = torch.device("cuda", torch.cuda.current_device())
        bind_agreed(self._bind, self.group)
        self.state = MonoState(self, self.group, self.rank, self.tp, self.device)
        self.widths = WidthBuilds(
            self._build_width, self._kernels, self.group, "Qwen mono"
        )
        self._checked = set()
        self._closed = False
        self._running = False
        self.check = envs.ATOM_MONO_CHECK

    def _bind(self) -> None:
        if self.tp not in SUPPORTED_TP:
            raise MonoUnsupported(f"Qwen mono has no TP{self.tp} kernel ABI")
        props = torch.cuda.get_device_properties(self.device)
        arch = getattr(props, "gcnArchName", "").split(":")[0]
        if torch.version.hip is None or arch != "gfx950":
            raise MonoUnsupported(
                f"Qwen mono requires ROCm gfx950, got {arch or 'unknown'}"
            )
        if props.multi_processor_count != BLOCKS:
            raise MonoUnsupported(
                f"{props.multi_processor_count} CUs: Qwen mono needs all "
                f"{BLOCKS} CTAs co-resident"
            )
        if self.model.model.return_hc_state:
            raise MonoUnsupported("Qwen mono does not serve speculative target output")
        self.layers = self.model.model.layers
        if self.tp == 1:
            # S4 rows avoid the native GEMM seam and its strided inject logits.
            for layer in self.layers:
                layer.attn_hyper_connection.hc_rows_max = 4
                layer.mlp_hyper_connection.hc_rows_max = 4
            self.model.model.hyper_connection_mixer.hc_rows_max = 4
        self.plan = make_plan(self.layers)
        self.weights = bind_model(self.model, self.plan, self.tp)

    def _build_key(self, tokens: int, kind: LayerKind) -> MonoBuild:
        return MonoBuild(
            tokens,
            kind,
            tp=self.tp,
            expert_layout=ExpertLayout.for_tp(self.tp),
        )

    def _build_width(self, tokens: int) -> WidthBuild:
        if tokens not in SUPPORTED_WIDTHS:
            raise ValueError(f"Qwen MonoKernel has no {tokens}-token build")
        if self.tp == 8:
            if tokens != 1:
                raise ValueError("Qwen TP8 MonoKernel currently supports only S=1")
            key = self._build_key(tokens, LayerKind.GDN)
            return WidthBuild(
                gdn_layer={},
                qsa_pre=None,
                qsa_post={},
                tp8_finish=finish.build_finish(key),
                tp8_pull=collective.build_pull(tokens, self.tp),
                tp8_gdn_k1=gdn_tp8.build_gdn_k1(key),
                tp8_gdn_k2=gdn_tp8.build_gdn_k2(key),
                tp8_qsa_k1=qsa_tp8.build_qsa_k1(
                    self._build_key(tokens, LayerKind.QSA)
                ),
                tp8_qsa_k2={
                    QSA_DIRECT_CONTEXT: qsa_tp8.build_qsa_k2(
                        self._build_key(tokens, LayerKind.QSA),
                        direct=True,
                    ),
                },
            )
        return WidthBuild(
            {
                kind: gdn_layer.build_gdn_layer(self._build_key(tokens, kind))
                for kind in (LayerKind.GDN, LayerKind.PLE_GDN)
            },
            qsa_pre.build_qsa_pre(self._build_key(tokens, LayerKind.QSA)),
            {
                QSA_DIRECT_CONTEXT: qsa_post.build_qsa_post(
                    self._build_key(tokens, LayerKind.QSA),
                    direct=True,
                    direct_splits=qsa_post.DIRECT_ATTN_SPLITS,
                ),
                QSA_MAX_MODEL_LEN: qsa_post.build_qsa_post(
                    self._build_key(tokens, LayerKind.QSA)
                ),
            },
        )

    @staticmethod
    def _kernels(build: WidthBuild):
        if build.tp8_finish is not None:
            assert build.tp8_pull is not None
            assert build.tp8_gdn_k1 is not None
            assert build.tp8_gdn_k2 is not None
            assert build.tp8_qsa_k1 is not None
            assert build.tp8_qsa_k2 is not None
            return [
                (build.tp8_finish, finish.TP8_ABI),
                (build.tp8_pull, collective.PULL_ABI),
                (build.tp8_gdn_k1, gdn_tp8.K1_ABI),
                (build.tp8_gdn_k2, gdn_tp8.K2_ABI),
                (build.tp8_qsa_k1, qsa_tp8.ABI),
                *[
                    (launcher, qsa_tp8.K2_ABI)
                    for launcher in build.tp8_qsa_k2.values()
                ],
            ]
        assert build.qsa_pre is not None
        return [
            *[(launcher, gdn_layer.ABI) for launcher in build.gdn_layer.values()],
            (build.qsa_pre, qsa_pre.ABI),
            *[(launcher, qsa_post.ABI) for launcher in build.qsa_post.values()],
        ]

    def prepare(self, tokens: int) -> bool:
        if tokens not in ENABLED_WIDTHS:
            return False
        self.state.prepare(tokens, self.device)
        if not self.widths.prepare(tokens):
            return False
        if tokens not in self._checked:
            try:
                if self.tp == 8:
                    build = self.widths[tokens]
                    assert build.tp8_finish is not None
                    assert build.tp8_pull is not None
                    assert build.tp8_gdn_k1 is not None
                    assert build.tp8_gdn_k2 is not None
                    assert build.tp8_qsa_k1 is not None
                    assert build.tp8_qsa_k2 is not None
                    contract.check_launcher(
                        build.tp8_finish,
                        max_sgpr_spill=88,
                        max_lds=128 * 1024,
                    )
                    contract.check_launcher(
                        build.tp8_pull,
                        max_sgpr_spill=0,
                        max_lds=128 * 1024,
                    )
                    contract.check_launcher(
                        build.tp8_gdn_k1,
                        max_sgpr_spill=0,
                        max_lds=128 * 1024,
                    )
                    contract.check_launcher(
                        build.tp8_gdn_k2,
                        max_sgpr_spill=88,
                        max_lds=128 * 1024,
                    )
                    contract.check_launcher(
                        build.tp8_qsa_k1,
                        max_sgpr_spill=0,
                        max_lds=128 * 1024,
                    )
                    for launcher in build.tp8_qsa_k2.values():
                        contract.check_launcher(
                            launcher,
                            max_sgpr_spill=72,
                            max_lds=128 * 1024,
                        )
                    self._checked.add(tokens)
                    return True
                for launcher in self.widths[tokens].gdn_layer.values():
                    contract.check_launcher(
                        launcher,
                        max_sgpr_spill=contract.sgpr_spill_limit(tokens, "gdn"),
                    )
                contract.check_launcher(
                    self.widths[tokens].qsa_pre,
                    max_sgpr_spill=contract.sgpr_spill_limit(tokens, "qsa_pre"),
                )
                for context, launcher in self.widths[tokens].qsa_post.items():
                    stage = (
                        "qsa_direct" if context == QSA_DIRECT_CONTEXT else "qsa_full"
                    )
                    contract.check_launcher(
                        launcher,
                        max_sgpr_spill=contract.sgpr_spill_limit(tokens, stage),
                    )
            except Exception:
                logger.exception("Qwen mono S%d resource contract failed", tokens)
                return False
            self._checked.add(tokens)
        return True

    def _tp8_pull(self, layer_id: int, output: torch.Tensor, view: StateView) -> None:
        build = self.widths[view.tokens]
        if build.tp8_pull is None:
            raise RuntimeError("Qwen TP8 deferred pull is unavailable")
        build.tp8_pull(
            self.state.peers.addresses.data_ptr(),
            output.data_ptr(),
            layer_id,
            stream=torch.cuda.current_stream(),
        )

    def _tp8_layer(
        self,
        task,
        layer,
        weights,
        step,
        positions,
        hidden_states,
        input_ids,
        pending,
        view,
    ):
        if pending is not None:
            self._tp8_pull(task.layer_id - 1, pending[0], view)
        if layer.ple is not None:
            if pending is not None:
                hidden_states = layer.attn_hyper_connection.apply_pending(
                    hidden_states, pending
                )
                pending = None
            hidden_states = layer._apply_ple(hidden_states, input_ids)

        mixed, (streams, raw) = layer.attn_hyper_connection.mix_fused(
            hidden_states, pending
        )
        if task.is_gdn:
            if step is None:
                raise RuntimeError(f"Qwen TP8 GDN layer {task.layer_id} has no state")
            return self._tp8_gdn_layer(
                task,
                layer,
                weights,
                streams,
                mixed,
                raw,
                step,
                view,
            )

        if step is None:
            raise RuntimeError(f"Qwen TP8 QSA layer {task.layer_id} has no metadata")
        return self._tp8_qsa_layer(
            task,
            layer,
            weights,
            positions,
            streams,
            mixed,
            raw,
            step,
            view,
        )

    def _tp8_qsa_layer(
        self,
        task,
        layer,
        weights: LayerWeights,
        positions: torch.Tensor,
        streams: torch.Tensor,
        mixed: torch.Tensor,
        raw: torch.Tensor,
        step: QsaStepView,
        view: StateView,
    ) -> torch.Tensor:
        bound = weights.qsa
        if bound is None:
            raise RuntimeError(f"Qwen TP8 layer {task.layer_id} has no QSA weights")
        build = self.widths[view.tokens]
        if build.tp8_qsa_k1 is None:
            raise RuntimeError("Qwen TP8 QSA K1 is unavailable")

        def p(value) -> int:
            return 0 if value is None else value.data_ptr()

        args = qsa_tp8.ABI.pack(
            {
                "mixed": p(mixed),
                "qkv_weight": p(bound.qkv),
                "q_norm": p(bound.q_norm),
                "k_norm": p(bound.k_norm),
                "index_weight": p(bound.index_qk),
                "index_q_norm": p(bound.index_q_norm),
                "index_k_norm": p(bound.index_k_norm),
                "cos": p(bound.cos),
                "sin": p(bound.sin),
                "index_cos": p(bound.index_cos),
                "index_sin": p(bound.index_sin),
                "positions": p(positions),
                "position_rows": 1 if positions.dim() == 1 else positions.shape[0],
                "block_tables": p(step.block_tables),
                "table_stride": step.block_tables.stride(0),
                "slot_mapping": p(step.slot_mapping),
                "compressed_slot_mapping": p(step.compressed_slot_mapping),
                "token_to_req": p(step.token_to_req),
                "logical_positions": p(step.logical_positions),
                "page_size": step.k_cache.shape[1],
                "k_cache": p(step.k_cache),
                "v_cache": p(step.v_cache),
                "raw_key_cache": p(step.raw_key_cache),
                "compressed_key_cache": p(step.compressed_key_cache),
                "rope_position_cache": p(step.rope_position_cache),
                "query_out": p(view.qsa.query),
                "gate_out": p(view.qsa.gate),
                "index_query_out": p(view.qsa.index_query),
                "scratch": p(view.qsa_scratch),
                "layer": task.layer_id,
            }
        )
        build.tp8_qsa_k1(*args, stream=torch.cuda.current_stream())

        reference = None
        if self.check:
            prepared = Qwen4ExpQSAPrepared(
                view.qsa.gate,
                view.qsa.query,
                step.metadata,
                view.qsa.index_query,
            )
            slot = int(step.slot_mapping.item())
            compressed_slot = int(step.compressed_slot_mapping.item())
            mono_cache = {
                "k": step.k_cache.view(-1, *step.k_cache.shape[-2:])[slot].clone(),
                "v": step.v_cache.view(-1, *step.v_cache.shape[-2:])[slot].clone(),
                "raw": step.raw_key_cache.view(-1, *step.raw_key_cache.shape[-2:])[
                    slot
                ].clone(),
            }
            if compressed_slot >= 0:
                mono_cache["compressed"] = step.compressed_key_cache.view(
                    -1, *step.compressed_key_cache.shape[-2:]
                )[compressed_slot].clone()
            if step.rope_position_cache is not None:
                mono_cache["position"] = step.rope_position_cache.view(
                    -1, *step.rope_position_cache.shape[-2:]
                )[slot].clone()

            reference = layer.self_attn.prepare(positions, mixed)
            reference_cache = {
                "k": step.k_cache.view(-1, *step.k_cache.shape[-2:])[slot],
                "v": step.v_cache.view(-1, *step.v_cache.shape[-2:])[slot],
                "raw": step.raw_key_cache.view(-1, *step.raw_key_cache.shape[-2:])[
                    slot
                ],
            }
            if compressed_slot >= 0:
                reference_cache["compressed"] = step.compressed_key_cache.view(
                    -1, *step.compressed_key_cache.shape[-2:]
                )[compressed_slot]
            if step.rope_position_cache is not None:
                reference_cache["position"] = step.rope_position_cache.view(
                    -1, *step.rope_position_cache.shape[-2:]
                )[slot]
            check_qsa_pre(
                self.rank,
                task.layer_id,
                streams,
                streams,
                raw,
                raw,
                prepared,
                reference,
                mono_cache,
                reference_cache,
            )

        if build.tp8_qsa_k2 is None:
            raise RuntimeError("Qwen TP8 QSA K2 is unavailable")
        hc, moe = weights.mlp_hc, weights.moe
        if moe.expert_map is None:
            raise RuntimeError(f"Qwen TP8 layer {task.layer_id} has no expert map")
        streams_out = view.streams_out[task.layer_id & 1]
        if step.max_seq_len > QSA_DIRECT_CONTEXT:
            prepared = Qwen4ExpQSAPrepared(
                view.qsa.gate,
                view.qsa.query,
                step.metadata,
                view.qsa.index_query,
            )
            attn_out = tensor_model_parallel_all_reduce(
                layer.self_attn.finish(prepared)
            )
            finish_args = finish.TP8_ABI.pack(
                {
                    "streams": p(streams),
                    "attn_out": p(attn_out),
                    "attn_raw": p(raw),
                    "hc_norm": p(hc.norm),
                    "hc_down_inject": p(hc.down_inject),
                    "hc_up": p(hc.up),
                    "router": p(moe.router),
                    "shared_w13": p(moe.shared_w13),
                    "shared_w2": p(moe.shared_w2),
                    "routed_w13": p(moe.w13),
                    "routed_w13_scale": p(moe.w13_scale),
                    "routed_w2": p(moe.w2),
                    "routed_w2_scale": p(moe.w2_scale),
                    "expert_map": p(moe.expert_map),
                    "peer_sym": self.state.peers.local,
                    "streams_out": p(streams_out),
                    "pending_out": p(view.pending_out),
                    "pending_raw": p(view.pending_raw),
                    "moe_out": p(view.moe_out),
                    "scratch": p(view.qsa_post_scratch),
                    "layer": task.layer_id,
                }
            )
            assert build.tp8_finish is not None
            build.tp8_finish(*finish_args, stream=torch.cuda.current_stream())
            if self.check:
                assert reference is not None
                reference_attn = tensor_model_parallel_all_reduce(
                    layer.self_attn.finish(reference)
                )
                check_qsa_post(
                    self.rank,
                    task.layer_id,
                    attn_out,
                    reference_attn,
                )
                self._tp8_pull(task.layer_id, view.moe_out, view)
                check_finish(
                    self.rank,
                    task.layer_id,
                    layer,
                    streams,
                    attn_out,
                    raw,
                    streams_out,
                    view.pending_raw,
                    view.moe_out,
                )
            return streams_out, (view.moe_out, view.pending_raw)

        k2_args = qsa_tp8.K2_ABI.pack(
            {
                "peers": self.state.peers.addresses.data_ptr(),
                "streams": p(streams),
                "attn_raw": p(raw),
                "query": p(view.qsa.query),
                "gate": p(view.qsa.gate),
                "index_query": p(view.qsa.index_query),
                "block_tables": p(step.block_tables),
                "table_stride": step.block_tables.stride(0),
                "table_width": step.block_tables.shape[1],
                "seq_lens": p(step.seq_lens),
                "token_to_req": p(step.token_to_req),
                "logical_positions": p(step.logical_positions),
                "page_size": step.k_cache.shape[1],
                "cache_pages": step.k_cache.shape[0],
                "k_cache": p(step.k_cache),
                "v_cache": p(step.v_cache),
                "compressed_key_cache": p(step.compressed_key_cache),
                "o_proj": p(bound.o_proj),
                "hc_norm": p(hc.norm),
                "hc_down_inject": p(hc.down_inject),
                "hc_up": p(hc.up),
                "router": p(moe.router),
                "shared_w13": p(moe.shared_w13),
                "shared_w2": p(moe.shared_w2),
                "routed_w13": p(moe.w13),
                "routed_w13_scale": p(moe.w13_scale),
                "routed_w2": p(moe.w2),
                "routed_w2_scale": p(moe.w2_scale),
                "expert_map": p(moe.expert_map),
                "peer_sym": self.state.peers.local,
                "attn_out": p(view.qsa_post_attn_out),
                "streams_out": p(streams_out),
                "pending_out": p(view.pending_out),
                "pending_raw": p(view.pending_raw),
                "moe_out": p(view.moe_out),
                "scratch": p(view.qsa_post_scratch),
                "layer": task.layer_id,
            }
        )
        build.tp8_qsa_k2[QSA_DIRECT_CONTEXT](
            *k2_args, stream=torch.cuda.current_stream()
        )
        if self.check:
            assert reference is not None
            reference_attn = tensor_model_parallel_all_reduce(
                layer.self_attn.finish(reference)
            )
            check_qsa_post(
                self.rank,
                task.layer_id,
                view.qsa_post_attn_out,
                reference_attn,
            )
            self._tp8_pull(task.layer_id, view.moe_out, view)
            check_finish(
                self.rank,
                task.layer_id,
                layer,
                streams,
                view.qsa_post_attn_out,
                raw,
                streams_out,
                view.pending_raw,
                view.moe_out,
            )
        return streams_out, (view.moe_out, view.pending_raw)


    def _tp8_gdn_layer(
        self,
        task,
        layer,
        weights: LayerWeights,
        streams: torch.Tensor,
        mixed: torch.Tensor,
        raw: torch.Tensor,
        step: GdnStepView,
        view: StateView,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        bound = weights.gdn
        if bound is None:
            raise RuntimeError(f"Qwen TP8 layer {task.layer_id} has no GDN weights")
        build = self.widths[view.tokens]
        if build.tp8_gdn_k1 is None or build.tp8_gdn_k2 is None:
            raise RuntimeError("Qwen TP8 GDN K1/K2 is unavailable")

        def p(value) -> int:
            return value.data_ptr()

        write_slot = None
        original = None
        if self.check:
            write_slot = int(step.state_write.item())
            original = (
                step.conv_state[write_slot].clone(),
                step.recurrent_state[write_slot].clone(),
            )

        k1_args = gdn_tp8.K1_ABI.pack(
            {
                "x": p(mixed),
                "projection_weight": p(bound.qkvzba),
                "conv_weight": p(bound.conv),
                "conv_state": p(step.conv_state),
                "state_read": p(step.state_read),
                "state_write": p(step.state_write),
                "a_log": p(bound.a_log),
                "dt_bias": p(bound.dt_bias),
                "state": p(step.recurrent_state),
                "norm_weight": p(bound.norm),
                "output_weight": p(bound.out),
                "partial_out": p(view.gdn.out),
                "peer_sym": self.state.peers.local,
                "scratch": p(view.scratch),
                "layer": task.layer_id,
            }
        )
        build.tp8_gdn_k1(*k1_args, stream=torch.cuda.current_stream())

        reference_attn = None
        if self.check:
            assert write_slot is not None and original is not None
            mono_partial = view.gdn.out.clone()
            mono_state = (
                step.conv_state[write_slot].clone(),
                step.recurrent_state[write_slot].clone(),
            )
            step.conv_state[write_slot].copy_(original[0])
            step.recurrent_state[write_slot].copy_(original[1])
            reference_partial = layer.linear_attn(mixed)
            check_gdn(
                self.rank,
                task.layer_id,
                mono_partial,
                reference_partial,
                mono_state[0],
                step.conv_state[write_slot],
                mono_state[1],
                step.recurrent_state[write_slot],
            )
            step.conv_state[write_slot].copy_(mono_state[0])
            step.recurrent_state[write_slot].copy_(mono_state[1])
            reference_attn = tensor_model_parallel_all_reduce(reference_partial)

        return self._tp8_gdn_finish(
            task,
            layer,
            weights,
            streams,
            raw,
            view,
            reference_attn,
        )

    def _tp8_gdn_finish(
        self,
        task,
        layer,
        weights: LayerWeights,
        streams: torch.Tensor,
        raw: torch.Tensor,
        view: StateView,
        reference_attn: torch.Tensor | None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Run the shared GDN K2 after the rank-local GDN K1."""

        hc, moe = weights.mlp_hc, weights.moe
        if moe.expert_map is None:
            raise RuntimeError(f"Qwen TP8 layer {task.layer_id} has no expert map")
        build = self.widths[view.tokens]
        if build.tp8_gdn_k2 is None:
            raise RuntimeError("Qwen TP8 GDN K2 is unavailable")

        def p(value) -> int:
            return value.data_ptr()

        streams_out = view.streams_out[task.layer_id & 1]
        k2_args = gdn_tp8.K2_ABI.pack(
            {
                "peers": self.state.peers.addresses.data_ptr(),
                "streams": p(streams),
                "attn_out": p(view.gdn.out),
                "attn_raw": p(raw),
                "hc_norm": p(hc.norm),
                "hc_down_inject": p(hc.down_inject),
                "hc_up": p(hc.up),
                "router": p(moe.router),
                "shared_w13": p(moe.shared_w13),
                "shared_w2": p(moe.shared_w2),
                "routed_w13": p(moe.w13),
                "routed_w13_scale": p(moe.w13_scale),
                "routed_w2": p(moe.w2),
                "routed_w2_scale": p(moe.w2_scale),
                "expert_map": p(moe.expert_map),
                "peer_sym": self.state.peers.local,
                "streams_out": p(streams_out),
                "pending_out": p(view.pending_out),
                "pending_raw": p(view.pending_raw),
                "moe_out": p(view.moe_out),
                "scratch": p(view.scratch),
                "layer": task.layer_id,
            }
        )
        build.tp8_gdn_k2(*k2_args, stream=torch.cuda.current_stream())

        if self.check:
            assert reference_attn is not None
            check_gdn_reduce(
                self.rank,
                task.layer_id,
                view.gdn.out,
                reference_attn,
            )
            self._tp8_pull(task.layer_id, view.moe_out, view)
            check_finish(
                self.rank,
                task.layer_id,
                layer,
                streams,
                view.gdn.out,
                raw,
                streams_out,
                view.pending_raw,
                view.moe_out,
            )
        return streams_out, (view.moe_out, view.pending_raw)

    def close(self) -> None:
        if self._closed:
            return
        if self._running:
            raise RuntimeError("cannot close an active Qwen mono runner")
        self.state.close()
        self._closed = True

    def _gdn_layer(
        self,
        layer_id: int,
        kind: LayerKind,
        streams: torch.Tensor,
        mixed: torch.Tensor,
        raw: torch.Tensor,
        weights: LayerWeights,
        step: GdnStepView,
        view: StateView,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        bound = weights.gdn
        if bound is None:
            raise RuntimeError(f"Qwen layer {layer_id} has no bound GDN weights")
        # Native HC uses a slice of its rank+inject GEMM output at S > 2.
        # The Mono ABI requires packed per-token inject logits.
        raw = raw.contiguous()
        hc, moe = weights.mlp_hc, weights.moe
        streams_out = view.streams_out[layer_id & 1]
        args = gdn_layer.ABI.pack(
            {
                "streams": streams.data_ptr(),
                "gdn_input": mixed.data_ptr(),
                "attn_raw": raw.data_ptr(),
                "gdn_qkvzba": bound.qkvzba.data_ptr(),
                "gdn_conv": bound.conv.data_ptr(),
                "gdn_a_log": bound.a_log.data_ptr(),
                "gdn_dt_bias": bound.dt_bias.data_ptr(),
                "gdn_norm": bound.norm.data_ptr(),
                "gdn_out": bound.out.data_ptr(),
                "gdn_conv_state": step.conv_state.data_ptr(),
                "gdn_recurrent_state": step.recurrent_state.data_ptr(),
                "gdn_state_read": step.state_read.data_ptr(),
                "gdn_state_write": step.state_write.data_ptr(),
                "hc_norm": hc.norm.data_ptr(),
                "hc_down_inject": hc.down_inject.data_ptr(),
                "hc_up": hc.up.data_ptr(),
                "router": moe.router.data_ptr(),
                "shared_w13": moe.shared_w13.data_ptr(),
                "shared_w2": moe.shared_w2.data_ptr(),
                "routed_w13": moe.w13.data_ptr(),
                "routed_w13_scale": moe.w13_scale.data_ptr(),
                "routed_w2": moe.w2.data_ptr(),
                "routed_w2_scale": moe.w2_scale.data_ptr(),
                "streams_out": streams_out.data_ptr(),
                "pending_out": view.pending_out.data_ptr(),
                "pending_raw": view.pending_raw.data_ptr(),
                "moe_out": view.moe_out.data_ptr(),
                "scratch": view.scratch.data_ptr(),
                "layer": layer_id,
            }
        )
        self.widths[view.tokens].gdn_layer[kind](
            *args, stream=torch.cuda.current_stream()
        )
        return streams_out, (view.moe_out, view.pending_raw)

    def _qsa_layer(
        self,
        layer_id: int,
        positions: torch.Tensor,
        streams: torch.Tensor,
        pending: tuple[torch.Tensor, torch.Tensor] | None,
        layer,
        weights: LayerWeights,
        step: QsaStepView,
        view: StateView,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        if pending is None:
            raise RuntimeError(f"Qwen QSA layer {layer_id} has no deferred input")
        qsa = weights.qsa
        if qsa is None:
            raise RuntimeError(f"Qwen layer {layer_id} has no bound QSA weights")
        block_out, raw = pending

        def p(value) -> int:
            return 0 if value is None else value.data_ptr()

        args = qsa_pre.ABI.pack(
            {
                "streams": p(streams),
                "pending_out": p(block_out),
                "pending_raw": p(raw),
                "hc_norm": p(weights.attn_hc.norm),
                "hc_down_inject": p(weights.attn_hc.down_inject),
                "hc_up": p(weights.attn_hc.up),
                "qkv_weight": p(qsa.qkv),
                "q_norm": p(qsa.q_norm),
                "k_norm": p(qsa.k_norm),
                "index_weight": p(qsa.index_qk),
                "index_q_norm": p(qsa.index_q_norm),
                "index_k_norm": p(qsa.index_k_norm),
                "cos": p(qsa.cos),
                "sin": p(qsa.sin),
                "index_cos": p(qsa.index_cos),
                "index_sin": p(qsa.index_sin),
                "positions": p(positions),
                "position_rows": 1 if positions.dim() == 1 else positions.shape[0],
                "block_tables": p(step.block_tables),
                "table_stride": step.block_tables.stride(0),
                "slot_mapping": p(step.slot_mapping),
                "compressed_slot_mapping": p(step.compressed_slot_mapping),
                "token_to_req": p(step.token_to_req),
                "logical_positions": p(step.logical_positions),
                "page_size": step.k_cache.shape[1],
                "k_cache": p(step.k_cache),
                "v_cache": p(step.v_cache),
                "raw_key_cache": p(step.raw_key_cache),
                "compressed_key_cache": p(step.compressed_key_cache),
                "rope_position_cache": p(step.rope_position_cache),
                "streams_out": p(view.qsa.streams),
                "attn_raw_out": p(view.qsa.attn_raw),
                "query_out": p(view.qsa.query),
                "gate_out": p(view.qsa.gate),
                "index_query_out": p(view.qsa.index_query),
                "scratch": p(view.qsa_scratch),
                "layer": layer_id,
            }
        )
        self.widths[view.tokens].qsa_pre(*args, stream=torch.cuda.current_stream())
        reference_attn = None
        if self.check:
            prepared = Qwen4ExpQSAPrepared(
                view.qsa.gate,
                view.qsa.query,
                step.metadata,
                view.qsa.index_query,
            )
            slot = int(step.slot_mapping.item())
            compressed_slot = int(step.compressed_slot_mapping.item())
            mono_cache = {
                "k": step.k_cache.view(-1, *step.k_cache.shape[-2:])[slot].clone(),
                "v": step.v_cache.view(-1, *step.v_cache.shape[-2:])[slot].clone(),
                "raw": step.raw_key_cache.view(-1, *step.raw_key_cache.shape[-2:])[
                    slot
                ].clone(),
            }
            if compressed_slot >= 0:
                mono_cache["compressed"] = step.compressed_key_cache.view(
                    -1, *step.compressed_key_cache.shape[-2:]
                )[compressed_slot].clone()
            if step.rope_position_cache is not None:
                mono_cache["position"] = step.rope_position_cache.view(
                    -1, *step.rope_position_cache.shape[-2:]
                )[slot].clone()
            ref_mixed, (ref_streams, ref_raw) = layer.attn_hyper_connection.mix_fused(
                streams, pending
            )
            reference = layer.self_attn.prepare(positions, ref_mixed)
            reference_cache = {
                "k": step.k_cache.view(-1, *step.k_cache.shape[-2:])[slot],
                "v": step.v_cache.view(-1, *step.v_cache.shape[-2:])[slot],
                "raw": step.raw_key_cache.view(-1, *step.raw_key_cache.shape[-2:])[
                    slot
                ],
            }
            if compressed_slot >= 0:
                reference_cache["compressed"] = step.compressed_key_cache.view(
                    -1, *step.compressed_key_cache.shape[-2:]
                )[compressed_slot]
            if step.rope_position_cache is not None:
                reference_cache["position"] = step.rope_position_cache.view(
                    -1, *step.rope_position_cache.shape[-2:]
                )[slot]
            check_qsa_pre(
                self.rank,
                layer_id,
                view.qsa.streams,
                ref_streams,
                view.qsa.attn_raw,
                ref_raw,
                prepared,
                reference,
                mono_cache,
                reference_cache,
            )
            reference_attn = layer.self_attn.finish(reference)

        streams_out = view.streams_out[layer_id & 1]
        args = qsa_post.ABI.pack(
            {
                "streams": p(view.qsa.streams),
                "attn_raw": p(view.qsa.attn_raw),
                "query": p(view.qsa.query),
                "gate": p(view.qsa.gate),
                "index_query": p(view.qsa.index_query),
                "block_tables": p(step.block_tables),
                "table_stride": step.block_tables.stride(0),
                "table_width": step.block_tables.shape[1],
                "seq_lens": p(step.seq_lens),
                "token_to_req": p(step.token_to_req),
                "logical_positions": p(step.logical_positions),
                "page_size": step.k_cache.shape[1],
                "cache_pages": step.k_cache.shape[0],
                "k_cache": p(step.k_cache),
                "v_cache": p(step.v_cache),
                "compressed_key_cache": p(step.compressed_key_cache),
                "o_proj": p(qsa.o_proj),
                "hc_norm": p(weights.mlp_hc.norm),
                "hc_down_inject": p(weights.mlp_hc.down_inject),
                "hc_up": p(weights.mlp_hc.up),
                "router": p(weights.moe.router),
                "shared_w13": p(weights.moe.shared_w13),
                "shared_w2": p(weights.moe.shared_w2),
                "routed_w13": p(weights.moe.w13),
                "routed_w13_scale": p(weights.moe.w13_scale),
                "routed_w2": p(weights.moe.w2),
                "routed_w2_scale": p(weights.moe.w2_scale),
                "streams_out": p(streams_out),
                "pending_out": p(view.pending_out),
                "pending_raw": p(view.pending_raw),
                "moe_out": p(view.moe_out),
                "scratch": p(view.qsa_post_scratch),
                "layer": layer_id,
            }
        )
        qsa_metadata = get_forward_context().attn_metadata.qsa_metadata
        max_seq_len = int(qsa_metadata.max_seq_len)
        context = next(
            bucket for bucket in QSA_CONTEXT_BUCKETS if max_seq_len <= bucket
        )
        self.widths[view.tokens].qsa_post[context](
            *args, stream=torch.cuda.current_stream()
        )
        result = streams_out, (view.moe_out, view.pending_raw)
        if self.check:
            assert reference_attn is not None
            streams_out, (moe_out, pending_raw) = result
            check_qsa_post(
                self.rank,
                layer_id,
                view.qsa_post_attn_out,
                reference_attn,
            )
            check_finish(
                self.rank,
                layer_id,
                layer,
                view.qsa.streams,
                reference_attn,
                view.qsa.attn_raw,
                streams_out,
                pending_raw,
                moe_out,
            )
        return result

    def _layer(
        self,
        task,
        layer,
        weights,
        step,
        positions,
        hidden_states,
        input_ids,
        pending,
        view,
    ):
        if self.tp == 8:
            return self._tp8_layer(
                task,
                layer,
                weights,
                step,
                positions,
                hidden_states,
                input_ids,
                pending,
                view,
            )
        if layer.ple is not None:
            if pending is not None:
                hidden_states = layer.attn_hyper_connection.apply_pending(
                    hidden_states, pending
                )
                pending = None
            hidden_states = layer._apply_ple(hidden_states, input_ids)

        if not task.is_gdn:
            return self._qsa_layer(
                task.layer_id,
                positions,
                hidden_states,
                pending,
                layer,
                weights,
                step,
                view,
            )

        mixed, (hidden_states, raw) = layer.attn_hyper_connection.mix_fused(
            hidden_states, pending
        )
        if task.is_gdn:
            gdn_step = step
            original = None
            write_slot = None
            if self.check:
                write_slot = int(gdn_step.state_write.item())
                original = (
                    gdn_step.conv_state[write_slot].clone(),
                    gdn_step.recurrent_state[write_slot].clone(),
                )
            result = self._gdn_layer(
                task.layer_id,
                task.kind,
                hidden_states,
                mixed,
                raw,
                weights,
                gdn_step,
                view,
            )
            if self.check:
                mono_out = view.gdn.out.clone()
                mono_state = (
                    gdn_step.conv_state[write_slot].clone(),
                    gdn_step.recurrent_state[write_slot].clone(),
                )
                gdn_step.conv_state[write_slot].copy_(original[0])
                gdn_step.recurrent_state[write_slot].copy_(original[1])
                ref_out = layer.linear_attn(mixed)
                check_gdn(
                    self.rank,
                    task.layer_id,
                    mono_out,
                    ref_out,
                    mono_state[0],
                    gdn_step.conv_state[write_slot],
                    mono_state[1],
                    gdn_step.recurrent_state[write_slot],
                )
                gdn_step.conv_state[write_slot].copy_(mono_state[0])
                gdn_step.recurrent_state[write_slot].copy_(mono_state[1])
                streams_out, (moe_out, pending_raw) = result
                check_finish(
                    self.rank,
                    task.layer_id,
                    layer,
                    hidden_states,
                    mono_out,
                    raw,
                    streams_out,
                    pending_raw,
                    moe_out,
                )
            return result
        raise AssertionError(f"unexpected non-GDN task {task}")

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor):
        if self._closed:
            raise RuntimeError("closed Qwen mono runner")
        if self._running:
            raise RuntimeError("nested Qwen mono forward")
        self._running = True
        try:
            model = self.model.model
            tokens = input_ids.numel()
            view = self.state.view(tokens)
            self.state.begin_step(tokens)
            context = get_forward_context()
            if self.tp == 8:
                steps = {
                    task.layer_id: (
                        gdn_step_view(
                            context, task.layer_id, tokens, tp=self.tp
                        )
                        if task.is_gdn
                        else qsa_step_view(
                            context,
                            self.layers[task.layer_id].self_attn,
                            tokens,
                            tp=self.tp,
                        )
                    )
                    for task in self.plan.layers
                }
            else:
                steps = {
                    task.layer_id: (
                        gdn_step_view(context, task.layer_id, tokens)
                        if task.is_gdn
                        else qsa_step_view(
                            context, self.layers[task.layer_id].self_attn, tokens
                        )
                    )
                    for task in self.plan.layers
                }
            hidden_states = model.get_input_embeddings(input_ids).repeat(
                1, model.hc_count
            )
            pending = None
            for task, layer, weights in zip(
                self.plan.layers, self.layers, self.weights, strict=True
            ):
                hidden_states, pending = self._layer(
                    task,
                    layer,
                    weights,
                    steps[task.layer_id],
                    positions,
                    hidden_states,
                    input_ids,
                    pending,
                    view,
                )
            if self.tp == 8 and pending is not None:
                self._tp8_pull(self.plan.layers[-1].layer_id, pending[0], view)
            if model.return_hc_state:
                hidden_states = model.hyper_connection_mixer.apply_pending(
                    hidden_states, pending
                )
                return hidden_states.view(-1, model.hc_count, model.config.hidden_size)
            mixed, _ = model.hyper_connection_mixer.mix_fused(hidden_states, pending)
            return mixed
        finally:
            self._running = False
