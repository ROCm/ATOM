# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""One Qwen layer lifecycle with TP-specific kernel and attention seams."""

import torch
from aiter.dist.communication_op import tensor_model_parallel_all_reduce

from atom.models.qwen4_exp_mono import batch
from atom.models.qwen4_exp_mono.check import (
    check_finish,
    check_gdn_reduce,
    check_qsa_post,
    gdn_reference,
    gdn_snapshot,
    qsa_reference,
)
from atom.models.qwen4_exp_mono.kernels import (
    finish,
    gdn_layer,
    gdn_tp8,
    qsa_post,
    qsa_pre,
    qsa_tp8,
)
from atom.models.qwen4_exp_mono.launch_args import (
    apply_ple,
    bind_static_weights,
    gdn_state_args,
    output_args,
    qsa_cache_args,
    qsa_decode_args,
)
from atom.models.qwen4_exp_mono.launch_args import (
    pointer as p,
)
from atom.models.qwen4_exp_mono.native_qsa import PreparedQsa
from atom.models.qwen4_exp_mono.shapes import QSA_DIRECT_CONTEXT


class LayerExecution:
    def __init__(self, runner):
        self.runner = runner
        self.distributed = runner.tp == 8
        self.static = tuple(bind_static_weights(w, runner.tp) for w in runner.weights)

    def finish_step(self, pending, view):
        if view.batched:
            return
        if self.distributed and pending is not None:
            self._tp8_pull(self.runner.plan.layers[-1].layer_id, pending[0], view)

    def layer(
        self, task, layer, weights, step, positions, streams, input_ids, pending, view
    ):
        batched = view.batched
        if self.distributed and not batched and pending is not None:
            self._tp8_pull(task.layer_id - 1, pending[0], view)
        streams, pending = apply_ple(layer, streams, pending, input_ids)
        if batched:
            mixed, (streams, raw) = batch.mix(
                layer.attn_hyper_connection, streams, pending, view
            )
            sub = (
                batch.attention(
                    view, self.static[task.layer_id]["batch_gdn"], step, mixed
                )
                if task.is_gdn
                else batch.qsa_attention(
                    layer.self_attn,
                    positions,
                    mixed,
                    view,
                    self.static[task.layer_id]["batch_qsa"],
                )
            )
            sub = tensor_model_parallel_all_reduce(sub)
            if view.hc_front is not None:
                return batch.mlp_front(
                    layer.mlp_hyper_connection,
                    streams,
                    (sub, raw),
                    view,
                    self.static[task.layer_id]["batch_moe"],
                )
            mixed, (streams, raw) = batch.mix(
                layer.mlp_hyper_connection, streams, (sub, raw), view
            )
            return streams, (
                batch.moe(view, self.static[task.layer_id]["batch_moe"], mixed),
                raw,
            )
        if not task.is_gdn and not self.distributed:
            if pending is None:
                raise RuntimeError(
                    f"Qwen QSA layer {task.layer_id} has no deferred input"
                )
            return self.qsa(
                task.layer_id,
                layer,
                weights,
                positions,
                streams,
                None,
                pending[1],
                pending,
                step,
                view,
            )
        mixed, (streams, raw) = layer.attn_hyper_connection.mix_fused(streams, pending)
        if task.is_gdn:
            return self.gdn(
                task.layer_id, layer, weights, streams, mixed, raw, step, view
            )
        return self.qsa(
            task.layer_id,
            layer,
            weights,
            positions,
            streams,
            mixed,
            raw,
            None,
            step,
            view,
        )

    @staticmethod
    def mix(module, streams, pending, view):
        return (
            batch.mix(module, streams, pending, view)
            if view.batched
            else module.mix_fused(streams, pending)
        )

    def finish_values(self, layer_id, weights, streams, raw, view):
        values = dict(
            streams=p(streams),
            attn_raw=p(raw),
            **self.static[layer_id]["finish"],
            **output_args(view.streams_out[layer_id & 1], view),
            layer=layer_id,
        )
        if self.distributed:
            if weights.moe.expert_map is None:
                raise RuntimeError(f"Qwen TP8 layer {layer_id} has no expert map")
            values.update(
                peer_sym=self.runner.state.peers.local,
            )
        return values

    @staticmethod
    def launch(launcher, abi, values):
        launcher(*abi.pack(values), stream=torch.cuda.current_stream())

    def gdn(self, layer_id, layer, weights, streams, mixed, raw, step, view):
        if weights.gdn is None:
            raise RuntimeError(f"Qwen layer {layer_id} has no bound GDN weights")
        build = self.runner.widths[view.tokens]
        snapshot = gdn_snapshot(step) if self.runner.check else None
        if self.distributed:
            values = dict(
                x=p(mixed),
                **self.static[layer_id]["gdn"],
                **gdn_state_args(step, 8),
                partial_out=p(view.gdn_out),
                peer_sym=self.runner.state.peers.local,
                scratch=p(view.scratch),
                layer=layer_id,
            )
            self.launch(build.tp8_gdn_k1, gdn_tp8.K1_ABI, values)
        else:
            raw = raw.contiguous()
            values = self.finish_values(layer_id, weights, streams, raw, view)
            values.update(
                gdn_input=p(mixed),
                **self.static[layer_id]["gdn"],
                **gdn_state_args(step, 1),
                scratch=p(view.scratch),
            )
            self.launch(build.gdn_layer, gdn_layer.ABI, values)
        reference = None
        if self.runner.check:
            reference = gdn_reference(
                self.runner.rank,
                layer_id,
                layer,
                mixed,
                step,
                view.gdn_out.clone(),
                snapshot,
            )
            if self.distributed:
                reference = tensor_model_parallel_all_reduce(reference)
        if self.distributed:
            values = self.finish_values(layer_id, weights, streams, raw, view)
            values.update(
                peers=p(self.runner.state.peers.addresses),
                attn_out=p(view.gdn_out),
                scratch=p(view.scratch),
            )
            self.launch(build.tp8_gdn_k2, gdn_tp8.K2_ABI, values)
        if self.runner.check:
            if self.distributed:
                check_gdn_reduce(self.runner.rank, layer_id, view.gdn_out, reference)
            self.check_finish(layer_id, layer, streams, view.gdn_out, raw, view)
        return view.streams_out[layer_id & 1], (view.moe_out, view.pending_raw)

    def qsa(
        self,
        layer_id,
        layer,
        weights,
        positions,
        streams,
        mixed,
        raw,
        pending,
        step,
        view,
    ):
        bound = weights.qsa
        if bound is None:
            raise RuntimeError(f"Qwen layer {layer_id} has no bound QSA weights")
        build = self.runner.widths[view.tokens]
        values = dict(
            **self.static[layer_id]["qsa"],
            positions=p(positions),
            position_rows=1 if positions.dim() == 1 else positions.shape[0],
            **qsa_cache_args(step),
            query_out=p(view.qsa.query),
            gate_out=p(view.qsa.gate),
            index_query_out=p(view.qsa.index_query),
            scratch=p(view.qsa_scratch),
            layer=layer_id,
        )
        if self.distributed:
            values["mixed"] = p(mixed)
            self.launch(build.tp8_qsa_k1, qsa_tp8.ABI, values)
        else:
            values.update(
                streams=p(streams),
                pending_out=p(pending[0]),
                pending_raw=p(raw),
                **self.static[layer_id]["attn_hc"],
                streams_out=p(view.qsa.streams),
                attn_raw_out=p(view.qsa.attn_raw),
            )
            self.launch(build.qsa_pre, qsa_pre.ABI, values)
        reference = None
        if self.runner.check:
            ref_streams, ref_raw = streams, raw
            if not self.distributed:
                mixed, (ref_streams, ref_raw) = layer.attn_hyper_connection.mix_fused(
                    streams, pending
                )
            reference = qsa_reference(
                self.runner.rank,
                layer_id,
                layer,
                positions,
                mixed,
                step,
                view.qsa,
                streams if self.distributed else view.qsa.streams,
                ref_streams,
                raw if self.distributed else view.qsa.attn_raw,
                ref_raw,
            )
        if not self.distributed:
            streams, raw = view.qsa.streams, view.qsa.attn_raw
        values = self.finish_values(layer_id, weights, streams, raw, view)
        values["scratch"] = p(view.qsa_post_scratch)
        if self.distributed and step.max_seq_len > QSA_DIRECT_CONTEXT:
            prepared = PreparedQsa(
                view.qsa.gate, view.qsa.query, step.metadata, view.qsa.index_query
            )
            attn_out = tensor_model_parallel_all_reduce(
                batch.qsa_finish(layer.self_attn, prepared)
            )
            values["attn_out"] = p(attn_out)
            self.launch(build.tp8_finish, finish.TP8_ABI, values)
        else:
            values.update(
                query=p(view.qsa.query),
                gate=p(view.qsa.gate),
                index_query=p(view.qsa.index_query),
                **qsa_decode_args(step),
                o_proj=p(bound.o_proj),
            )
            attn_out = view.qsa_post_attn_out
            if self.distributed:
                values.update(
                    peers=p(self.runner.state.peers.addresses), attn_out=p(attn_out)
                )
                self.launch(build.tp8_qsa_k2, qsa_tp8.K2_ABI, values)
            else:
                self.launch(build.qsa_post, qsa_post.ABI, values)
        if self.runner.check:
            reference_attn = reference
            if self.distributed:
                reference_attn = tensor_model_parallel_all_reduce(reference_attn)
            check_qsa_post(self.runner.rank, layer_id, attn_out, reference_attn)
            self.check_finish(layer_id, layer, streams, attn_out, raw, view)
        return view.streams_out[layer_id & 1], (view.moe_out, view.pending_raw)

    def check_finish(self, layer_id, layer, streams, attn_out, raw, view):
        if self.distributed:
            self._tp8_pull(layer_id, view.moe_out, view)
        check_finish(
            self.runner.rank,
            layer_id,
            layer,
            streams,
            attn_out,
            raw,
            view.streams_out[layer_id & 1],
            view.pending_raw,
            view.moe_out,
        )
