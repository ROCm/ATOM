# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Frozen gfx950 layer loop with the gfx1250 TP2/TP4 c1 kernels.

Weights are installed before KV sizing. Only pointer ABI and cooperative
launch binding differ from V41MonoDecodeRunner; its forward is inherited.
Fixed ingress and metadata storage permit preparing every eager graph before
any peer-dependent launch. Candidate/unit tables retain their original
materialization kernels and position in the layer loop.
"""

import logging
import os
from dataclasses import dataclass

import torch
from aiter.dist.parallel_state import get_tp_group
from torch import nn

from atom.model_ops.deepseek_v41.candidate_table import candidate_block_table
from atom.model_ops.deepseek_v41.unit_table import unit_table
from atom.models.deepseek_v41.mono import index_plan as ip
from atom.models.deepseek_v41.mono.config import config_refusal
from atom.models.deepseek_v41.mono.dispatch import step_supported
from atom.models.deepseek_v41.mono.gfx1250.gfx1250_diagnostics import _diagnostics
from atom.models.deepseek_v41.mono.gfx1250.gfx1250_reference_weights import bind_reference_model
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_k1 import _CooperativeKernel
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_index_query import (
    build_reference_k1_c1, reference_k1_symbol,
)
from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_wo import (
    FFN_DOWN_ARGS, INDEX_ARGS, INDEX_ATTENTION_ARGS,
    attention_wob_symbol_c1, build_reference_attention_wob_c1,
)
from atom.models.deepseek_v41.mono.gfx950.runner import V41MonoDecodeRunner, f32_bits
from atom.model_ops.v4_kernels.paged_decode import LOG2E
from atom.mono.runtime.aux_taps import AuxTaps
from atom.mono.runtime.consensus import MonoUnsupported, bind_agreed
from atom.mono.runtime.lifecycle import owned_peer_buffer
from atom.mono.runtime.mailboxes import StepMailboxes
from atom.models.deepseek_v41.mono.kernels.peer_layout import peer_bytes
from atom.utils.forward_context import get_forward_context

logger = logging.getLogger("atom")
K1_ARGS = (
    "res_in", "pend", "post_in", "comb_in", "pre_in", "hc_fn", "res_out", "scratch",
    "diag", "layer", "attn_w", "normed", "weight", "scale", "hc_scale", "hc_base",
    "pre_out", "post_out", "comb_out", "qn_w", "kvn_w", "cos", "sin", "pos",
    "ring_rows", "ring", "ring_off", "qr_out", "qrs_out", "wqb", "wqb_s", "q_out",
    "iq_w", "iq_ws", "projection_values", "iw_w", "iq_out", "iqs_out", "iw_out",
)
K2_ARGS = (
    "q", "pool", "sel", "sbound", "kmeta", "sink", "pos", "cos", "sin", "scratch",
    "diag", "layer", "config", "out", "woa", "woa_s", "woa_out", "wob", "wob_s",
    "sym", "peers", "rank", "reduce_out",
)


@dataclass(frozen=True)
class ReferenceK1Key:
    fold: bool
    feeds_index: bool


class Gfx1250ReferenceRunner(V41MonoDecodeRunner):
    max_rows = 1
    tok_par_from = 2

    def __init__(self, model, layers, max_model_len):
        self._installed_layers = layers
        self._cache = None
        self._ready = False
        super().__init__(model, None, max_model_len, False)

    def _bind(self, model, drafter):
        self.model = model
        self._bind_rank()
        # The gfx950 runner launches on the current stream. Keep that host
        # schedule: real cooperative nodes add device-queue fences in HIP.
        # Both modes retain the checked capacity for all persistent CTAs.
        launch_mode = os.environ.get(
            "ATOM_DSV41_GFX1250_REFERENCE_LAUNCH_MODE", "ordinary"
        )
        if launch_mode not in ("cooperative", "ordinary"):
            raise MonoUnsupported("reference launch mode must be cooperative or ordinary")
        self.cooperative_dispatch = launch_mode == "cooperative"
        if self.tp not in (2, 4) or len(model.layers) != 40 or len(self._installed_layers) != 40:
            raise MonoUnsupported("reference runner requires all 40 TP2/TP4 layers")
        if not 0 < self.index_bound_max <= 16384:
            raise MonoUnsupported("reference FP8 index bound must be <=16384")
        self.layers = self._installed_layers
        self.aux = AuxTaps((), ())
        self.check = self.split_k2 = self.debug = self.timeline = False
        self.timelines = {}
        self.selecting = [s.layer_id for s, b in zip(model.topology, model.layers)
                          if b.attn.indexer is not None]
        for rope in (model.window_rope, model.global_rope):
            for table in (rope.cos_cache, rope.sin_cache):
                if table.dtype != torch.bfloat16 or table.shape[-1] != 32:
                    raise MonoUnsupported("reference RoPE requires original BF16 tables")
        self._allocate(model.config)
        from atom.models.deepseek_v41.mono.gfx1250.kernels.gfx1250_reference_ffn import ffn_input_layout_c1
        reference_layout = ffn_input_layout_c1(self.index_bound_max,self.tp)
        required_scratch = max(o+n for o,n in reference_layout.values())
        if self.scratch.numel() < required_scratch:
            # Allocate before StepMailboxes/graph binding so the additional
            # TP2 producer payloads and flags have one stable lifetime.
            self.scratch = torch.zeros((required_scratch+15)//16*16,
                                       dtype=torch.uint8,device=self.scratch.device)
        # The mother's final XRDY is eight bytes; step_begin clears 16-byte
        # vectors. Preserve every region offset and pad only the allocation.
        if self.scratch.numel() % 16:
            self.scratch = torch.zeros((self.scratch.numel() + 15) // 16 * 16,
                                       dtype=torch.uint8, device=self.scratch.device)
        dev = self.scratch.device
        # _key_row performs the raw selection load before choosing WINDOW.
        # Retain the rows-sized bound view, backed by every token's512-int span.
        self._window_selection_storage = torch.zeros(self.max_rows * 512,
            dtype=torch.int32, device=dev)
        self.no_bound = self._window_selection_storage[:self.max_rows]
        self._full['no_bound'] = self.no_bound
        self.diag = torch.zeros(256 * 8 + 16, dtype=torch.int32, device=dev)
        self.ingress = torch.empty_like(self.res_ffn)
        self.ingress_pre = torch.empty_like(self.gates_ffn[0])
        self.rank_tensor = torch.tensor([self.rank], dtype=torch.int32, device=dev)
        self.layer_tensors = [torch.tensor([s.layer_id], dtype=torch.int32, device=dev)
                              for s in model.topology]
        self.keys = [ReferenceK1Key(s.layer_id != 0 and b.engram is None,
                                   b.attn.indexer is not None)
                     for s, b in zip(model.topology, model.layers)]
        self.build = type("ReferenceRowsBuild", (), {"keys": self.keys})()
        self._dummy = torch.zeros(5120, dtype=torch.float32, device=dev)

    def _bind_agreed(self, bind):
        group = get_tp_group().cpu_group
        bind_agreed(bind, group)
        self.peers, self._finalizer = owned_peer_buffer(
            self, peer_bytes(1, self.tp), group, self.rank, self.tp, self.scratch.device,
        )
        self.mailboxes = StepMailboxes(self.peers, self.scratch, self.diag,
                                       debug=True, gfx1250=True)

    def prepare(self, rows):
        if rows != 1:
            return False
        if self._ready:
            return True
        if torch.cuda.is_current_stream_capturing():
            raise MonoUnsupported("prepare reference kernels before capture")
        def build():
            stream = torch.cuda.current_stream()
            self.front_kernels = {}
            for key in set(self.keys):
                self.front_kernels[key] = _CooperativeKernel(
                    build_reference_k1_c1(key.fold, key.feeds_index, tp=self.tp),
                    (self._dummy.data_ptr(),) * len(K1_ARGS) + (stream,),
                    reference_k1_symbol(key.fold, key.feeds_index, tp=self.tp), 1, 256,
                    torch.cuda.current_device(), smem=0, threads=256,
                    label="reference K1", cooperative_capture=True,
                    cooperative_dispatch=self.cooperative_dispatch,
                )
            flags = (False, True, True, True, True, True, True, True, True, True, True,
                     self.index_bound_max, self.model.config.sliding_window, self.tp)
            self.post_kernel = _CooperativeKernel(
                build_reference_attention_wob_c1(*flags),
                (self._dummy.data_ptr(),) * len(K2_ARGS) + (stream,),
                attention_wob_symbol_c1(*flags), 1, 256, torch.cuda.current_device(),
                smem=0, threads=256, label="reference complete K2", cooperative_capture=True,
                cooperative_dispatch=self.cooperative_dispatch,
            )
        bind_agreed(build, get_tp_group().cpu_group)
        self._ready = True
        return True

    def _bind_cache(self, cache, step):
        if cache.packed or cache.geometry.index_fp4 or cache.geometry.speculative_tokens:
            raise MonoUnsupported("reference c1 requires split BF16 KV and FP8 index without speculation")
        if step.width != 1 or step.positions.dtype != torch.int64:
            raise MonoUnsupported("reference c1 position ABI")
        if cache.geometry.index_block_rows != ip.BLOCK_ROWS:
            raise MonoUnsupported("reference index block geometry")
        for spec in self.model.topology:
            window = cache.geometry.window(spec.layer_id, cache.num_pages)
            if window.ring_slots != self.model.config.sliding_window:
                raise MonoUnsupported("reference attention ring differs from its build")
            if not 0 <= window.ring_start < cache.pool.shape[0]:
                raise MonoUnsupported("reference ring base outside pool")
            if window.ring_start > 2**31 - 1:
                raise MonoUnsupported("reference K2 ring offset exceeds int32 row ABI")
        dev = self.scratch.device
        self.positions = torch.empty_like(step.positions)
        self.page_table = torch.empty_like(step.block_tables)
        self.batch_ids = torch.empty_like(step.batch_ids)
        self.visible = {r: torch.empty_like(b) for r, b in step.visible.items()}
        ratios = {s.ratio for s, b in zip(self.model.topology, self.model.layers)
                  if b.attn.indexer is not None and s.candidate_source is None}
        self.unit_tables = {r: torch.empty(1, self.page_table.shape[1] *
            cache.geometry.index_blocks_per_page(r), dtype=torch.int32, device=dev)
            for r in ratios}
        self.candidate_output = (
            torch.empty_like(self.icand), torch.empty(1, dtype=torch.int32, device=dev),
        )
        self.ring_offsets = [torch.tensor([cache.geometry.window(s.layer_id, cache.num_pages).ring_start],
                                         dtype=torch.int64, device=dev) for s in self.model.topology]
        self.front_bindings, self.post_bindings, self.parameter_tables = [], [], []
        for spec, block, weights, key in zip(self.model.topology, self.model.layers, self.layers, self.keys):
            lid = spec.layer_id
            residual = self.res_attn if key.fold else self.ingress
            pre, post, comb = self.gates_attn
            w = weights
            a = dict(res_in=self.res_ffn if key.fold else self.ingress,
                pend=self.ffn_out, post_in=self.gates_ffn[1], comb_in=self.gates_ffn[2],
                pre_in=self.gates_ffn[0] if key.fold else self.ingress_pre,
                hc_fn=w["hc_attn_fn"], res_out=residual, scratch=self.scratch,
                diag=self.diag, layer=self.layer_tensors[lid], attn_w=w["attn_norm.weight"],
                normed=self.normed, weight=w["attn.wqkv_a.weight"], scale=w["attn.wqkv_a.weight_scale"],
                hc_scale=w["hc_attn_scale"], hc_base=w["hc_attn_base"], pre_out=pre, post_out=post,
                comb_out=comb, qn_w=w["attn.q_norm.weight"], kvn_w=w["attn.kv_norm.weight"],
                cos=(self.model.global_rope if spec.ratio else self.model.window_rope).cos_cache,
                sin=(self.model.global_rope if spec.ratio else self.model.window_rope).sin_cache,
                pos=self.positions, ring_rows=self.ring_rel, ring=cache.pool, ring_off=self.ring_offsets[lid],
                qr_out=self.qr, qrs_out=self.qr_scale, wqb=w["attn.wq_b.weight"],
                wqb_s=w["attn.wq_b.weight_scale"], q_out=self.q,
                iq_w=w["attn.indexer.wq_b.weight"] if key.feeds_index else self._dummy,
                iq_ws=w["attn.indexer.wq_b.weight_scale"] if key.feeds_index else self._dummy,
                projection_values=self._dummy,
                iw_w=w["attn.indexer.weights_proj.weight"] if key.feeds_index else self._dummy,
                iq_out=self.iq, iqs_out=self.iq_scale, iw_out=self.iw)
            front = tuple(a[n] for n in K1_ARGS)
            self.front_bindings.append(front)
            owner = spec.topk_owner if spec.ratio else None
            bound = self.no_bound if owner is None else (
                self.candidate_output[1] if self.model.topology[owner].candidate_source is not None
                else self.visible[self.model.topology[owner].ratio])
            if key.feeds_index:
                self.bounds[lid] = bound
            itab = (self.candidate_output[0] if spec.candidate_source is not None
                    else self.unit_tables.get(spec.ratio, self.no_bound))
            cfg = torch.tensor([int(key.feeds_index), cache.geometry.window(lid, cache.num_pages).ring_start,
                f32_bits(block.attn.softmax_scale * LOG2E), int(spec.produces_candidates),
                int(spec.candidate_source is not None), itab.shape[-1],
                cache.geometry.rows_per_page(spec.ratio or 1),
                cache.geometry.page_bytes // cache.geometry.row_bytes,
                cache.geometry.main_offset(spec.kv_owner) if spec.ratio else 0],
                dtype=torch.int32, device=dev)
            b = dict(q=self.q, pool=cache.pool, sel=self.no_bound if owner is None else self.irow[owner],
                sbound=bound, kmeta=self.key_meta, sink=block.attn.attn_sink, pos=self.positions,
                cos=a["cos"], sin=a["sin"], scratch=self.scratch, diag=self.diag,
                layer=self.layer_tensors[lid], config=cfg, out=self.ffn_out,
                woa=w["attn.wo_a.weight"], woa_s=w["attn.wo_a.weight_scale"],
                woa_out=self._dummy, wob=w["attn.wo_b.weight"], wob_s=w["attn.wo_b.weight_scale"],
                sym=self.peers.bytes, peers=self.peers.addresses, rank=self.rank_tensor)
            f = dict(res_in=residual, post_in=post, comb_in=comb, pre_in=pre,
                hc_fn=w["hc_ffn_fn"], res_out=self.res_ffn, attn_w=w["ffn_norm.weight"],
                normed=self.ffn_normed, reduce_out=self._dummy, gate_w=w["ffn.gate.weight"],
                bias=w["ffn.gate.e_score_correction_bias"],
                sgu=w["ffn.shared_experts.gate_up_proj.weight"],
                sgu_s=w["ffn.shared_experts.gate_up_proj.weight_scale"],
                hc_scale=w["hc_ffn_scale"], hc_base=w["hc_ffn_base"],
                pre_out=self.gates_ffn[0], post_out=self.gates_ffn[1], comb_out=self.gates_ffn[2],
                w13=w["ffn.experts.w13_weight"], w13_s=w["ffn.experts.w13_weight_scale"],
                w2=w["ffn.experts.w2_weight"], w2_s=w["ffn.experts.w2_weight_scale"],
                sw2=w["ffn.shared_experts.w2.weight"], sw2_s=w["ffn.shared_experts.w2.weight_scale"],
                out=self.ffn_out)
            ix = dict(plane=cache.index_units[spec.kv_owner].values if key.feeds_index else self._dummy,
                itab=itab, iq=self.iq, iw=self.iw, ibound=bound, icand=self.icand,
                table=self.page_table, ilog=self.ilog, ibmax=self.ibmax,
                isel=self.isel[lid] if key.feeds_index else self.no_bound,
                irow=self.no_bound if owner is None else self.irow[owner], icout=self.icand)
            owners = tuple(f[n] for n in FFN_DOWN_ARGS) + tuple(ix[n] for n in INDEX_ARGS) + tuple(b[n] for n in INDEX_ATTENTION_ARGS)
            table = torch.tensor([t.data_ptr() for t in owners], dtype=torch.int64, device=dev)
            self.parameter_tables.append((table, owners))
            b["reduce_out"] = table
            self.post_bindings.append(tuple(b[n] for n in K2_ARGS))
        _diagnostics(cache, 1)
        # Every capture is local and contains no dispatch. Agreement precedes
        # the first step fence and any K1/K2 peer wait.
        for key, front in zip(self.keys, self.front_bindings):
            self.front_kernels[key].prepare_graph(front)
        for post in self.post_bindings:
            self.post_kernel.prepare_graph(post)
        self._cache = cache

    def _use_rows(self, rows):
        if rows != 1 or not self._ready:
            raise MonoUnsupported("reference runner was not prepared for c1")
        metadata = get_forward_context().attn_metadata
        cache, step = metadata.cache, metadata.step
        if self._cache is None:
            if torch.cuda.is_current_stream_capturing():
                raise MonoUnsupported("bind reference cache outside capture")
            bind_agreed(lambda: self._bind_cache(cache, step), get_tp_group().cpu_group)
        elif cache is not self._cache:
            raise MonoUnsupported("reference runner cannot replace a live cache")
        if (step.block_tables.shape != self.page_table.shape
            or step.positions.shape != self.positions.shape
            or step.batch_ids.shape != self.batch_ids.shape):
            raise MonoUnsupported("reference c1 metadata storage geometry changed")
        if getattr(cache, "_gfx1250_k1_failed", False):
            raise RuntimeError("failed reference cache cannot dispatch")
        self.positions.copy_(step.positions)
        self.page_table.copy_(step.block_tables)
        self.batch_ids.copy_(step.batch_ids)
        for ratio, buf in self.visible.items():
            buf.copy_(step.visible[ratio])
        _diagnostics(cache, 1)[0].zero_()

    def _front(self, layer_id, key, weights, state, cache, step, rope):
        if (state.pending is not None) != key.fold:
            raise MonoUnsupported(f"layer {layer_id}: owed post differs from frozen build")
        if not key.fold:
            self.ingress.copy_(state.residual)
            self.ingress_pre.copy_(state.pre_mix)
        self.front_kernels[key](self.front_bindings[layer_id], torch.cuda.current_stream().cuda_stream)
        return (self.res_attn if key.fold else self.ingress), self.gates_attn

    def _layer_post(self, spec, block, weights, residual, gates, step, cache, rope):
        # The mother's FULL table materialization remains after compression.
        if block.attn.indexer is not None and spec.candidate_source is None:
            unit_table(self.page_table, self.batch_ids,
                       cache.geometry.index_blocks_per_page(spec.ratio),
                       out=self.unit_tables[spec.ratio])
        self.post_kernel(self.post_bindings[spec.layer_id], torch.cuda.current_stream().cuda_stream)

    def _candidate_block_table(self, *args, **kwargs):
        return candidate_block_table(*args, **kwargs, out=self.candidate_output)

    def _raise_on_given_up_waits(self, rows):
        # Device-only summary; the native output transfer accepts it later.
        from atom.models.deepseek_v41.mono.gfx1250.gfx1250_reference_status import finish_reference_step
        finish_reference_step(get_forward_context().attn_metadata, self.diag,
                              self.mailboxes._state)


class ReferenceDecodeModel(nn.Module):
    def __init__(self, model, runner):
        super().__init__()
        self.model, self.runner = model, runner

    def forward(self, input_ids, positions, inputs_embeds=None):
        if input_ids.numel() == 1 and step_supported(input_ids, inputs_embeds):
            self.runner.prepare(1)
            return self.runner.forward(input_ids, positions)
        if inputs_embeds is None:
            return self.model(input_ids, positions)
        return self.model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states):
        return self.model.compute_logits(hidden_states)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def install_reference_decode(model, config, drafter, metadata_builder):
    """Install immutable decode copies before ModelRunner profiles KV memory."""
    why = config_refusal(config, arch="gfx1250", enabled=True)
    if why is not None:
        raise MonoUnsupported(f"requested reference backend refused: {why}")
    if (config.tensor_parallel_size not in (2, 4) or drafter is not None
        or config.index_cache_dtype != "fp8"
        or len(model.layers) != 40 or config.max_num_seqs != 1):
        raise MonoUnsupported("reference backend currently requires 40-layer TP2/TP4 c1 FP8 index")
    if any(os.getenv(flag, "0") == "1" for flag in (
        "ATOM_DSV41_GFX1250_K1", "ATOM_DSV41_GFX1250_K2", "ATOM_DSV41_GFX1250_K2_MERGED",
    )):
        raise MonoUnsupported("reference and historical gfx1250 backends cannot be combined")
    if config.compilation_config.level != 0 or any(getattr(config, name, False) for name in (
        "enable_decoder_swa_bounded_replay", "enable_encoder_swa_bounded_replay", "enable_rapidserve",
    )):
        raise MonoUnsupported("reference backend requires level0 eager/FULL graph without bounded replay")
    if config.tensor_parallel_size == 4 and os.getenv("ATOM_DSV41_GFX1250_REFERENCE_TP4_TRIAL", "0") != "1":
        raise MonoUnsupported("TP4 reference remains a trial until four-rank graph/numerical acceptance")
    bound = []
    bind_agreed(lambda: bound.extend(bind_reference_model(model, config.tensor_parallel_size)), get_tp_group().cpu_group)
    runner = Gfx1250ReferenceRunner(model, bound, config.max_model_len)
    logger.info("gfx1250 reference mhtjstest installed %d layers before KV sizing", len(bound))
    return ReferenceDecodeModel(model, runner)
