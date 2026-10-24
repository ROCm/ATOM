# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Opt-in local K1 with native attention, FFN and TP communication.

Level0 eager/FULL graph binding, FP8 index and BF16 split cache only. The model owns
one bounded buffer set per row width. Calls and graph replays must be ordered
on one compute stream; warm up both widths before capture. This adapter does
uses the model's native layer loop; opt-in merged K2 shares gfx950 peer/mailbox
lifetime through Gfx1250DecodeContext.
The native ModelRunner accepts diagnostics before returning the corresponding
output, immediately or at the existing deferred-output boundary.
"""

import os
from dataclasses import replace

import torch
import torch.distributed as dist
import triton
import triton.language as tl
from aiter.dist.parallel_state import get_tp_group

from atom.model_ops.v4_kernels import sparse_attn_v4_paged_2src
from atom.models.deepseek_v41.mono.gfx1250_diagnostics import (
    _diagnostics,
    _raise_diagnostic_values,
)
from atom.models.deepseek_v41.mono.kernels.gfx1250_attn_pre import (
    STATE_SIZE,
    attn_pre_gfx1250,
)
from atom.models.deepseek_v41.mono.step_meta import KMETA, write_step_meta


@triton.jit
def _guard_outputs(
    state,
    diagnostics,
    query,
    hn,
    residual,
    pre,
    post,
    comb,
    iq8,
    swiw,
    ring,
    ring_rows,
    ROWS: tl.constexpr,
    HEADS: tl.constexpr,
    INDEX: tl.constexpr,
    LAYER: tl.constexpr,
    RING_OFF: tl.constexpr,
    POOL_ROWS: tl.constexpr,
    BLOCK: tl.constexpr,
    STATE_WORDS: tl.constexpr = STATE_SIZE,
):
    """Publish the layer result; on failure make native numeric inputs safe."""
    code = tl.load(state + STATE_WORDS - 1)
    tl.store(diagnostics + LAYER, code)
    if code != 0:
        for start in range(0, ROWS * 4 * 5120, BLOCK):
            x = start + tl.arange(0, BLOCK)
            tl.store(residual + x, 0, x < ROWS * 4 * 5120)
            tl.store(query + x, 0, x < ROWS * HEADS * 512)
            tl.store(hn + x, 0, x < ROWS * 5120)
            tl.store(pre + x, 0, x < ROWS * 4)
            tl.store(post + x, 0, x < ROWS * 4)
            tl.store(comb + x, 0, x < ROWS * 16)
            if INDEX:
                tl.store(iq8 + x, 0.0, x < ROWS * 32 * 128)
                tl.store(swiw + x, 0, x < ROWS * 32)
        # The failed step is discarded. Only clear its legal current rows;
        # malformed metadata must never become a cleanup address.
        for row in range(ROWS):
            rr = tl.load(ring_rows + row).to(tl.int64)
            valid = (rr >= 0) & (rr < POOL_ROWS - RING_OFF)
            x = tl.arange(0, 512)
            tl.store(ring + (rr + RING_OFF) * 512 + x, 0, valid)


@triton.jit
def _reduce_diagnostics(diagnostics, sticky, LAYERS: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.arange(0, BLOCK)
    codes = tl.load(diagnostics + x, x < LAYERS, other=0)
    # -1 means a layer failed to execute its guard, and is a step failure too.
    failed = tl.maximum(tl.load(sticky), tl.max(tl.abs(codes), 0))
    tl.store(diagnostics + LAYERS, failed)
    tl.store(sticky, failed)


def finish_k1_forward(metadata):
    step, cache = metadata.step, metadata.cache
    if not step.decode or step.width not in (1, 2):
        return
    codes, _ = _diagnostics(cache, step.width)
    layers = cache.geometry.layers
    _reduce_diagnostics[(1,)](
        codes, cache._gfx1250_diagnostic_sticky, layers, triton.next_power_of_2(layers)
    )
    # Both ranks observe failure before entering the LM-head collectives or
    # the next forward. A local-only CPU exception could strand its TP peer.
    group = get_tp_group()
    if group.world_size > 1:
        dist.all_reduce(codes[-1:], op=dist.ReduceOp.MAX, group=group.device_group)
    cache._gfx1250_diagnostic_sticky.copy_(codes[-1:])


def check_k1_diagnostics(metadata, rows):
    """Called outside graphs by native ModelRunner, before sampling/reuse."""
    if not metadata.step.decode or rows not in (1, 2):
        return
    cache = metadata.cache
    stores = cache.__dict__.get("_gfx1250_k1_diagnostics", {})
    if rows not in stores:
        if metadata.step.requests:
            raise RuntimeError("[gfx1250 K1 runner] no diagnostics for decode")
        return
    codes, host = stores[rows]
    host.copy_(codes, non_blocking=False)
    values = host.tolist()
    _raise_diagnostic_values(cache, values)


def begin_k1_forward(metadata):
    """Open one metadata preparation scope and join Engram's spinning gather."""
    metadata.step._gfx1250_k1_ready = set()
    if getattr(metadata.cache, "_gfx1250_k1_failed", False):
        raise RuntimeError("[gfx1250 K1 runner] failed cache cannot be reused")
    if metadata.step.width in (1, 2):
        _diagnostics(metadata.cache, metadata.step.width)[0][:-1].fill_(-1)
        context = getattr(metadata.cache, "_gfx1250_decode", None)
        if context is not None:
            context.begin_step(metadata.step.width, metadata.step)
    staged = metadata.engram_embeddings
    if getattr(staged, "join", None) is not None:
        staged.join()


def _step_buffers(cache, rope, rows):
    """Allocate stable metadata/RoPE addresses without submitting their work."""
    stores = cache.__dict__.setdefault("_gfx1250_k1_meta", {})
    if rows not in stores:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("[gfx1250 K1 runner] warm metadata before capture")
        dev = cache.pool.device
        stores[rows] = (
            torch.empty(rows, device=dev, dtype=torch.int32),
            torch.empty(rows, KMETA, device=dev, dtype=torch.int32),
        )
    tables = rope.__dict__.setdefault("_gfx1250_k1_tables", {})
    if rows not in tables:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("[gfx1250 K1 runner] warm RoPE tables before capture")
        tables[rows] = tuple(
            torch.empty_like(t[:rows]) for t in (rope.cos_cache, rope.sin_cache)
        )
    return (*tables[rows], *stores[rows])


def _step_inputs(cache, step, rope, rows):
    # Python readiness controls capture; preparation executes on every replay.
    ready = step._gfx1250_k1_ready
    cos, sin, rr, keys = _step_buffers(cache, rope, rows)
    if ("ring", rows) not in ready:
        window = cache.geometry.window(0, cache.num_pages)
        if window.ring_stride < window.ring_slots:
            raise ValueError("[gfx1250 K1 runner] unsupported ring row mapping")
        write_step_meta(step, cache.geometry, window, rr, keys)
        ready.add(("ring", rows))
    key = ("rope", id(rope), rows)
    if key not in ready:
        positions = step.positions.reshape(-1).to(torch.int64)
        torch.index_select(rope.cos_cache, 0, positions, out=cos)
        torch.index_select(rope.sin_cache, 0, positions, out=sin)
        ready.add(key)
    return cos, sin, rr


class LocalK1:
    def __init__(self, block):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("[gfx1250 K1 runner] warm layer before capture")
        attn = block.attn
        if attn.heads not in (16, 32) or attn.head_dim != 512:
            raise ValueError("[gfx1250 K1 runner] unsupported attention geometry")
        if block.hc_options != {
            "rms_eps": 1e-20,
            "hc_eps": 1e-6,
            "sinkhorn_iters": 20,
            "post_mult": 2.0,
        }:
            raise ValueError("[gfx1250 K1 runner] unsupported mHC configuration")
        if (
            block.attn_norm.eps != 1e-20
            or attn.q_norm.eps != 1e-20
            or attn.kv_norm.eps != 1e-20
        ):
            raise ValueError("[gfx1250 K1 runner] unsupported norm epsilon")
        if attn.compress_stream is not None or attn.index_stream is not None:
            raise ValueError("[gfx1250 K1 runner] requires side streams disabled")
        self.block, self.buffers, self.bindings = block, {}, {}
        self.k2 = None
        if os.getenv("ATOM_DSV41_GFX1250_K2", "0") == "1":
            from atom.models.deepseek_v41.mono.gfx1250_k2_runner import LocalK2

            self.k2 = LocalK2(block)
        params = dict(block.named_parameters())
        self.weights = {
            k: params[k]
            for k in (
                "hc_attn_fn",
                "hc_attn_scale",
                "hc_attn_base",
                "attn_norm.weight",
                "attn.q_norm.weight",
                "attn.kv_norm.weight",
            )
        }
        for target, prefix in (("a", "attn.wqkv_a"), ("b", "attn.wq_b")):
            self.weights["w" + target] = params[prefix + ".weight"]
            self.weights["s" + target] = params[prefix + ".weight_scale"]
        self.feeds = attn.indexer is not None
        if self.feeds:
            self.weights.update(
                wiq=params["attn.indexer.wq_b.weight"],
                siq=params["attn.indexer.wq_b.weight_scale"],
                wiw=params["attn.indexer.weights_proj.weight"],
            )

    def _buffers(self, rows, device):
        if rows in self.buffers:
            return self.buffers[rows]
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("[gfx1250 K1 runner] warm row width before capture")
        bf, f32, fp8 = torch.bfloat16, torch.float32, torch.float8_e4m3fn
        heads = self.block.attn.heads
        specs = {
            "rout": ((rows, 4, 5120), bf),
            "collapse": ((rows, 5120), bf),
            "partial": ((rows, 160, 26), f32),
            "pre": ((rows, 4), f32),
            "post": ((rows, 4), f32),
            "comb": ((rows, 4, 4), f32),
            "hn": ((rows, 5120), bf),
            "hx": ((rows, 5120), fp8),
            "hs": ((rows, 160), torch.uint8),
            "qkv": ((rows, 1792), bf),
            "q": ((rows, 1280), fp8),
            "qs": ((rows, 40), torch.uint8),
            "query": ((rows, heads, 512), bf),
            "kv": ((rows, 512), bf),
            "iq": ((rows, 32, 128), bf),
            "iw": ((rows, 32), bf),
            "iq8": ((rows, 32, 128), fp8),
            "iqs": ((rows, 32, 1), f32),
            "swiw": ((rows, 32), f32),
            "res_input": ((rows, 4, 5120), bf),
            "pending_input": ((rows, 5120), bf),
            "pre_input": ((rows, 4), f32),
            "post_input": ((rows, 4), f32),
            "comb_input": ((rows, 4, 4), f32),
        }
        b = {
            k: torch.empty(shape, dtype=dtype, device=device)
            for k, (shape, dtype) in specs.items()
        }
        b.update(
            state=torch.zeros(STATE_SIZE, dtype=torch.int32, device=device),
            dummy_y=torch.zeros(rows, 5120, dtype=bf, device=device),
            dummy_post=torch.zeros(rows, 4, dtype=f32, device=device),
            dummy_comb=torch.zeros(rows, 4, 4, dtype=f32, device=device),
        )
        self.buffers[rows] = b
        return b

    def prepare_binding(self, rows, fold, inputs, cache, rope):
        """Validate real addresses and capture the eager node without replay."""
        context = self.k2.context
        b = self.buffers[rows]
        cos, sin, rr, _ = _step_buffers(cache, rope, rows)
        binding = attn_pre_gfx1250(
            *inputs, self.weights, cos, sin, b, context.rings[self.block.attn.spec.layer_id], rr,
            fold=fold, feeds_index=self.feeds, ring_off=0, _prepare_only=True,
            generation=context.metadata[self.block.attn.spec.layer_id][1:],
        )
        binding.runtime.prepare_graph(binding.tensors)
        self.bindings[(rows, fold)] = binding
        return binding

    def guard(self, rows, cache, codes, residual, pre, post, comb, *, prepare=False):
        b = self.buffers[rows]
        rr = cache._gfx1250_k1_meta[rows][0]
        args = (b["state"], codes, b["query"], b["hn"], residual, pre, post, comb,
                b["iq8"], b["swiw"], cache.pool, rr)
        options = dict(ROWS=rows, HEADS=self.block.attn.heads, INDEX=self.feeds,
            LAYER=self.block.attn.spec.layer_id,
            RING_OFF=cache.geometry.window(self.block.attn.spec.layer_id, cache.num_pages).ring_start,
            POOL_ROWS=cache.pool.shape[0], BLOCK=256)
        if prepare:
            _guard_outputs.warmup(*args, grid=(1,), **options)
        else:
            _guard_outputs[(1,)](*args, **options)

    def forward(self, state, cache, step, rope, embeddings, image_mask):
        block = self.block
        if block.engram is not None:
            state = state.settle()
            state = replace(
                state,
                residual=block.engram_forward(state.residual, embeddings, image_mask),
            )
        rows = state.residual.shape[-3]
        b = self._buffers(rows, state.residual.device)
        cos, sin, rr = _step_inputs(cache, step, rope, rows)
        fold = state.pending is not None
        context = self.k2.context if self.k2 is not None else None
        if (context is not None and block.attn.spec.layer_id > 0 and block.engram is None) or (context is None and torch.cuda.is_current_stream_capturing()):
            # The outer graph owns these stable allocations for its lifetime.
            # No input copies or new per-token inner graph cache entries.
            inputs = (
                state.residual.reshape(rows, 4, 5120),
                state.pending.reshape(rows, 5120) if fold else b["dummy_y"],
                state.pre_mix.reshape(rows, 4),
                state.post_mix.reshape(rows, 4) if fold else b["dummy_post"],
                state.combination.reshape(rows, 4, 4) if fold else b["dummy_comb"],
            )
        else:
            # Eager dispatch caches by every tensor address. Keep one input set
            # instead of retaining fresh graphs for native FFN outputs per token.
            b["res_input"].copy_(state.residual.reshape(rows, 4, 5120))
            b["pre_input"].copy_(state.pre_mix.reshape(rows, 4))
            if fold:
                b["pending_input"].copy_(state.pending.reshape(rows, 5120))
                b["post_input"].copy_(state.post_mix.reshape(rows, 4))
                b["comb_input"].copy_(state.combination.reshape(rows, 4, 4))
            inputs = (
                b["res_input"],
                b["pending_input"] if fold else b["dummy_y"],
                b["pre_input"],
                b["post_input"] if fold else b["dummy_post"],
                b["comb_input"] if fold else b["dummy_comb"],
            )
        ring_off = cache.geometry.window(block.attn.spec.layer_id, cache.num_pages).ring_start
        ring = context.rings[block.attn.spec.layer_id] if context else cache.pool
        binding_key = (rows, fold)
        if context is not None:
            if binding_key not in self.bindings:
                raise RuntimeError("[gfx1250 K1 runner] prepare the real binding before the TP chain")
            residual, pre, post, comb, query, _ = self.bindings[binding_key]()
        else:
            residual, pre, post, comb, query, _ = attn_pre_gfx1250(
                *inputs, self.weights, cos, sin, b, ring, rr,
                fold=fold, feeds_index=self.feeds, ring_off=ring_off,
            )
        codes, _ = _diagnostics(cache, rows)
        self.guard(rows, cache, codes, residual, pre, post, comb)
        # K1's ring write and index inputs are complete on this compute stream.
        # Compressor and scorer retain the original model/cache semantics.
        attn = block.attn
        if self.feeds:
            attn._compress_batch(b["hn"].unsqueeze(0), cache, step, rope)
            if context is None:
                attn.indexer.score_quantized(b["iq8"], b["swiw"], cache, step)
        if self.k2 is not None:
            return self.k2.forward(
                query, residual, pre, post, comb, cache, step, rope, codes, image_mask
            )
        prefix, pptr, second, sptr = cache.attention_indices(attn.spec, step)
        output = sparse_attn_v4_paged_2src(
            query,
            cache.pool,
            prefix,
            pptr,
            cache.ring_view(attn.spec.layer_id),
            second,
            sptr,
            attn.attn_sink,
            attn.softmax_scale,
            out=query,
        )
        output = attn._project_out(
            output.unsqueeze(0), rope, cache.rope_positions(step)
        )
        return block.ffn_forward(
            output,
            residual.unsqueeze(0),
            pre.unsqueeze(0),
            post.unsqueeze(0),
            comb.unsqueeze(0),
        )


def try_decode_k1(block, state, metadata, rope, embeddings, image_mask, queries):
    step, cache = metadata.step, metadata.cache
    rows = state.residual.shape[-3]
    if (
        not step.requests
        or not step.decode
        or rows not in (1, 2)
        or queries is not None
    ):
        return None
    if cache.packed or cache.geometry.index_fp4:
        raise ValueError("[gfx1250 K1 runner] requires BF16 split KV and FP8 index")
    if step.tentative or cache.geometry.speculative_tokens:
        raise ValueError("[gfx1250 K1 runner] speculative decode is not supported")
    if torch.compiler.is_compiling():
        raise RuntimeError("[gfx1250 K1 runner] requires compilation level0")
    device = state.residual.device
    if torch.cuda.get_device_properties(device).gcnArchName.split(":")[0] != "gfx1250":
        raise ValueError("[gfx1250 K1 runner] requires gfx1250")
    runner = block.__dict__.get("_gfx1250_k1")
    if runner is None:
        runner = LocalK1(block)
        block.__dict__["_gfx1250_k1"] = runner
    return runner.forward(state, cache, step, rope, embeddings, image_mask)
