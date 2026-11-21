# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""The draft context path's host side on one GPU without the engine: the real
``KimiK3DSpark.write_context_kv`` (the override and the base it falls back
to), the real ``K3DSparkDecoderLayer`` / ``K3DSparkMLAAttention`` context
writes, the real ``LinearBase.forward``, ``RMSNorm`` and
``MLAAttention.write_context_kv_latent``, and the real
``step_gap.draft_context``, on modules built without their constructors (those
need the engine's config and its process groups): the weights, the rope cache
and the KV caches are put where the constructors would leave them.

Two models share every weight and own a set of KV caches each: one with the
switch's attribute off (production's path), one with it on. Per context width
the five caches are compared whole (one projection launch and one store
launch a step); then the store's fallbacks (the part off, a layer on another
cache layout, a cache rebound), the steps the kernels refuse (a dummy run, a
prefill step, a width past 64, a PCP split, a capture in progress, a width
whose build is refused, a quantized context they cannot read) and a model
they cannot bind, each against production. At the end: the lines of
``draft_context.py`` that ran.

    python tools/kimi_k3_step_gap_ctx_runner_check.py
    env: WIDTHS (8,16,24,32,40,48,56,64,5,10,15,20,25,30,35)
"""

import json
import os
import sys
from types import SimpleNamespace as NS

os.environ["ATOM_KIMI_MONO_DRAFT_CONTEXT"] = "1"
TRACED = "step_gap/draft_context.py"
RAN = set()


def _trace(frame, event, arg):
    name = frame.f_code.co_filename
    if isinstance(name, str) and name.endswith(TRACED):
        RAN.add(frame.f_lineno)

        def local(frame, event, arg):
            if event == "line":
                RAN.add(frame.f_lineno)
            return local

        return local
    return None


sys.settrace(_trace)  # before the module is imported

import torch
from aiter import QuantType, dtypes
from aiter.ops.shuffle import shuffle_weight
from torch import nn

from atom.model_ops import layernorm
from atom.model_ops.attention_mla import MLAAttention
from atom.model_ops.linear import LinearBase
from atom.models.kimi_k3.mono import draft_dispatch
from atom.models.kimi_k3.step_gap import ctx_kv_store, draft_context
from atom.models.kimi_k3_dspark import (
    K3DSparkDecoderLayer,
    K3DSparkMLAAttention,
    KimiK3DSpark,
)
from atom.mono.runtime.consensus import MonoUnsupported
from atom.quantization.quark.utils import quant_weight_online
from atom.utils import forward_context

WIDTHS = [
    int(r)
    for r in os.environ.get(
        "WIDTHS", "8,16,24,32,40,48,56,64,5,10,15,20,25,30,35"
    ).split(",")
]
HIDDEN, TAPS, LAYERS = 7168, 5, 5
Q_LORA, KV_LORA, ROPE = 1536, 512, 64
QKV = Q_LORA + KV_LORA + ROPE
SLOTS = 4096
EPS = 1e-5
KEEP = []

dev = torch.device("cuda", torch.cuda.current_device())
gen = torch.Generator(device=dev).manual_seed(23)
layernorm.get_tensor_model_parallel_world_size = lambda: 1
FC = NS(context=NS(is_dummy_run=False, is_prefill=False))
forward_context.get_forward_context = lambda: FC


def bf16(*shape, scale=1.0, shift=0.0):
    return (torch.randn(shape, generator=gen, device=dev) * scale + shift).to(
        torch.bfloat16
    )


def fp8_linear(rows, k, scale=0.02):
    q, s = quant_weight_online(
        bf16(rows, k, scale=scale),
        QuantType.per_1x128,
        dtypes.fp8,
        scale_dtype=dtypes.fp8_e8m0,
    )
    pad = -rows % 128
    if pad:
        q = torch.nn.functional.pad(q.view(torch.uint8), (0, 0, 0, pad)).view(q.dtype)
    return shuffle_weight(q, layout=(16, 16)).contiguous(), s.contiguous()


def rope():
    from aiter.rotary_embedding import get_rope

    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    r = get_rope(
        64, rotary_dim=64, max_position=1048576, base=50000.0, is_neox_style=False,
        rope_scaling={
            "rope_type": "deepseek_yarn", "factor": 32.0, "original_max_position_embeddings": 32768,
            "beta_fast": 32, "beta_slow": 1, "mscale": 1.0, "mscale_all_dim": 1.0,
        },
    ).to(dev)  # fmt: skip
    torch.set_default_dtype(default)
    return r


ROTARY = rope()
ONE = torch.ones(1, dtype=torch.float32, device=dev)


def linear(weight, scale, prefix, rows):
    """A ``LinearBase`` as its constructor and loader leave it: the plain
    BF16 GEMM (``scale`` None) or the preshuffled per-1x128 E8M0 FP8 one, its
    output rows padded to 128."""
    lin = LinearBase.__new__(LinearBase)
    nn.Module.__init__(lin)
    w = nn.Parameter(weight, requires_grad=False)
    w.is_shuffled = scale is not None
    lin.weight = w
    lin.register_parameter("bias", None)
    lin.prefix = prefix
    lin.native_a8_group_rows = None
    lin.tp_dim, lin.tp_size, lin.reduce_results = None, 1, False
    lin.input_size, lin.output_size = weight.shape[1], weight.shape[0]
    lin.is_output_padded = weight.shape[0] != rows
    lin._output_size_before_padding = rows
    if scale is None:
        lin.quant_type, lin.params_dtype = QuantType.No, torch.bfloat16
    else:
        lin.quant_type, lin.params_dtype = QuantType.per_1x128, dtypes.fp8
        lin.scale_dtype = dtypes.fp8_e8m0
        lin.weight_scale = nn.Parameter(scale, requires_grad=False)
        lin.quant_func = None
    return lin


def rmsnorm(weight):
    norm = layernorm.RMSNorm(weight.numel(), eps=EPS)
    norm.weight.data = weight
    return norm


class Attn(K3DSparkMLAAttention):
    """The attention's context-write state, without the constructor."""

    def __init__(self, shared):
        nn.Module.__init__(self)
        self.hidden_size, self.q_lora_rank = HIDDEN, Q_LORA
        self.kv_lora_rank, self.qk_rope_head_dim = KV_LORA, ROPE
        self.fused_qkv_a_proj = shared["linear"]
        self.kv_a_layernorm = shared["norm"]
        impl = MLAAttention.__new__(MLAAttention)
        nn.Module.__init__(impl)
        impl._ctx_kv_fusion_enabled = True
        impl.kv_lora_rank, impl.qk_rope_head_dim = KV_LORA, ROPE
        impl.rotary_emb = ROTARY
        impl._k_scale_device = ONE
        impl.kv_cache_dtype = "fp8"
        self.mla_attn = NS(impl=impl, kv_cache=None)
        self.rebind(torch.zeros(SLOTS, 1, KV_LORA + ROPE, dtype=dtypes.fp8, device=dev))

    def rebind(self, cache):
        """A cache bound as the engine binds its pool's."""
        self.cache = self.mla_attn.kv_cache = cache


class Layer(K3DSparkDecoderLayer):
    def __init__(self, shared):
        nn.Module.__init__(self)
        self.self_attn = Attn(shared)


class Model(KimiK3DSpark):
    def __init__(self, shared, on):
        nn.Module.__init__(self)
        self.context_proj = shared["context_proj"]
        self.context_norm = shared["context_norm"]
        self.layers = nn.ModuleList([Layer(s) for s in shared["layers"]])
        self._try_mono_context = on
        self.mono_context = None

    def caches(self):
        return [layer.self_attn.cache for layer in self.layers]


def shared_weights(bias_on=None):
    layers = []
    for i in range(LAYERS):
        w, s = fp8_linear(QKV, HIDDEN)
        lin = linear(w, s, f"layers.{i}.self_attn.fused_qkv_a_proj", QKV)
        if bias_on == i:
            lin.bias = nn.Parameter(
                torch.zeros(lin.output_size, dtype=torch.bfloat16, device=dev),
                requires_grad=False,
            )
        layers.append(
            {"linear": lin, "norm": rmsnorm(bf16(KV_LORA, scale=0.1, shift=1.0))}
        )
    return {
        "context_proj": linear(
            bf16(HIDDEN, TAPS * HIDDEN, scale=0.01), None, "context_proj", HIDDEN
        ),
        "context_norm": rmsnorm(bf16(HIDDEN, scale=0.1, shift=1.0)),
        "layers": layers,
    }


def inputs(rows):
    taps = [bf16(rows, HIDDEN, scale=0.5) for _ in range(TAPS)]
    slots = torch.randperm(SLOTS, generator=gen, device=dev)[:rows].long()
    if rows > 2:
        slots[1::3] = -1
    positions = torch.randint(0, 200000, (rows,), generator=gen, device=dev).long()
    return taps, positions, slots


def bits(t):
    return t.contiguous().view(torch.uint8)


def clear(*models):
    for m in models:
        for c in m.caches():
            c.zero_()


def same(a, b):
    torch.cuda.synchronize()
    return all(torch.equal(bits(x), bits(y)) for x, y in zip(a.caches(), b.caches()))


def written(m):
    return sum(int((bits(c) != 0).any(dim=-1).sum()) for c in m.caches())


LAUNCHES = []
_write = draft_context.DraftContextRunner.write


def counted_write(self, kernel, ctx_hidden, positions, slot_mapping):
    done = _write(self, kernel, ctx_hidden, positions, slot_mapping)
    LAUNCHES.append((ctx_hidden.size(0), done))
    return done


draft_context.DraftContextRunner.write = counted_write
STORES = []
_store_layers = ctx_kv_store.store_layers


def counted_store(kv_lora, *args):
    STORES.append(tuple(kv_lora.shape))
    return _store_layers(kv_lora, *args)


ctx_kv_store.store_layers = counted_store
RESULTS = []


def report(what, **checks):
    ok = all(bool(v) for v in checks.values())
    RESULTS.append(ok)
    print(
        json.dumps(
            {
                "case": what,
                "ok": ok,
                **{
                    k: (v if isinstance(v, (int, str)) else bool(v))
                    for k, v in checks.items()
                },
            }
        ),
        flush=True,
    )


def step(prod, mono, rows, **kw):
    taps, positions, slots = inputs(rows)
    clear(prod, mono)
    before = len(LAUNCHES)
    prod.write_context_kv(taps, positions, slots, **kw)
    mono.write_context_kv(taps, positions, slots, **kw)
    return taps, positions, slots, len(LAUNCHES) - before


def main():
    shared = shared_weights()
    KEEP.append(shared)
    prod, mono = Model(shared, False), Model(shared, True)
    for rows in WIDTHS:
        stores = len(STORES)
        *_, launched = step(prod, mono, rows)
        runner = mono.mono_context
        kernel = runner.kernels.get(rows) if runner is not None else None
        report(
            f"{rows} rows", caches_equal=same(prod, mono), launched_once=launched == 1,
            one_store=STORES[stores:] == [(LAYERS, rows, KV_LORA + ROPE)],
            kernel=getattr(kernel, "name", ""), rows_written=written(prod),
            production_untouched=prod.mono_context is None,
        )  # fmt: skip
    runner = mono.mono_context
    report(
        "refusals so far",
        none=not runner.refused,
        parts=",".join(sorted(runner.parts)) == "kv,store",
    )

    # the other shape of the draft's interface (the runtime image's): no slots
    # passed, the layers read the target's slot mapping off the forward context
    taps, positions, slots = inputs(24)
    clear(prod, mono)
    prod.write_context_kv(taps, positions, slots)
    FC.attn_metadata = NS(slot_mapping=torch.cat([slots, slots.new_full((5,), -1)]))
    before, stores = len(LAUNCHES), len(STORES)
    mono.write_context_kv(taps, positions)
    report(
        "no slots passed: the forward context's, cut to the rows",
        caches_equal=same(prod, mono),
        launched_once=LAUNCHES[before:] == [(24, True)],
        one_store=len(STORES) == stores + 1,
        rows_written=written(mono),
    )

    # the engine binds a pool's cache after warmup: the plan follows it
    table = runner._store_plan.tab.tolist()
    for m in (prod, mono):
        m.layers[1].self_attn.rebind(
            torch.zeros(SLOTS, 1, KV_LORA + ROPE, dtype=dtypes.fp8, device=dev)
        )
    stores = len(STORES)
    *_, launched = step(prod, mono, 16)
    report(
        "a rebound cache is written, in one launch", caches_equal=same(prod, mono), launched_once=launched == 1,
        one_store=len(STORES) == stores + 1, table_moved=runner._store_plan.tab.tolist() != table,
        rows_written=written(mono),
    )  # fmt: skip

    # a layer on another cache layout: every layer writes through its attention
    mixed_p, mixed_m = Model(shared, False), Model(shared, True)
    for m in (mixed_p, mixed_m):
        m.layers[3].self_attn.rebind(
            torch.zeros(SLOTS // 2, 2, KV_LORA + ROPE, dtype=dtypes.fp8, device=dev)
        )
    stores = len(STORES)
    *_, launched = step(mixed_p, mixed_m, 16)
    report(
        "a layer on another cache layout: production's five writes", caches_equal=same(mixed_p, mixed_m),
        launched_once=launched == 1, no_store_launch=len(STORES) == stores, rows_written=written(mixed_m),
    )  # fmt: skip

    os.environ["ATOM_KIMI_MONO_DRAFT_CONTEXT"] = "kv"
    kv_only = Model(shared, True)
    stores = len(STORES)
    *_, launched = step(prod, kv_only, 16)
    os.environ["ATOM_KIMI_MONO_DRAFT_CONTEXT"] = "1"
    report(
        "the store part off: production's five writes", caches_equal=same(prod, kv_only), launched_once=launched == 1,
        no_store_launch=len(STORES) == stores, parts=",".join(sorted(kv_only.mono_context.parts)) == "kv",
    )  # fmt: skip

    FC.context.is_dummy_run = True
    *_, launched = step(prod, mono, 16)
    report(
        "a dummy run writes nothing",
        untouched=written(prod) == 0 and written(mono) == 0,
        no_launch=launched == 0,
    )
    FC.context.is_dummy_run = False

    FC.context.is_prefill = True
    *_, launched = step(prod, mono, 16)
    report(
        "a prefill step is production's",
        caches_equal=same(prod, mono),
        no_launch=launched == 0,
        rows_written=written(mono),
    )
    FC.context.is_prefill = False

    *_, launched = step(prod, mono, 65)
    report(
        "65 rows are production's",
        caches_equal=same(prod, mono),
        no_launch=launched == 0,
        rows_written=written(mono),
    )

    split = []
    for m in (prod, mono):
        m.write_split_context_kv = lambda taps, positions, n, slots, m=m: split.append(
            n
        )
    *_, launched = step(prod, mono, 16, pcp_num_tokens=16)
    report(
        "a PCP split takes the base path",
        both_split=split == [16, 16],
        no_launch=launched == 0,
    )

    taps, positions, slots = inputs(24)
    clear(prod, mono)
    prod.write_context_kv(taps, positions, slots)
    before = len(LAUNCHES)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        mono.write_context_kv(taps, positions, slots)
    clear(mono)
    graph.replay()
    KEEP.append((graph, taps, positions, slots))
    report(
        "a capture records production's kernels",
        caches_equal=same(prod, mono),
        no_launch=len(LAUNCHES) == before,
    )

    real_split = draft_dispatch.qkv_a_split_k

    def no_build(rows):
        raise MonoUnsupported(f"{rows} rows: refused for the check")

    draft_dispatch.qkv_a_split_k = no_build
    *_, launched = step(prod, mono, 12)
    draft_dispatch.qkv_a_split_k = real_split
    report(
        "a width whose build is refused is production's", caches_equal=same(prod, mono), no_launch=launched == 0,
        remembered=mono.mono_context.kernels.get(12, "unset") is None, logged=any("12 context rows" in r for r in mono.mono_context.refused),
    )  # fmt: skip
    *_, launched = step(prod, mono, 12)
    report("... and stays so", caches_equal=same(prod, mono), no_launch=launched == 0)

    import atom.model_ops.linear as linear_ops

    real_quant = linear_ops.per_1x128_e8m0_quant
    calls = []

    def short_scale(x, quant_dtype, transpose_scale, scale=None):
        x8, xs = real_quant(
            x, quant_dtype=quant_dtype, transpose_scale=transpose_scale, scale=scale
        )
        calls.append(1)
        return (x8, xs.reshape(-1)[:-1]) if len(calls) == 1 else (x8, xs)

    # (production's own linears call the same quant: only the context
    # runner's call, the first after the patch, gets the short scale)
    taps, positions, slots = inputs(16)
    clear(prod, mono)
    prod.write_context_kv(taps, positions, slots)
    before = len(LAUNCHES)
    linear_ops.per_1x128_e8m0_quant = short_scale
    try:
        mono.write_context_kv(taps, positions, slots)
    finally:
        linear_ops.per_1x128_e8m0_quant = real_quant
    report(
        "a quantized context the kernel cannot read: production finishes the step", caches_equal=same(prod, mono),
        tried_and_refused=LAUNCHES[before:] == [(16, False)], production_quants=len(calls) == 1 + LAYERS,
        logged=any("quantized context" in r for r in mono.mono_context.refused),
    )  # fmt: skip

    odd = Model(shared, True)
    odd.project_context = lambda taps: (
        KimiK3DSpark.project_context(odd, taps)
        .float()
        .to(torch.bfloat16)[:, : HIDDEN - 128]
    )
    taps, positions, slots = inputs(16)
    before = len(LAUNCHES)
    try:
        odd.write_context_kv(taps, positions, slots)
        outcome = "ran"
    except Exception as err:  # noqa: BLE001 -- production's own answer
        outcome = f"production raised {type(err).__name__}"
    torch.cuda.synchronize()
    report(
        "a context of another shape is handed to production",
        no_launch=len(LAUNCHES) == before,
        outcome=outcome,
    )

    biased = shared_weights(bias_on=2)
    KEEP.append(biased)
    prod_b, mono_b = Model(biased, False), Model(biased, True)
    *_, launched = step(prod_b, mono_b, 16)
    report(
        "a model the kernels cannot bind keeps production", caches_equal=same(prod_b, mono_b), no_launch=launched == 0,
        switched_off=mono_b._try_mono_context is False and mono_b.mono_context is None, rows_written=written(mono_b),
    )  # fmt: skip
    *_, launched = step(prod_b, mono_b, 16)
    report(
        "... and never asks again",
        caches_equal=same(prod_b, mono_b),
        no_launch=launched == 0,
    )

    sys.settrace(None)
    path = draft_context.__file__
    with open(path) as f:
        text = f.read()
    code = compile(text, path, "exec")
    lines, stack = set(), [code]
    while stack:
        c = stack.pop()
        lines.update(line for _, _, line in c.co_lines() if line)
        stack.extend(k for k in c.co_consts if hasattr(k, "co_lines"))
    source = text.splitlines()
    missed = sorted(
        n
        for n in lines - RAN
        if source[n - 1].strip() and not source[n - 1].strip().startswith(('"""', "#"))
    )
    print(
        json.dumps(
            {
                "draft_context_lines": len(lines),
                "ran": len(lines & RAN),
                "not_run": [f"{n}: {source[n - 1].strip()[:70]}" for n in missed],
            }
        ),
        flush=True,
    )
    print(json.dumps({"all_ok": all(RESULTS), "cases": len(RESULTS)}), flush=True)


try:
    main()
finally:
    sys.settrace(None)
