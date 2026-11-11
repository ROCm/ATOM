# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Timing harness of the layer kernel's overlap probes (``atom.mono.plan.probe``).

Armed by ``ATOM_MONO_OVERLAP_PROBE=<spec.json>``; for measurement only, never on in
production. ``MonoDecode.forward`` calls ``maybe_probe`` after a real decode step; on
the first steady step of each target width n, every TP rank, in lockstep:

1. re-runs the step eagerly layer by layer, snapshotting each layer's inputs (ar, res)
   and the router's (routing key, sigmoid) pairs (the ``pin`` table);
2. builds and compiles every variant's kernel for n (``variants``: name -> probe string);
3. captures one CUDA graph per variant: begin_step then the sparse layers back to back,
   each on its own snapshotted inputs (a variant's garbage never reaches another layer);
4. replays the graphs interleaved (``rounds`` x ``reps`` replays, shuffled variant order
   per round, same on every rank), events around each batch of replays;
5. writes ``<out>/rank<r>_S<n>.json``: per variant the per-round ms a replay, the
   layers' distinct routed experts, the weight / cache byte sizes.
Per-layer us of a variant = (graph - begin_step graph) / layers.
"""

import hashlib
import json
import os
import random
import threading
import time

import torch
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

from atom.distributed.group_agreement import group_agree  # noqa: F401
from atom.models.minimax_m3.mono.config import MAX_TOKENS, N_ROUTED, TOP_K
from atom.models.minimax_m3.mono.kernels.post_attn import K4_ABI
from atom.models.minimax_m3.mono.layout import SCRATCH
from atom.models.minimax_m3.mono.runner import _ptr, _sparse_metadata, token_rows
from atom.mono.plan.probe import parse
from atom.mono.runtime.compile import compile_kernels
from atom.utils.forward_context import get_forward_context

_STATE = {"last_n": None, "run": 0, "done": set(), "spec": None}
MAX_LAYER_ID = 64


def _spec():
    if _STATE["spec"] is None:
        path = os.environ.get("ATOM_MONO_OVERLAP_PROBE", "")
        _STATE["spec"] = json.load(open(path)) if path else {}
    return _STATE["spec"]


class _Guard:
    """A watchdog: a variant that deadlocks kills this rank (and names the variant)
    instead of hanging the server for ever."""

    def __init__(self, what, secs=180):
        self.what, self.secs = what, secs

    def __enter__(self):
        def die():
            print(f"[overlap_probe] WATCHDOG: {self.what} did not finish", flush=True)
            os._exit(3)

        self.t = threading.Timer(self.secs, die)
        self.t.daemon = True
        self.t.start()

    def __exit__(self, *a):
        self.t.cancel()


def maybe_probe(runner, input_ids, positions) -> None:
    spec = _spec()
    if not spec:
        return
    n = input_ids.numel()
    st = _STATE
    st["run"] = st["run"] + 1 if n == st["last_n"] else 1
    st["last_n"] = n
    if n not in spec["targets"] or n in st["done"] or st["run"] < spec.get("steady", 3):
        return
    st["done"].add(n)
    run_probe(runner, input_ids, positions, n, spec)


def _capture(fn, name):
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with _Guard(f"capture {name}"), torch.cuda.graph(g):
        fn()
    return g


def run_probe(runner, input_ids, positions, n, spec) -> None:
    dev = runner.dev
    rank = runner.rank
    out = spec["out"]
    os.makedirs(out, exist_ok=True)
    log = lambda *a: print(f"[overlap_probe r{rank} S{n}]", *a, flush=True)
    fwd = get_forward_context()
    rows = token_rows(fwd, n)
    sparse_md = _sparse_metadata(fwd)
    block_table, seq_lens = rows
    L = min(spec.get("layers", len(runner.sparse)), len(runner.sparse))
    sparse = runner.sparse[:L]
    torch.cuda.current_stream()

    # ---- 1. the real step again, layer by layer: inputs and routing of every layer
    t0 = time.time()
    runner.mailboxes.begin_step()
    aux: list = []
    res = (
        runner.run_dense_mono(input_ids, positions, fwd, (), aux)
        if runner.dense_mono
        else runner.run_dense(input_ids, positions, (), aux)
    )
    ar_in, res_in = [], []
    pin = torch.zeros(
        MAX_LAYER_ID, MAX_TOKENS, N_ROUTED, 4, dtype=torch.int32, device=dev
    )
    sc_off = SCRATCH["scores"]
    for i, lw in enumerate(sparse):
        ar_in.append(runner.ars[i % 2][:n].clone())
        res_in.append(res.clone())
        res = runner.run_sparse_layer(i, lw, fwd, positions, res, rows)
        torch.cuda.synchronize()
        # copy 0 of the router's pairs: token k, expert e at words 4 (k N_ROUTED + e): (key, tag, sigmoid, tag)
        words = (
            runner.scratch4[sc_off : sc_off + MAX_TOKENS * N_ROUTED * 16]
            .view(torch.int32)
            .view(MAX_TOKENS, N_ROUTED, 4)
        )
        pin[lw.layer_id] = words
    pin_cpu = pin.cpu()
    experts = []  # per layer: the distinct routed experts of the step's tokens
    for lw in sparse:
        keys = pin_cpu[lw.layer_id, :n, :, 0].to(torch.int64)  # [n, 128]
        top = keys.topk(TOP_K, dim=1).indices  # the picks: the largest keys
        experts.append(sorted(set(top.flatten().tolist())))
    log(
        f"reference pass {time.time() - t0:.1f}s, distinct experts/layer mean "
        f"{sum(map(len, experts)) / len(experts):.1f}"
    )

    # ---- 2. per-layer buffers and argument sets, built once
    out_h = torch.zeros(MAX_TOKENS, 6144, dtype=torch.bfloat16, device=dev)
    out_ar = torch.zeros(MAX_TOKENS, 6144, dtype=torch.bfloat16, device=dev)
    from atom.model_ops.minimax_m3.sparse_attn import to_page16_shuffle
    from atom.models.minimax_m3.mono.kernels.pre_attn import K1_ARGS

    keep = []  # the tensors the captured graphs point at
    layer_args = []
    for i, lw in enumerate(sparse):
        kv = fwd.kv_cache_data[f"layer_{lw.layer_id}"]
        k16, v16, ks16, vs16 = to_page16_shuffle(
            kv.k_cache, kv.v_cache, kv.k_scale, kv.v_scale
        )
        ptrs = {
            "ar": _ptr(ar_in[i]), "g_in": _ptr(lw.g_in), "w_qkv": _ptr(lw.w_qkv),
            "s_qkv": _ptr(lw.s_qkv), "g_q": _ptr(lw.g_q), "g_k": _ptr(lw.g_k),
            "g_iq": _ptr(lw.g_iq), "g_ik": _ptr(lw.g_ik), "cos_sin": _ptr(lw.cos_sin),
            "index_cache": _ptr(lw.attn_impl.index_cache), "iq_out": _ptr(runner.iq),
            "scratch": _ptr(runner.scratch1),
        }  # fmt: skip
        k1_args = torch.tensor(
            [ptrs[a] for a in K1_ARGS], dtype=torch.int64, device=dev
        )
        keep += [k16, v16, ks16, vs16, k1_args]
        layer_args.append(
            {
                "h_in": _ptr(runner.h),
                "q": _ptr(runner.q),
                "block_table": _ptr(block_table),
                "seq_lens": _ptr(seq_lens),
                "k_cache": _ptr(k16),
                "v_cache": _ptr(v16),
                "k_scale": _ptr(ks16),
                "v_scale": _ptr(vs16),
                "w_o": _ptr(lw.w_o),
                "s_o": _ptr(lw.s_o),
                "g_post": _ptr(lw.g_post),
                "w_gate": _ptr(lw.gate),
                "bias": _ptr(lw.bias),
                "w13": _ptr(lw.w13),
                "s13": _ptr(lw.s13),
                "w2": _ptr(lw.w2),
                "s2": _ptr(lw.s2),
                "h_mid": _ptr(out_h),
                "ar_out": _ptr(out_ar),
                "scratch": _ptr(runner.scratch4),
                **runner.peers.kernel_args(),
                "layer": lw.layer_id,
                "bt_width": block_table.shape[1],
                "q_len": sparse_md.decode.max_query_len,
                "tl": 0,
                "k1_args": _ptr(k1_args),
                "positions": _ptr(positions),
                "slot_mapping": _ptr(sparse_md.slot_mapping),
                "res": _ptr(res_in[i]),
                "batch_ids": _ptr(fwd.attn_metadata.batch_id_per_q_token),
                "w13s": _ptr(lw.w13s) if lw.shared_mxfp8 else 0,
                "s13s": _ptr(lw.s13s) if lw.shared_mxfp8 else 0,
                "w2s": _ptr(lw.w2s) if lw.shared_mxfp8 else 0,
                "s2s": _ptr(lw.s2s) if lw.shared_mxfp8 else 0,
            }
        )

    # ---- 3. variants: build, compile (a rank's share first, then all from the cache)
    variants = spec["variants"]
    kinds = sorted({lw.index_topk for lw in sparse})
    launchers = {}
    for name, probe in variants.items():
        if probe == "@begin":
            continue
        parse(probe)  # a malformed probe fails here, on every rank
        launchers[name] = {
            k: runner._build_k4(n, index_topk=k, **({"probe": probe} if probe else {}))
            for k in kinds
        }
    items = [
        (ln, abi) for d in launchers.values() for ln in d.values() for abi in (K4_ABI,)
    ]
    t0 = time.time()
    group = runner.k4._group
    import torch.distributed as dist

    world = dist.get_world_size(group)
    with _Guard("compile share", 3600):
        compile_kernels(items[rank::world])
    dist.barrier(group)
    with _Guard("compile all", 3600):
        compile_kernels(items)
    dist.barrier(group)
    log(
        f"compiled {len(launchers)} variants x {len(kinds)} kinds in {time.time() - t0:.0f}s"
    )

    # ---- 4. graphs
    graphs = {}
    for name, probe in variants.items():
        tl = _ptr(pin) if probe != "@begin" and parse(probe).pin else 0

        def body(name=name, tl=tl):
            runner.mailboxes.begin_step()
            if name in launchers:
                for i, lw in enumerate(sparse):
                    a = dict(layer_args[i])
                    a["tl"] = tl
                    _run_compiled(
                        launchers[name][lw.index_topk],
                        *K4_ABI.pack(a),
                        torch.cuda.current_stream(),
                    )

        graphs[name] = _capture(body, name)
    for name, g in graphs.items():  # one replay each, a hang names its variant
        with _Guard(f"first replay {name}"):
            g.replay()
            torch.cuda.synchronize()
    log("graphs captured and replayed once")

    # ---- 5. interleaved timing
    rounds, reps, warm = (
        spec.get("rounds", 7),
        spec.get("reps", 20),
        spec.get("warm", 4),
    )
    times = {v: [] for v in variants}
    stamps = []
    names = list(variants)
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for rnd in range(rounds):
        order = names[:]
        random.Random(1000 * n + rnd).shuffle(order)
        stamps.append(time.time())
        for name in order:
            g = graphs[name]
            with _Guard(f"time {name}"):
                for _ in range(warm):
                    g.replay()
                e0.record()
                for _ in range(reps):
                    g.replay()
                e1.record()
                e1.synchronize()
            times[name].append(e0.elapsed_time(e1) / reps)
    sizes = {
        "layer_bytes": [
            {
                "attn_w": sum(
                    t.nbytes
                    for t in (lw.w_qkv, lw.s_qkv, lw.w_o, lw.s_o, lw.gate, lw.bias)
                ),
                "expert": (lw.w13.nbytes + lw.s13.nbytes + lw.w2.nbytes + lw.s2.nbytes)
                // lw.w13.shape[0],
                "shared": (
                    sum(t.nbytes for t in (lw.w13s, lw.s13s, lw.w2s, lw.s2s))
                    if lw.shared_mxfp8
                    else 0
                ),
            }
            for lw in sparse[:1]
        ],
        "seq_lens": seq_lens[:n].tolist(),
    }
    res_json = {
        "n": n, "rank": rank, "layers": L, "rounds": rounds, "reps": reps,
        "variants": variants, "times_ms": times, "round_stamps": stamps,
        "experts": experts, "sizes": sizes,
        "probe_hash": {v: hashlib.sha256(p.encode()).hexdigest()[:8] for v, p in variants.items()},
    }  # fmt: skip
    json.dump(res_json, open(f"{out}/rank{rank}_S{n}.json", "w"))
    # leave the caches as the real step wrote them: the production variant last
    graphs[spec.get("restore", "base")].replay()
    torch.cuda.synchronize()
    log("done")
