# SPDX-License-Identifier: MIT
"""PRE-COMMFUSED-RS (v13b: replica Parameter-attribute fix + replica self-test): comm-fused Stage2+reduce-scatter for large row-sharded prefill FFN.

Separate module (bootstrap-installed) so qualified ``prefill_ffn``/``runtime`` stay
byte-identical. ``install()`` composes over ``prefill_ffn.prequantized_owned_ffn``:
rows in (MIN_ROWS, 32768] take the comm-fused path, every other call is delegated.

Fused path (bucket 32768 only, TP4):
  * per-rank 8192-row padded gather of fp8 value / e8m0 scale / logits-or-routes, so
    rank r's rows sit at [r*8192, r*8192+cap) -- the runner's fixed RS layout;
  * routed experts run through the unmodified ``forward_impl`` (router, validation,
    quant method); only the final ``atom.model_ops.moe.fused_moe`` call is intercepted
    (thread-local, active only inside this call) and replaced by
    ``CommFusedMoeRuntime.run`` with pad rows masked (ids -1, weight 0);
  * shared expert: per-layer replicas of every rank's TP shard (gathered once, lazy),
    evaluated on the local rows only and summed fp32 -> full shared for local rows,
    added by the fused kernel after RS (joined at first use).
Default off: nothing installed unless ATOM_DSV41_PREFILL_FFN_COMMFUSED=1.
"""

import copy  # noqa: I001
import inspect
import json
import os
import threading
import time

import torch
import torch.distributed as dist
from aiter.dist.parallel_state import get_tp_group

from atom.model_ops.blockscale import quantize_fp8

import atom.models.deepseek_v41.prefill_ffn as pf
from atom.models.deepseek_v41.prefill_ffn import _gather
from atom.models.deepseek_v41.prefill_router import owned_router_logits

_ENABLED = os.environ.get("ATOM_DSV41_PREFILL_FFN_COMMFUSED", "0") == "1"
_MIN_ROWS = int(os.environ.get("ATOM_DSV41_PREFILL_FFN_COMMFUSED_MIN_ROWS", "16384"))
_HIT_DIR = os.environ.get("ATOM_DSV41_PREFILL_FFN_COMMFUSED_HIT_DIR")
_CHECK_DIR = os.environ.get("ATOM_DSV41_PREFILL_FFN_COMMFUSED_CHECK_DIR")
_CHECK_LIMIT = int(os.environ.get("ATOM_DSV41_PREFILL_FFN_COMMFUSED_CHECK_LIMIT", "64"))
_TUNER_DIR = os.environ.get("ATOM_DSV41_PREFILL_FFN_COMMFUSED_TUNER_DIR")
_BUCKET = 32768
_WORLD = 4
_PER_RANK = _BUCKET // _WORLD
_HIDDEN = 5120

_DELEGATE = {"fn": None}
_HITS = {
    "fused": 0,
    "checked": 0,
    "delegated": 0,
    "unsupported": 0,
    "rows": 0,
    "calls": 0,
    "errors": 0,
    "replica_layers": 0,
    "replica_attr_copied": 0,
    "replica_selftest_fail": 0,
}
_STATE = {
    "runtime": None,
    "alt": None,
    "orig_fused_moe": None,
    "disabled": None,
    "abi_dumped": False,
    "check_n": 0,
}
_REPLICAS = {}
_TLS = threading.local()
_IMPL_PARAMS = None
_ARG_NAMES = ("hidden_states", "w1", "w2", "topk_weight", "topk_ids")
# fused_moe kwargs with no _fused_moe_impl equivalent; allowed only at their no-op value.
_NOOP_ONLY = {
    "splitk": (0, None),
    "clamp_shared": (True,),
    "shared_w1": (None,),
    "shared_w2": (None,),
    "shared_w1_scale": (None,),
    "shared_w2_scale": (None,),
    "shared_expert_id": (-1,),
    "stage2_scatter": (None,),
    "output": (None,),
    "num_local_tokens": (None,),
}


def _rank():
    return dist.get_rank() if dist.is_initialized() else -1


def _dump_hits():
    os.makedirs(_HIT_DIR, exist_ok=True)
    with open(
        os.path.join(_HIT_DIR, f"commfused_hits_rank{_rank()}_pid{os.getpid()}.json"),
        "w",
    ) as f:
        json.dump(dict(_HITS, disabled=_STATE["disabled"]), f)


def _log(name, rec):
    d = _CHECK_DIR or _HIT_DIR
    if not d:
        return
    os.makedirs(d, exist_ok=True)
    with open(
        os.path.join(d, f"{name}_rank{_rank()}_pid{os.getpid()}.jsonl"), "a"
    ) as f:
        f.write(json.dumps(rec) + "\n")


def install():
    if not _ENABLED:
        return {"installed": False, "reason": "default_off"}
    if _DELEGATE["fn"] is not None:
        raise RuntimeError("Install exactly once per process")
    import atom.model_ops.moe as moe_mod

    delegate = pf.prequantized_owned_ffn
    _DELEGATE["fn"] = delegate
    orig = moe_mod.fused_moe
    _STATE["orig_fused_moe"] = orig

    def fused_moe_intercept(*args, **kwargs):
        ctx = getattr(_TLS, "ctx", None)
        if ctx is None:
            return orig(*args, **kwargs)
        _TLS.ctx = None  # exactly one interception per activation
        return _run_commfused(ctx, args, kwargs)

    moe_mod.fused_moe = fused_moe_intercept

    def dispatch(moe, local, rows):
        if _MIN_ROWS < rows <= _BUCKET and _STATE["disabled"] is None:
            return commfused_owned_ffn(moe, local, rows)
        _HITS["delegated"] += 1
        return delegate(moe, local, rows)

    pf.prequantized_owned_ffn = dispatch
    if _HIT_DIR:
        import atexit

        atexit.register(_dump_hits)
    return {
        "installed": True,
        "min_rows": _MIN_ROWS,
        "bucket": _BUCKET,
        "check": bool(_CHECK_DIR),
        "delegate": f"{delegate.__module__}.{delegate.__qualname__}",
        "fused_moe": f"{orig.__module__}.{orig.__qualname__}",
    }


# ---------------------------------------------------------------- runtime / replicas
def _runtime():
    if _STATE["runtime"] is not None:
        return _STATE["runtime"]
    import sys

    tuner = _TUNER_DIR
    if tuner is None:
        import aiter

        tuner = os.path.join(
            os.path.dirname(os.path.dirname(aiter.__file__)), "op_tests", "tuners"
        )
    if tuner not in sys.path:
        sys.path.insert(0, tuner)
    from tune_comm_fused_moe import candidate_configs  # noqa: I001
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime
    from aiter.ops.comm_fused_moe_runtime import CommFusedMoeRuntime
    from aiter.ops.flydsl.comm_fused_moe_host import ShapeKey, _LazyRunners, config_name

    shape = ShapeKey(
        get_gfx_runtime(),
        _HIDDEN,
        640,
        384,
        6,
        _WORLD,
        get_cu_num(),
        add_shared=True,
        comm="rs",
    )
    # Frozen from cwin_r1 (validation/pre_commfused_rs_20261103/cwin_bench.py, M=32768, sbm64).
    axes = (
        "gather_output=false",
        "tile_m=64",
        "tile_n=256",
        "tile_k=128",
        "window=1024",
        "local_workers=640",
        "reduce_scatter_grid=72",
        "all_gather_grid=99",
        "local_rows_per_cta=2",
        "drain_reduce_scatter_grid=140",
        "reduce_scatter_load_cache_modifier=1",
        "producer_spatial_partition=7202",
        "sort_block_m=64",
    )
    cfgs = candidate_configs(
        None, "window", axes, shape=shape.kernel_shape(), m=_BUCKET
    )
    if len(cfgs) != 1:
        raise RuntimeError(f"expected one frozen config, got {len(cfgs)}")
    _STATE["config"] = config_name(cfgs[0])
    _STATE["runtime"] = CommFusedMoeRuntime(
        runners=_LazyRunners(get_tp_group(), shape, {_BUCKET: cfgs[0]})
    )
    _STATE["alt"] = torch.cuda.Stream(device=torch.cuda.current_device())
    return _STATE["runtime"]


def _tensor_slots(module):
    """(kind, name, tensor) for every CUDA tensor directly owned by ``module``."""
    out = []
    for n, p in module._parameters.items():
        if p is not None:
            out.append(("param", n, p))
    for n, b in module._buffers.items():
        if b is not None:
            out.append(("buffer", n, b))
    for n, v in module.__dict__.items():
        if n in ("_parameters", "_buffers", "_modules"):
            continue
        if isinstance(v, torch.Tensor) and not isinstance(v, torch.nn.Parameter):
            out.append(("attr", n, v))
    return [s for s in out if s[2].is_cuda]


def _shared_replicas(moe, group):
    key = id(moe)
    reps = _REPLICAS.get(key)
    if reps is not None:
        return reps
    t0 = time.time()
    shared = moe.shared_experts
    world, rank = group.world_size, group.rank_in_group
    reps = [shared if j == rank else copy.deepcopy(shared) for j in range(world)]
    mods = [list(r.modules()) for r in reps]
    nbytes = 0
    done = set()
    for i, m in enumerate(mods[rank]):
        slots = _tensor_slots(m)
        rep_slots = [_tensor_slots(mods[j][i]) for j in range(world)]
        if any(len(rs) != len(slots) for rs in rep_slots):
            raise RuntimeError(
                f"replica slot mismatch in module {i} {type(m).__name__}"
            )
        for s, (kind, name, t) in enumerate(slots):
            if t.data_ptr() in done:
                continue
            done.add(t.data_ptr())
            if not t.is_contiguous():
                raise RuntimeError(
                    f"non-contiguous shared tensor {type(m).__name__}.{name}"
                )
            src = t.detach().reshape(-1).view(torch.uint8)
            full = src.new_empty((world, src.numel()))
            dist.all_gather_into_tensor(full, src, group=group.device_group)
            for j in range(world):
                if j == rank:
                    continue
                dst = rep_slots[j][s][2]
                if (
                    dst.shape != t.shape
                    or dst.dtype != t.dtype
                    or dst.data_ptr() == t.data_ptr()
                ):
                    raise RuntimeError(
                        f"replica tensor mismatch {type(m).__name__}.{name}"
                    )
                dst.detach().reshape(-1).view(torch.uint8).copy_(full[j])
            nbytes += src.numel() * (world - 1)
    attrs = _copy_tensor_attrs(mods, rank, world)
    torch.cuda.synchronize()
    selftest = _replica_selftest(shared, reps, group, key)
    _REPLICAS[key] = reps
    _HITS["replica_layers"] += 1
    free, total = torch.cuda.mem_get_info()
    _log(
        "commfused_replica",
        {
            "layer": (
                getattr(moe, "layer_id", getattr(moe, "prefix", None))
                if not isinstance(getattr(moe, "prefix", None), torch.Tensor)
                else None
            ),
            "bytes": nbytes,
            "tensors": len(done),
            "sec": time.time() - t0,
            "attrs_copied": attrs,
            "selftest": selftest,
            "mem_free": free,
            "mem_total": total,
        },
    )
    return reps


def _plain_attrs(t):
    return {
        k: v
        for k, v in getattr(t, "__dict__", {}).items()
        if not isinstance(v, torch.Tensor)
    }


def _copy_tensor_attrs(mods, rank, world):
    """Parameter.__deepcopy__ drops Parameter.__dict__ (is_shuffled, ...); restore it on every replica."""
    n = 0
    for i, m in enumerate(mods[rank]):
        slots = _tensor_slots(m)
        for j in range(world):
            if j == rank:
                continue
            for (kind, name, t), (_, _, d) in zip(slots, _tensor_slots(mods[j][i])):
                src_attrs = _plain_attrs(t)
                for k, v in src_attrs.items():
                    if getattr(d, k, _MISSING) is not v:
                        setattr(d, k, v)
                        n += 1
                if _plain_attrs(d).keys() != src_attrs.keys():
                    raise RuntimeError(
                        f"replica attr mismatch {type(m).__name__}.{name}: "
                        f"{sorted(src_attrs)} vs {sorted(_plain_attrs(d))}"
                    )
    _HITS["replica_attr_copied"] += n
    return n


_MISSING = object()
_PROBE_ROWS = 256


def _replica_selftest(shared, reps, group, key):
    """Replica j must reproduce rank j's own module on an identical probe (bitwise expected)."""
    world, rank = group.world_size, group.rank_in_group  # noqa: F841
    g = torch.Generator(device="cuda").manual_seed(1234 + len(_REPLICAS))
    probe = torch.randn(
        (_PROBE_ROWS, _HIDDEN), generator=g, device="cuda", dtype=torch.float32
    ).to(torch.bfloat16)
    pv, ps = quantize_fp8(probe)
    own = shared(pv, x_scale=ps).contiguous()
    full = own.new_empty((world,) + tuple(own.shape))
    dist.all_gather_into_tensor(full, own, group=group.device_group)
    res = []
    for j in range(world):
        o = reps[j](pv, x_scale=ps)
        c = _cmp(o, full[j])
        res.append({"j": j, "rel_l2": c["rel_l2"], "bitwise": c["bitwise_equal"]})
    bad = [r for r in res if not (r["bitwise"] or r["rel_l2"] <= 1e-3)]
    if bad:
        _HITS["replica_selftest_fail"] += 1
        raise RuntimeError(f"shared replica self-test failed: {bad}")
    return res


def _local_shared(reps, value, scale, valid, out_rows):
    acc = None
    for r in reps:
        o = r(value, x_scale=scale).float()
        acc = o if acc is None else acc.add_(o)
    sh = torch.zeros((out_rows, _HIDDEN), dtype=torch.bfloat16, device=value.device)
    sh[:valid].copy_(acc[:valid])
    return sh


# ---------------------------------------------------------------- fused path
def _supported(moe, local, rows, group):
    return (
        group.world_size == _WORLD
        and _MIN_ROWS < rows <= _BUCKET
        and local.shape == ((rows + 3) // 4, _HIDDEN)
        and local.is_contiguous()
        and getattr(moe, "shared_experts", None) is not None
        and not getattr(moe.experts, "reduce_results", False)
    )


def _pad_gather(t, cap, group):
    buf = t.new_zeros((_PER_RANK,) + tuple(t.shape[1:]))
    buf[:cap].copy_(t)
    return _gather(buf, _BUCKET, group)


def commfused_owned_ffn(moe, local, rows):
    group = get_tp_group()
    if not _supported(moe, local, rows, group):
        _HITS["unsupported"] += 1
        return _DELEGATE["fn"](moe, local, rows)
    _HITS["calls"] += 1
    _HITS["rows"] += rows
    if _HIT_DIR and _HITS["calls"] % 64 == 0:
        _dump_hits()
    if _CHECK_DIR and _STATE["check_n"] < _CHECK_LIMIT:
        _STATE["check_n"] += 1
        _HITS["checked"] += 1
        return _checked(moe, local, rows, group)
    _HITS["fused"] += 1
    return _fused(moe, local, rows, group)


def _fused(moe, local, rows, group, timing=None):
    runtime = _runtime()
    alt = _STATE["alt"]
    rank = group.rank_in_group
    cap = local.shape[0]
    valid = cap if rank < _WORLD - 1 else rows - (_WORLD - 1) * cap
    logits = owned_router_logits(moe, local, rows, group)
    selected = None
    if getattr(moe, "prefill_route_packet", False):
        selected = moe._topk(local, logits, 6, True)
    value, scale = quantize_fp8(local)
    reps = _shared_replicas(moe, group)
    main = torch.cuda.current_stream()
    alt.wait_stream(main)
    with torch.cuda.stream(alt):
        sh = _local_shared(reps, value, scale, valid, _PER_RANK)
    v_full = _pad_gather(value, cap, group)
    s_full = _pad_gather(scale, cap, group)
    if selected is None:
        l_full = _pad_gather(logits, cap, group)
        sel_full = None
    else:
        packet = torch.cat((selected[0].view(torch.int32), selected[1]), dim=1)
        p_full = _pad_gather(packet, cap, group)
        sel_full = (
            p_full[:, :6].contiguous().view(torch.float32),
            p_full[:, 6:].contiguous(),
        )
        l_full = None
    ctx = {
        "runtime": runtime,
        "sh": sh,
        "alt": alt,
        "main": main,
        "cap": cap,
        "last": rows - (_WORLD - 1) * cap,
        "hit": False,
    }
    _TLS.ctx = ctx
    try:
        routed = moe.experts.forward_impl(
            v_full, l_full, x_scale=s_full, preselected_routes=sel_full
        )
    finally:
        _TLS.ctx = None
    if not ctx["hit"]:
        raise RuntimeError("fused_moe interception did not fire")
    if routed.shape != (_PER_RANK, _HIDDEN):
        raise RuntimeError(f"unexpected comm-fused output {tuple(routed.shape)}")
    sh.record_stream(main)
    out = routed[:cap]
    if valid < cap:
        out[valid:].zero_()  # native combine_pad tail is zero
    return out


def _impl_params():
    global _IMPL_PARAMS
    if _IMPL_PARAMS is None:
        from aiter.fused_moe import _fused_moe_impl

        _IMPL_PARAMS = {
            n
            for n, p in inspect.signature(_fused_moe_impl).parameters.items()
            if not n.startswith("_")
        }
    return _IMPL_PARAMS


def _abi(args, kwargs):
    def d(v):
        if isinstance(v, torch.Tensor):
            return {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "contig": v.is_contiguous(),
            }
        return repr(v)[:120]

    return {
        "args": [d(a) for a in args],
        "kwargs": {k: d(v) for k, v in kwargs.items()},
    }


def _run_commfused(ctx, args, kwargs):
    ctx["hit"] = True
    if not _STATE["abi_dumped"]:
        _STATE["abi_dumped"] = True
        _log("commfused_abi", _abi(args, kwargs))
    if len(args) > len(_ARG_NAMES):
        raise RuntimeError(f"unexpected positional fused_moe args: {len(args)}")
    moe_args = dict(zip(_ARG_NAMES, args))
    for k, v in kwargs.items():
        if k in moe_args:
            raise RuntimeError(f"duplicate fused_moe arg {k}")
        moe_args[k] = v
    params = _impl_params()
    for k in list(moe_args):
        if k in _NOOP_ONLY:
            v = moe_args.pop(k)
            if not any(v is ok or v == ok for ok in _NOOP_ONLY[k]):
                raise RuntimeError(
                    f"fused_moe arg {k}={v!r} unsupported by comm-fused path"
                )
        elif k not in params:
            raise RuntimeError(f"fused_moe arg {k} unknown to _fused_moe_impl")
    if moe_args.get("block_size_M") is None:
        moe_args["block_size_M"] = -1
    hidden = moe_args["hidden_states"]
    if hidden.shape[0] != _BUCKET or moe_args.get("a1_scale") is None:
        raise RuntimeError("comm-fused path expects 32768 prequantized rows")
    ids = moe_args["topk_ids"].clone()
    wts = moe_args["topk_weight"].clone()
    pos = torch.arange(_BUCKET, device=ids.device)
    j, r = pos % _PER_RANK, pos // _PER_RANK
    limit = torch.where(
        r < _WORLD - 1, torch.full_like(j, ctx["cap"]), torch.full_like(j, ctx["last"])
    )
    pad = j >= limit
    ids.masked_fill_(pad[:, None], -1)
    wts.masked_fill_(pad[:, None], 0)
    moe_args["topk_ids"] = ids
    moe_args["topk_weight"] = wts
    sh, main, alt = ctx["sh"], ctx["main"], ctx["alt"]
    out = ctx["runtime"].run(
        shared_partial=None,
        before_stage2_for_rows=lambda n: sh,
        before_shared_add=lambda: main.wait_stream(alt),
        routes_may_be_missing=True,
        **moe_args,
    )
    main.wait_stream(alt)
    return out.clone()


# ---------------------------------------------------------------- check mode
def _cmp(a, b):
    d = a.float() - b.float()
    return {
        "max_abs": float(d.abs().max()) if d.numel() else 0.0,
        "rel_l2": float(d.norm() / b.float().norm().clamp_min(1e-12)),
        "bitwise_equal": bool(torch.equal(a, b)),
        "nan_a": int(torch.isnan(a).sum()),
        "nan_b": int(torch.isnan(b).sum()),
    }


def _checked(moe, local, rows, group):
    """Run native twice and fused once on the live call; log; return native."""
    rank = group.rank_in_group
    cap = local.shape[0]
    valid = cap if rank < _WORLD - 1 else rows - (_WORLD - 1) * cap
    rec = {
        "call": _STATE["check_n"],
        "rank": rank,
        "rows": rows,
        "cap": cap,
        "valid": valid,
        "route_packet": bool(getattr(moe, "prefill_route_packet", False)),
    }
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
    torch.cuda.synchronize()
    ev[0].record()
    native = _DELEGATE["fn"](moe, local, rows).clone()
    ev[1].record()
    native2 = _DELEGATE["fn"](moe, local, rows).clone()
    torch.cuda.synchronize()
    try:
        _shared_replicas(moe, group)  # first-hit collective outside the timed region
        torch.cuda.synchronize()
        ev[2].record()
        fused = _fused(moe, local, rows, group).clone()
        ev[3].record()
        torch.cuda.synchronize()
        rec["native_repeat"] = _cmp(native2[:valid], native[:valid])
        rec["fused_vs_native"] = _cmp(fused[:valid], native[:valid])
        rec["fused_tail_zero"] = bool((fused[valid:] == 0).all())
        rec["native_ms"] = ev[0].elapsed_time(ev[1])
        rec["fused_ms_incl_first_jit"] = ev[2].elapsed_time(ev[3])
        # shared-only: replica sum vs native shared partial reduced over TP, local rows.
        value, scale = quantize_fp8(local)
        reps = _shared_replicas(moe, group)
        mine = _local_shared(reps, value, scale, valid, cap)
        v_full = _gather(value, rows, group)
        s_full = _gather(scale, rows, group)
        part = moe.shared_experts(v_full, x_scale=s_full)
        padded = part.new_zeros((cap * _WORLD, _HIDDEN))
        padded[:rows].copy_(part)
        red = part.new_empty((cap, _HIDDEN))
        dist.reduce_scatter_tensor(red, padded, group=group.device_group)
        rec["shared_replica_vs_native"] = _cmp(mine[:valid], red[:valid])
        # routed-only (bf16 subtraction; diagnostic): fused minus replica shared vs native minus TP-reduced shared.
        rec["routed_vs_native"] = _cmp(
            (fused[:valid].float() - mine[:valid].float()),
            (native[:valid].float() - red[:valid].float()),
        )
    except Exception as e:  # noqa: BLE001 -- diagnostic: record and keep serving native
        _HITS["errors"] += 1
        rec["error"] = f"{type(e).__name__}: {e}"[:2000]
        _STATE["disabled"] = rec["error"][:200]
        torch.cuda.synchronize()
    rec["t"] = time.time()
    _log("commfused_check", rec)
    return native
