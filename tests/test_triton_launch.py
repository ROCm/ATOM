# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`CachedLauncher`: its key against Triton's own, the hit / miss / bypass
paths, and the Qwen3.8-Flash-Next HC launches against plain `kernel[grid]`."""

import itertools
from types import SimpleNamespace

import pytest
import torch

triton = pytest.importorskip("triton")

import triton.language as tl
from triton import knobs
from triton.backends.amd.compiler import HIPBackend
from triton.backends.compiler import GPUTarget
from triton.compiler import CompiledKernel
from triton.runtime.jit import (
    compute_cache_key,
    create_function_from_signature,
)

from atom.utils import triton_launch
from atom.utils.triton_launch import CachedLauncher


@triton.jit
def _probe_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    rows,
    stride,
    scale,
    flag,
    BLOCK: tl.constexpr,
    MODE: tl.constexpr,
    ALPHA: tl.constexpr,
):
    pass


@triton.jit
def _mixed_probe_kernel(
    x_ptr,
    n: tl.int32,
    HEADS: tl.constexpr,
    y_ptr: tl.pointer_type(tl.float32),
    rows,
    WIDTH: tl.constexpr,
    scale: tl.float32,
    out_ptr,
    BLOCK: tl.constexpr,
):
    """Constexprs between runtime parameters, and annotated runtime ones."""


def _triton_key(launcher, args):
    """`JITFunction.run`'s cache key for `args`, from Triton's own binder."""
    kernel = launcher.kernel
    backend = HIPBackend(GPUTarget("hip", "gfx942", 64))
    binder = create_function_from_signature(kernel.signature, kernel.params, backend)
    options = dict(launcher.options)
    options["debug"] = options.get("debug", kernel.debug) or knobs.runtime.debug
    options["instrumentation_mode"] = knobs.compilation.instrumentation_mode
    options["fpsan_homomorphic_casts"] = knobs.compilation.fpsan_homomorphic_casts
    _, specialization, options = binder(*args, **options)
    return compute_cache_key({}, specialization, options)


_BIG = []


def _tensor(shape, dtype=torch.bfloat16, misaligned=False, big=False):
    """A CPU tensor whose data pointer is 16-byte aligned or not, on a storage
    of at most or just over 2 GiB (CPU pages are not touched)."""
    numel = 1
    for size in shape:
        numel *= size
    if big:
        if not _BIG:
            _BIG.append(torch.empty(2**31 + 64, dtype=torch.uint8))
        flat = _BIG[0].view(dtype)
    else:
        flat = torch.empty(numel + 16, dtype=dtype)
    start = 1 if misaligned else 0
    return flat[start : start + numel].view(shape)


def _probe_args():
    x = _tensor((64, 48))
    return [x, x, _tensor((64, 32)), 64, 48, 0.5, True, 32, 1, 2.0]


def _mixed_probe_args():
    x = _tensor((64, 48))
    return [x, 64, 8, _tensor((64, 48), torch.float32), 48, 32, 0.5, x, 64]


def _variants(launcher, base):
    """`base`, then each argument changed in every way Triton's
    specialization (or this key) can tell apart."""
    yield base
    for i, arg in enumerate(base):
        if launcher._constexpr[i]:
            options = [arg + 1, float(arg) + 0.5] if not isinstance(arg, bool) else []
            options += [not arg] if isinstance(arg, bool) else [True]
        elif isinstance(arg, torch.Tensor):
            shape, dtype = arg.shape, arg.dtype
            other = torch.float32 if dtype != torch.float32 else torch.float16
            options = [
                _tensor(shape, dtype, misaligned=True),
                _tensor(shape, dtype, big=True),
                _tensor(shape, dtype, misaligned=True, big=True),
                _tensor(shape, other),
                _tensor(shape, torch.float8_e4m3fnuz),
                torch.nn.Parameter(_tensor(shape, dtype)),
            ]
        elif isinstance(arg, bool):
            options = [not arg, 1, 1.0, None]
        elif isinstance(arg, int):
            options = [0, 1, 2, 15, 16, 17, 32, 324, 1056, -16, -3, True, 1.0]
        else:
            options = [0.25, 1.0, 1e-6, 0, 1, 16, True, None]
        for option in options:
            variant = list(base)
            variant[i] = option
            yield variant


def _assert_key_is_at_least_tritons(launcher, base):
    seen = {}
    distinct = set()
    for args in _variants(launcher, base):
        triton_key = _triton_key(launcher, args)
        distinct.add(triton_key)
        key = launcher._key(args)
        if key is not None:
            assert seen.setdefault(key, triton_key) == triton_key, args
    assert launcher._key(base) is not None
    return distinct


@pytest.mark.parametrize("buffer_ops", ["1", "0"])
def test_key_tells_apart_what_triton_tells_apart(monkeypatch, buffer_ops):
    """Two launches with the same `CachedLauncher` key must have the same
    Triton cache key, or a cached kernel would serve a launch it was not
    specialized for. Fails if a Triton upgrade changes what it specializes on."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", buffer_ops)
    launcher = CachedLauncher(_probe_kernel, num_warps=4)
    distinct = _assert_key_is_at_least_tritons(launcher, _probe_args())
    assert len(distinct) > 20


@pytest.mark.parametrize("buffer_ops", ["1", "0"])
def test_key_covers_interleaved_constexprs_and_annotations(monkeypatch, buffer_ops):
    """A constexpr ahead of a runtime parameter, and `tl.int32` / pointer /
    `tl.float32` annotations, which fix the type in Triton's key."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", buffer_ops)
    launcher = CachedLauncher(_mixed_probe_kernel, num_warps=4)
    assert launcher._enabled and launcher._split is not None
    distinct = _assert_key_is_at_least_tritons(launcher, _mixed_probe_args())
    assert len(distinct) > 15
    args = _mixed_probe_args()
    key = launcher._key(args)
    assert key[5] == (8, 32, 64)  # HEADS, WIDTH, BLOCK


@pytest.mark.parametrize(
    "knob,env",
    [
        ("runtime.debug", None),
        ("compilation.instrumentation_mode", None),
        (None, "TRITON_FPSAN_HOMOMORPHIC_CASTS"),
        (None, "AMDGCN_USE_BUFFER_OPS"),
    ],
)
def test_key_follows_the_knobs_in_tritons_key(monkeypatch, knob, env):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", "1")
    monkeypatch.delenv("TRITON_FPSAN_HOMOMORPHIC_CASTS", raising=False)
    launcher = CachedLauncher(_probe_kernel, num_warps=4)
    args = _probe_args()
    before = (launcher._key(args), _triton_key(launcher, args))
    if knob is not None:
        group, attr = knob.split(".")
        value = True if attr == "debug" else "fpsan"
        monkeypatch.setattr(getattr(knobs, group), attr, value)
    else:
        monkeypatch.setenv(env, "0" if env == "AMDGCN_USE_BUFFER_OPS" else "1")
    after = (launcher._key(args), _triton_key(launcher, args))
    assert before[1] != after[1]
    assert before[0] != after[0]


def test_key_falls_back_for_arguments_it_does_not_model(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    launcher = CachedLauncher(_probe_kernel, num_warps=4)
    for position, value in [
        (3, 2**31),
        (3, -(2**31) - 1),
        (4, (1, 2)),
        (0, [1]),
        (7, (32,)),
        (8, torch.float16),
    ]:
        args = _probe_args()
        args[position] = value
        assert launcher._key(args) is None


def test_device_is_part_of_the_key(monkeypatch):
    launcher = CachedLauncher(_probe_kernel, num_warps=4)
    keys = []
    for device in (0, 1):
        monkeypatch.setattr(torch.cuda, "current_device", lambda d=device: d)
        keys.append(launcher._key(_probe_args()))
    assert keys[0] != keys[1]


# --- hit / miss / bypass, with a stand-in kernel ---------------------------


class _Compiled(CompiledKernel):
    def __init__(self, tag, launches):
        self.module = None
        self.function = f"function-{tag}"
        self.packed_metadata = (tag,)
        self._run = lambda *args: launches.append(("cached", tag, args))


class _Kernel:
    """A `JITFunction` stand-in with `_probe_kernel`'s parameters."""

    def __init__(self, params=_probe_kernel.params):
        self.params = params
        self.do_not_specialize = []
        self.do_not_specialize_on_alignment = []
        self.pre_run_hooks = []
        self.used_global_vals = {}
        self.launches = []
        self.compiled = []
        self.result = "compile"

    def __getitem__(self, grid):
        def run(*args, **kwargs):
            self.launches.append(("triton", grid, args, kwargs))
            if self.result != "compile":
                return self.result
            compiled = _Compiled(len(self.compiled), self.launches)
            self.compiled.append(compiled)
            return compiled

        return run


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(
        triton_launch,
        "driver",
        SimpleNamespace(active=SimpleNamespace(get_current_stream=lambda d: 70 + d)),
    )
    kernel = _Kernel()
    return kernel, CachedLauncher(kernel, num_warps=4)


def _paths(kernel):
    return [launch[0] for launch in kernel.launches]


def test_first_launch_compiles_then_the_launcher_is_called_directly(fake):
    kernel, launcher = fake
    args = _probe_args()
    runtime, constexprs = args[:7], dict(zip(("BLOCK", "MODE", "ALPHA"), args[7:]))
    launcher((5, 2), *runtime, **constexprs)
    assert _paths(kernel) == ["triton"]
    assert kernel.launches[0][2] == tuple(runtime)
    assert kernel.launches[0][3] == {**constexprs, "num_warps": 4}
    launcher((5, 2), *runtime, **constexprs)
    launcher((5,), *args)
    assert _paths(kernel) == ["triton", "cached", "cached"]
    assert len(kernel.compiled) == 1
    expected_grids = [(5, 2, 1), (5, 1, 1)]
    for (_, tag, call), grid in zip(kernel.launches[1:], expected_grids):
        assert tag == 0
        assert call[:3] == grid
        assert call[3:9] == (73, "function-0", (0,), None, None, None)
        assert call[9:] == tuple(args)


@pytest.mark.usefixtures("fake")
def test_interleaved_constexprs_reach_the_launcher_in_signature_order():
    kernel = _Kernel(_mixed_probe_kernel.params)
    launcher = CachedLauncher(kernel, num_warps=4)
    args = _mixed_probe_args()
    launcher((3,), *args[:8], BLOCK=64)
    launcher((3,), *args)
    changed = list(args)
    changed[5] = 16  # WIDTH, a constexpr between runtime parameters
    launcher((3,), *changed)
    launcher((3,), *changed)
    assert _paths(kernel) == ["triton", "cached", "triton", "cached"]
    assert kernel.launches[0][3] == {"BLOCK": 64, "num_warps": 4}
    assert kernel.launches[1][2][9:] == tuple(args)
    assert kernel.launches[3][2][9:] == tuple(changed)
    assert len(launcher._kernels) == 2


def test_each_specialization_compiles_once(fake):
    kernel, launcher = fake
    base = _probe_args()

    def changed(position, value):
        args = list(base)
        args[position] = value
        return args

    for args in [
        base,  # rows 64: a multiple of 16
        changed(3, 80),  # also a multiple of 16: same specialization
        changed(3, 63),
        changed(3, 1),
        changed(0, _tensor((64, 48), misaligned=True)),
        changed(2, _tensor((64, 32), dtype=torch.float32)),
        changed(7, 64),  # constexpr
        base,
        changed(3, 1),
    ]:
        launcher((1,), *args)
    assert _paths(kernel) == [
        "triton",
        "cached",
        "triton",
        "triton",
        "triton",
        "triton",
        "triton",
        "cached",
        "cached",
    ]
    assert len(launcher._kernels) == 6


@pytest.mark.parametrize(
    "case",
    [
        "keywords_out_of_order",
        "missing_argument",
        "callable_grid",
        "list_grid",
        "launch_enter_hook",
        "launch_exit_hook",
        "pre_run_hook",
        "stages_hook",
        "compiling",
        "tuple_argument",
    ],
)
def test_bypass_goes_through_triton_every_time(fake, monkeypatch, case):
    kernel, launcher = fake
    grid = (1, 1)
    args = _probe_args()
    runtime, kwargs = args[:7], {"BLOCK": 32, "MODE": 1, "ALPHA": 2.0}
    if case == "keywords_out_of_order":
        kwargs = {"MODE": 1, "BLOCK": 32, "ALPHA": 2.0}
    elif case == "missing_argument":
        del kwargs["ALPHA"]
    elif case == "callable_grid":
        grid = lambda meta: (1, 1)
    elif case == "list_grid":
        grid = [1, 1]
    elif case in ("launch_enter_hook", "launch_exit_hook"):
        chain = getattr(knobs.runtime, case)
        monkeypatch.setattr(chain, "calls", [lambda metadata: None])
    elif case == "pre_run_hook":
        kernel.pre_run_hooks.append(lambda *a, **k: None)
    elif case == "stages_hook":
        monkeypatch.setattr(
            knobs.runtime, "add_stages_inspection_hook", lambda: ("k", "h")
        )
    elif case == "compiling":
        monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)
    elif case == "tuple_argument":
        runtime = runtime[:3] + [(1, 2)] + runtime[4:]
    for _ in range(3):
        launcher(grid, *runtime, **kwargs)
    assert _paths(kernel) == ["triton"] * 3
    assert not launcher._kernels


def test_a_plain_callable_hook_counts_as_hooked(fake, monkeypatch):
    kernel, launcher = fake
    monkeypatch.setattr(knobs.runtime, "launch_enter_hook", lambda metadata: None)
    for _ in range(2):
        launcher((1,), *_probe_args())
    assert _paths(kernel) == ["triton", "triton"]


@pytest.mark.parametrize("case", ["globals", "skipped_compile", "unspecialized"])
def test_kernels_that_cannot_be_cached_stay_on_triton(fake, case):
    kernel, _ = fake
    if case == "globals":
        kernel.used_global_vals = {("SCALE", 0): (2, {})}
    elif case == "skipped_compile":
        kernel.result = None
    else:
        kernel.do_not_specialize = ["rows"]
    launcher = CachedLauncher(kernel, num_warps=4)
    for _ in range(2):
        launcher((1,), *_probe_args())
    assert _paths(kernel) == ["triton", "triton"]
    assert not launcher._kernels


def test_interpreted_kernels_are_never_cached():
    launches = []

    class Interpreted:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: launches.append(grid)

    launcher = CachedLauncher(Interpreted(), num_warps=4)
    assert not launcher._enabled
    launcher((2,), 1, 2)
    launcher((2,), 1, 2)
    assert launches == [(2,), (2,)]


# --- the Qwen3.8-Flash-Next HC launches -------------------------------------


@pytest.fixture
def hc():
    pytest.importorskip("aiter.ops.enum", exc_type=ImportError)
    from atom.model_ops.qwen4_exp.ops import gated, hc_fused

    return SimpleNamespace(gated=gated, hc_fused=hc_fused)


def _hc_cpu_args(hc):
    """Representative arguments of the five HC launches, as their wrappers
    pass them, on CPU (`_key` never touches the data)."""
    m, hidden, streams, rank = 300, 2560, 4, 320
    h = _tensor((m, streams * hidden))
    d = _tensor((m, rank + streams))
    raw, x = d[:, rank:], d[:, :rank]
    block = _tensor((m, hidden))
    weight = torch.nn.Parameter(_tensor((streams * hidden,)))
    out = _tensor((m, hidden))
    fp8 = torch.float8_e4m3fnuz
    strides = [h.stride(0), block.stride(0), raw.stride(0)]
    return {
        hc.hc_fused._hc_combine_norm_rows: [h, block, raw, weight, h, h, m]
        + strides
        + [1e-6, hidden, streams, True, 2, 2048, 512],
        hc.hc_fused._hc_combine_norm: [h, block, raw, weight, h, h]
        + strides
        + [1e-6, hidden, streams, True, 4096],
        hc.hc_fused._hc_gated_mean_tiled: [h, h, out, m, h.stride(0), out.stride(0)]
        + [hidden, streams, 4, 512],
        hc.hc_fused._hc_gated_mean_quant: [h, h, out, _tensor((m, hidden), fp8)]
        + [_tensor((m, 1), torch.float32), m, h.stride(0)]
        + [hidden, streams, 240.0, 1, 4096],
        hc.gated._scaled_silu: [x, x, x, _tensor((m, rank)), x.stride(0), 0, 0]
        + [rank, rank, streams, True, False, 512],
    }


@pytest.mark.parametrize("buffer_ops", ["1", "0"])
def test_hc_keys_tell_apart_what_triton_tells_apart(monkeypatch, hc, buffer_ops):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setenv("AMDGCN_USE_BUFFER_OPS", buffer_ops)
    for launcher, args in _hc_cpu_args(hc).items():
        assert launcher._enabled
        assert len(args) == len(launcher._names)
        _assert_key_is_at_least_tritons(launcher, args)


gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def _hc_inputs(m, offset=0, seed=0, hidden=2560, streams=4, rank=320):
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape, scale=1.0):
        x = torch.randn(*shape, generator=gen, device="cuda", dtype=torch.bfloat16)
        return x * scale

    h = randn(m * streams * hidden + offset)[offset:].view(m, streams * hidden)
    d = randn(m, rank + streams)
    return SimpleNamespace(
        h=h,
        norm_w=randn(streams * hidden, scale=0.1),
        block=randn(m, hidden),
        raw=d[:, rank:],
        x=d[:, :rank],
        up=randn(m, streams * hidden),
        hidden=hidden,
        streams=streams,
        rank=rank,
    )


def _reference_combine_norm(hc, t, combine, eps=1e-6):
    """`hc_combine_norm` as it was: plain `kernel[grid]` launches."""
    kernels = hc.hc_fused
    m, hidden, streams = t.h.shape[0], t.hidden, t.streams
    normed = torch.empty_like(t.h)
    h_new = torch.empty_like(t.h) if combine else t.h
    block, raw = (t.block, t.raw) if combine else (t.h, t.h)
    strides = (
        t.h.stride(0),
        t.block.stride(0) if combine else 0,
        t.raw.stride(0) if combine else 0,
    )
    ba = 1 << (hidden.bit_length() - 1)
    if m >= kernels.COMBINE_NORM_ROWS_MIN_M:
        kernels._hc_combine_norm_rows_kernel[(triton.cdiv(m, 2), streams)](
            t.h, block, raw, t.norm_w, h_new, normed, m, *strides, eps,
            H=hidden, HC=streams, COMBINE=combine, RB=2, BA=ba, BB=hidden - ba,
            num_warps=4,
        )  # fmt: skip
    else:
        kernels._hc_combine_norm_kernel[(m, streams)](
            t.h, block, raw, t.norm_w, h_new, normed, *strides, eps,
            H=hidden, HC=streams, COMBINE=combine,
            BLOCK=triton.next_power_of_2(hidden), num_warps=4,
        )  # fmt: skip
    return h_new, normed


def _reference_tail(hc, t, normed, quant_dtype):
    """`scaled_silu` and `hc_gated_mean(_quant)` as they were."""
    m, hidden, streams, rank = t.h.shape[0], t.hidden, t.streams, t.rank
    gate = torch.empty((m, rank), dtype=t.x.dtype, device="cuda")
    block = min(1024, triton.next_power_of_2(rank))
    hc.gated._gated_pointwise_kernel[(m, triton.cdiv(rank, block))](
        t.x, t.x, t.x, gate, t.x.stride(0), 0, 0, rank, rank, streams, True, False,
        block, num_warps=4, enable_fp_fusion=False,
    )  # fmt: skip
    out = torch.empty((m, hidden), dtype=normed.dtype, device="cuda")
    if quant_dtype is None:
        hc.hc_fused._hc_gated_mean_tiled_kernel[(triton.cdiv(m, 4), hidden // 512)](
            normed, t.up, out, m, normed.stride(0), out.stride(0),
            H=hidden, HC=streams, RB=4, BH=512, num_warps=4,
        )  # fmt: skip
        return gate, out
    q = torch.empty((m, hidden), dtype=quant_dtype, device="cuda")
    scale = torch.empty((m, 1), dtype=torch.float32, device="cuda")
    hc.hc_fused._hc_gated_mean_quant_kernel[(m,)](
        normed, t.up, out, q, scale, m, normed.stride(0),
        H=hidden, HC=streams, FP8_MAX=torch.finfo(quant_dtype).max, RB=1,
        BH=triton.next_power_of_2(hidden), num_warps=8,
    )  # fmt: skip
    return gate, out, q, scale


def _bitwise_equal(a, b):
    return len(a) == len(b) and all(
        torch.equal(x.view(torch.uint8), y.view(torch.uint8)) for x, y in zip(a, b)
    )


def _hc_launches(hc, t, combine, quant_dtype):
    block, raw = (t.block, t.raw) if combine else (None, None)
    h, normed = hc.hc_fused.hc_combine_norm(t.h, t.norm_w, 1e-6, t.streams, block, raw)
    gate = hc.gated.scaled_silu(t.x, t.streams)
    if quant_dtype is None:
        tail = (gate, hc.hc_fused.hc_gated_mean(normed, t.up, t.streams))
    else:
        tail = (
            gate,
            *hc.hc_fused.hc_gated_mean_quant(normed, t.up, t.streams, quant_dtype),
        )
    return (h, normed), tail


@gpu
@pytest.mark.parametrize("offset", [0, 1], ids=["aligned", "misaligned"])
@pytest.mark.parametrize("combine", [True, False])
@pytest.mark.parametrize("m", [2112, 1056, 1050, 256, 16, 3])
@torch.inference_mode()
def test_hc_launches_match_plain_triton_bitwise(hc, m, combine, offset):
    """Prefill (whole and SP-shard), mid and small batches, combining or not,
    streams 16-byte aligned or not; twice, so the second is a cache hit."""
    t = _hc_inputs(m, offset=offset, seed=m + offset)
    quant = None if m > 1024 else torch.float8_e4m3fnuz
    ref_head = _reference_combine_norm(hc, t, combine)
    ref_tail = _reference_tail(hc, t, ref_head[1], quant)
    for _ in range(2):
        head, tail = _hc_launches(hc, t, combine, quant)
        torch.cuda.synchronize()
        assert _bitwise_equal(head, ref_head)
        assert _bitwise_equal(tail, ref_tail)


@gpu
@torch.inference_mode()
def test_hc_launches_reuse_tritons_kernels(hc):
    """Each cached kernel is one Triton itself compiled for that launch."""
    for m, offset, combine in itertools.product((1056, 1050), (0, 1), (True, False)):
        t = _hc_inputs(m, offset=offset, seed=7)
        assert _bitwise_equal(
            _hc_launches(hc, t, combine, None)[0],
            _reference_combine_norm(hc, t, combine),
        )
    launcher = hc.hc_fused._hc_combine_norm_rows
    device = torch.cuda.current_device()
    compiled = {id(k) for k in launcher.kernel.device_caches[device][0].values()}
    cached = {
        id(entry[3]) for key, entry in launcher._kernels.items() if key[0] == device
    }
    assert cached and cached <= compiled
    # rows % 16, h alignment and COMBINE (which also turns the strides to 0).
    assert len(cached) == 8


@gpu
@torch.inference_mode()
def test_hc_launches_replay_in_a_cuda_graph(hc):
    t = _hc_inputs(1056, seed=11)
    ref = _reference_combine_norm(hc, t, True)
    _hc_launches(hc, t, True, None)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        head, _ = _hc_launches(hc, t, True, None)
    graph.replay()
    torch.cuda.synchronize()
    assert _bitwise_equal(head, ref)
