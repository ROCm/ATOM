# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`ATOM_K3_DRAFT_QKROPE`: the one-launch norm + rope + pool write vs the chain.

The kernel runs on CPU tensors under Triton's interpreter
(`triton_interpreter`), so it needs Triton but no GPU and no AITER. The
reference is a torch emulation of the launches it replaces, at their rounding
points: RMSNorm (fp32 math, bf16 out), neox rope from the bf16 cos/sin cache
(fp32 math, bf16 out), `reshape_and_cache_flash` (fp8: `float(x) * (1/scale)`,
saturating e4m3fn).

Pass criterion for q and k: at least 99.5% of elements exact and never more than
one bf16 ulp off (the fp32 reduction order of the sum of squares and FMA
contraction are the compiler's). The pool bytes must equal the cast of the
kernel's own k, and of v untouched, so the write itself is exact.

`qkrope_supported` is plain torch, so its tests run on any CPU runner: without
Triton, the module is loaded under a private name against a stand-in `triton`.
The attention-module tests need the draft model module, hence AITER; they skip
where it cannot import. Against the real aiter ops on a GPU:
`RUN_QKROPE_GPU_TESTS=1 python -m pytest tests/test_draft_qk_rope_cache.py`.
"""

import functools
import importlib
import importlib.util
import os
import pathlib
import sys
import types
from unittest.mock import MagicMock

import pytest
import torch
from triton_interpreter import in_the_interpreter, interpreter_test, interpreting

HQ, HKV, D = 8, 2, 64
EPS = 1e-6

test_under_the_interpreter = interpreter_test(__file__)


def _kernels():
    # Imported inside the child only: with the interpreter off, Triton would
    # build compiled kernels that nothing here can launch on CPU tensors.
    from atom.model_ops import draft_qk_rope_cache

    return draft_qk_rope_cache


@functools.cache
def _cast_modes():
    """Whether the backend's narrowing casts round to nearest even.

    The interpreter's may truncate where the GPU rounds (`v_cvt_pk_bf16_f32`,
    `v_cvt_pk_fp8_f32`); the reference emulates whichever mode it finds, so a
    CPU run checks the kernel's logic and a GPU run the real rounding.
    """
    import triton
    import triton.language as tl

    @triton.jit
    def probe(x_ptr, b_ptr, f_ptr, N: tl.constexpr):
        o = tl.arange(0, N)
        x = tl.load(x_ptr + o)
        tl.store(b_ptr + o, x.to(tl.bfloat16))
        tl.store(f_ptr + o, x.to(tl.float8e4nv))

    x = torch.linspace(-3.3, 3.7, 64)
    b = torch.empty(64, dtype=torch.bfloat16)
    f = torch.empty(64, dtype=torch.float8_e4m3fn)
    probe[(1,)](x, b, f, N=64)
    return torch.equal(b, x.to(torch.bfloat16))


def to_bf16(x):
    if _cast_modes():
        return x.to(torch.bfloat16)
    return (x.float().view(torch.int32) & -65536).view(torch.float32).to(torch.bfloat16)


@functools.cache
def _cast_fp8_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def cast(x_ptr, f_ptr, n, B: tl.constexpr):
        o = tl.program_id(0) * B + tl.arange(0, B)
        m = o < n
        v = tl.load(x_ptr + o, mask=m, other=0.0)
        tl.store(f_ptr + o, v.to(tl.float8e4nv), mask=m)

    return cast


def to_fp8(x):
    """fp32 -> e4m3fn with the backend's own cast, so the pool check isolates
    addressing and scaling from rounding."""
    import triton

    flat = x.float().contiguous().view(-1)
    out = torch.empty(flat.shape, dtype=torch.float8_e4m3fn)
    _cast_fp8_kernel()[(triton.cdiv(flat.numel(), 256),)](
        flat, out, flat.numel(), B=256
    )
    return out.view(x.shape)


def ref_norm(x, w):
    xf = x.float()
    r = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return to_bf16(xf * r * w.float())


def ref_rope(x, c, s):
    xf = x.float()
    lo, hi = xf[..., : D // 2], xf[..., D // 2 :]
    return to_bf16(torch.cat([lo * c - hi * s, hi * c + lo * s], -1))


def fp8(x, scale):
    return to_fp8((x.float() * (1.0 / scale)).clamp(-448, 448))


def ulp_bf16(a, b):
    def ordered(t):
        i = t.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(i < 0, -(i & 0x7FFF), i)

    return (ordered(a) - ordered(b)).abs()


def _rope_tables(gen, max_pos):
    freqs = torch.outer(
        torch.arange(max_pos).float(),
        1.0 / (10000 ** (torch.arange(0, D, 2).float() / D)),
    )
    cos = (freqs.cos() * 1.2).to(torch.bfloat16).view(max_pos, 1, 1, D // 2)
    sin = (freqs.sin() * 1.2).to(torch.bfloat16).view(max_pos, 1, 1, D // 2)
    return cos, sin


def _assert_close(out, ref):
    ulp = ulp_bf16(out, ref)
    assert int(ulp.max()) <= 1
    assert float((ulp == 0).float().mean()) >= 0.995


@in_the_interpreter
@pytest.mark.parametrize(
    "seed,n,fp8_pool,with_q",
    [
        (100, 20, True, True),
        (101, 24, True, False),
        (102, 7, False, True),
        (103, 24, False, False),
        (104, 1, True, True),
    ],
)
@pytest.mark.parametrize("scale", [1.0, 0.5])
def test_kernel_matches_the_unfused_chain(seed, n, fp8_pool, with_q, scale):
    block_size, nblocks = 16, 12
    gen = torch.Generator().manual_seed(seed)
    width = HQ * D + 2 * HKV * D
    big = torch.randn(n, width + 64, generator=gen).to(torch.bfloat16) * 2
    qkv = big[:, :width]  # strided rows, unit inner stride
    orig = qkv.clone()
    q_w = (torch.rand(D, generator=gen) + 0.5).to(torch.bfloat16)
    k_w = (torch.rand(D, generator=gen) + 0.5).to(torch.bfloat16)
    max_pos = 4096
    cos, sin = _rope_tables(gen, max_pos)
    pos = torch.randint(0, max_pos, (n,), generator=gen)
    slots = torch.randperm(nblocks * block_size, generator=gen)[:n].to(torch.int64)
    slots[torch.rand(n, generator=gen) < 0.2] = -1
    dtype = torch.float8_e4m3fn if fp8_pool else torch.bfloat16
    k_cache = torch.zeros(nblocks, block_size, HKV, D).to(dtype)
    v_cache = torch.zeros(nblocks, block_size, HKV, D).to(dtype)

    _kernels().draft_qk_norm_rope_cache(
        qkv,
        q_w if with_q else None,
        k_w,
        EPS,
        pos,
        cos,
        sin,
        slots,
        k_cache,
        v_cache,
        scale,
        scale,
        HQ,
        HKV,
        D,
        write_k=with_q,
    )

    c = cos.view(max_pos, D // 2)[pos].float().unsqueeze(1)
    s = sin.view(max_pos, D // 2)[pos].float().unsqueeze(1)
    q0 = orig[:, : HQ * D].view(n, HQ, D)
    k0 = orig[:, HQ * D : HQ * D + HKV * D].view(n, HKV, D)
    v0 = orig[:, HQ * D + HKV * D :].view(n, HKV, D)
    k_ref = ref_rope(ref_norm(k0, k_w), c, s)
    q_out = qkv[:, : HQ * D].reshape(n, HQ, D)
    k_out = qkv[:, HQ * D : HQ * D + HKV * D].reshape(n, HKV, D)
    assert torch.equal(qkv[:, HQ * D + HKV * D :], orig[:, HQ * D + HKV * D :])
    if with_q:
        _assert_close(q_out, ref_rope(ref_norm(q0, q_w), c, s))
        _assert_close(k_out, k_ref)
    else:
        assert torch.equal(q_out, q0), "q changed in context mode"
        assert torch.equal(k_out, k0), "k written back in context mode"

    flat_k, flat_v = k_cache.view(-1, HKV, D), v_cache.view(-1, HKV, D)
    written = torch.zeros(flat_k.shape[0], dtype=torch.bool)
    for t in range(n):
        slot = int(slots[t])
        if slot < 0:
            continue
        written[slot] = True
        # with q the kernel's own k is the oracle; without, the reference's k
        k_used = k_out[t] if with_q else k_ref[t]
        k_want = fp8(k_used, scale) if fp8_pool else k_used
        v_want = fp8(v0[t], scale) if fp8_pool else v0[t]
        got_k, got_v = flat_k[slot], flat_v[slot]
        if with_q:
            assert torch.equal(
                got_k.contiguous().view(torch.uint8),
                k_want.contiguous().view(torch.uint8),
            )
        else:
            # k comes from the reference here, which may sit one ulp away
            tolerance = float(k_want.float().abs().max()) * 2**-3 + 1e-6
            assert float((got_k.float() - k_want.float()).abs().max()) <= tolerance
        assert torch.equal(
            got_v.contiguous().view(torch.uint8), v_want.contiguous().view(torch.uint8)
        )
    assert (
        float(flat_k[~written].float().abs().sum()) == 0.0
    ), "written at a skipped slot"
    assert (
        float(flat_v[~written].float().abs().sum()) == 0.0
    ), "written at a skipped slot"


@in_the_interpreter
def test_empty_batch_is_a_no_op():
    kernels = _kernels()
    qkv = torch.empty(0, HQ * D + 2 * HKV * D, dtype=torch.bfloat16)
    cache = torch.zeros(2, 16, HKV, D, dtype=torch.bfloat16)
    cos = torch.zeros(8, 1, 1, D // 2, dtype=torch.bfloat16)
    kernels.draft_qk_norm_rope_cache(
        qkv, torch.ones(D), torch.ones(D), EPS, torch.empty(0, dtype=torch.int64),
        cos, cos, torch.empty(0, dtype=torch.int64), cache, cache.clone(),
        1.0, 1.0, HQ, HKV, D,
    )  # fmt: skip


def _rope(**overrides):
    return types.SimpleNamespace(
        **{
            "is_neox_style": True,
            "rotary_dim": D,
            "cos_cache": torch.zeros(8, 1, 1, D // 2),
            "sin_cache": torch.zeros(8, 1, 1, D // 2),
            **overrides,
        }
    )


@functools.cache
def _qkrope_supported():
    """The real predicate; without Triton, from the module loaded against a
    stand-in `triton` (it only defines the kernel at import, never runs it)."""
    if importlib.util.find_spec("triton") is not None:
        return _kernels().qkrope_supported
    path = (
        pathlib.Path(__file__).resolve().parents[1]
        / "atom"
        / "model_ops"
        / "draft_qk_rope_cache.py"
    )
    fake = MagicMock(name="triton")
    names = {"triton": fake, "triton.language": fake.language}
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules.update(names)
    try:
        spec = importlib.util.spec_from_file_location("_draft_qk_rope_no_triton", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        for name, old in saved.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old
    return module.qkrope_supported


def _supported(**overrides):
    args = {
        "qkv": torch.zeros(4, HQ * D + 2 * HKV * D, dtype=torch.bfloat16),
        "k_cache": torch.zeros(2, 16, HKV, D, dtype=torch.float8_e4m3fn),
        "v_cache": torch.zeros(2, 16, HKV, D, dtype=torch.float8_e4m3fn),
        "num_kv_heads": HKV,
        "head_dim": D,
        "eps_q": EPS,
        "eps_k": EPS,
        "rope": _rope(),
    }
    args.update(overrides)
    return _qkrope_supported()(**args)


def test_the_supported_case_is_supported():
    assert _supported()
    assert _supported(
        k_cache=torch.zeros(2, 16, HKV, D, dtype=torch.bfloat16),
        v_cache=torch.zeros(2, 16, HKV, D, dtype=torch.bfloat16),
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"qkv": torch.zeros(4, HQ * D + 2 * HKV * D, dtype=torch.float16)},
        {"qkv": torch.zeros(HQ * D + 2 * HKV * D, 4, dtype=torch.bfloat16).t()},
        {"eps_k": 1e-5},
        {"rope": _rope(is_neox_style=False)},
        {"rope": _rope(rotary_dim=D // 2)},
        {"rope": _rope(cos_cache=None)},
        {"rope": _rope(sin_cache=None)},
        # gfx942's platform fp8 saturates at 240, not the kernel's 448
        {
            "k_cache": torch.zeros(2, 16, HKV, D, dtype=torch.float8_e4m3fnuz),
            "v_cache": torch.zeros(2, 16, HKV, D, dtype=torch.float8_e4m3fnuz),
        },
        {"v_cache": torch.zeros(2, 16, HKV, D, dtype=torch.bfloat16)},
        # a SHUFFLE-layout pool is 5-D
        {
            "k_cache": torch.zeros(2, HKV, 4, 16, 16, dtype=torch.float8_e4m3fn),
            "v_cache": torch.zeros(2, HKV, 4, 16, 16, dtype=torch.float8_e4m3fn),
        },
        {"num_kv_heads": HKV + 1},
    ],
    ids=[
        "fp16-qkv",
        "strided-qkv",
        "eps-differ",
        "not-neox",
        "partial-rope",
        "no-cos",
        "no-sin",
        "fnuz-pool",
        "mixed-pool-dtypes",
        "shuffle-pool",
        "kv-head-mismatch",
    ],
)
def test_anything_else_keeps_the_chain(overrides):
    assert not _supported(**overrides)


def _attention_module():
    """The draft's attention module, or a skip where the AITER stack is absent."""
    try:
        return importlib.import_module("atom.models.kimi_k3_dflash")
    except Exception:
        try:
            importlib.import_module("aiter")
            gpu = torch.cuda.is_available()
        except Exception as e:  # noqa: BLE001 - any failure means no stack
            pytest.skip(f"the draft model needs the aiter stack, absent here: {e}")
        if not gpu:
            pytest.skip("the draft model needs a GPU: aiter probes the arch at import")
        raise


class _Pool:
    """`_caches` and the unfused chain's two ends, recording what they get."""

    def __init__(self, fp8_pool, slots, scale):
        dtype = torch.float8_e4m3fn if fp8_pool else torch.bfloat16
        self.k = torch.zeros(8, 16, HKV, D).to(dtype)
        self.v = torch.zeros_like(self.k)
        self.slots, self.scale = slots, scale
        self.read = {}

    def write(self, k, v, slot_mapping, k_cache, v_cache):
        flat_k, flat_v = k_cache.view(-1, HKV, D), v_cache.view(-1, HKV, D)
        for t, slot in enumerate(slot_mapping.tolist()):
            if slot < 0:
                continue
            kk, vv = k.reshape(-1, HKV, D)[t], v.reshape(-1, HKV, D)[t]
            if k_cache.dtype.itemsize == 1:
                kk, vv = fp8(kk, self.scale), fp8(vv, self.scale)
            flat_k[slot], flat_v[slot] = kk, vv

    def attend(self, q, k, v, attn_metadata, k_cache, v_cache):
        self.read.update(q=q.clone(), k=k.clone(), v=v.clone(), q_stride=q.stride(0))
        return q


def _fake_attention(module, pool, qkrope, qkv_input, rope):
    attn_cls = module.KimiK3DFlashAttention
    gen = torch.Generator().manual_seed(5)

    def weight():
        return (torch.rand(D, generator=gen) + 0.5).to(torch.bfloat16)

    q_norm = types.SimpleNamespace(weight=weight(), eps=EPS)
    k_norm = types.SimpleNamespace(weight=weight(), eps=EPS)

    def norm_into(norm):
        def run(x):
            return ref_norm(x.reshape(-1, D), norm.weight).reshape(x.shape)

        run.weight, run.eps = norm.weight, norm.eps
        return run

    def rotary(positions, q, k=None):
        c = rope.cos_cache.view(-1, D // 2)[positions].float().unsqueeze(1)
        s = rope.sin_cache.view(-1, D // 2)[positions].float().unsqueeze(1)
        out = []
        for x in (q,) if k is None else (q, k):
            x.copy_(ref_rope(x.reshape(x.shape[0], -1, D), c, s).reshape(x.shape))
            out.append(x)
        return out[0] if k is None else tuple(out)

    rotary.cos_cache, rotary.sin_cache = rope.cos_cache, rope.sin_cache
    rotary.is_neox_style, rotary.rotary_dim = True, D
    qn, kn = norm_into(q_norm), norm_into(k_norm)
    scale = torch.full((1,), float(pool.scale))
    attn = types.SimpleNamespace(
        qkrope=qkrope,
        num_heads=HQ,
        num_kv_heads=HKV,
        head_dim=D,
        q_size=HQ * D,
        kv_size=HKV * D,
        q_norm=qn,
        k_norm=kn,
        qk_norm=lambda q, k: (qn(q), kn(k)),
        rotary_emb=rotary,
        kv=types.SimpleNamespace(k_scale=scale, v_scale=scale),
        _qkrope_scales=None,
        # the fused projection: the test's hidden rows already are its output
        qkv_proj=types.SimpleNamespace(weight=None),
        _caches=lambda: (pool.k, pool.v),
        _write_block_kv=lambda k, v, slots, kc, vc: pool.write(k, v, slots, kc, vc),
        _attend_non_causal=pool.attend,
        o_proj=lambda o: o,
    )
    # `forward` calls the projection; `write_context_kv` multiplies the k | v
    # rows of its weight (`tgemm.mm`, patched in `wiring`)
    attn.qkv_proj = _Projection(qkv_input)
    attn._qkrope_cache = types.MethodType(attn_cls._qkrope_cache, attn)
    attn.q_norm.weight, attn.k_norm.weight = q_norm.weight, k_norm.weight
    return attn


class _Projection:
    """The draft's fused QKV linear for the wiring tests: the test's hidden
    rows already are its output, so it returns a copy (the kernel works in
    place). Its `weight` marks a k | v row slice for the patched `tgemm.mm`."""

    def __init__(self, hidden):
        self.weight = _RowSlicer()

    def __call__(self, hidden):
        return hidden.clone()


class _RowSlicer:
    def __getitem__(self, rows):
        return ("rows", rows)


def _kv_mm(x, w, bias):
    """`tgemm.mm` on `_Projection.weight[q_size:]`: the k | v columns of x."""
    assert w[0] == "rows" and bias is None
    return x[:, w[1]].contiguous()


@pytest.fixture
def wiring(monkeypatch):
    module = _attention_module()
    tuned_gemm = importlib.import_module("aiter.tuned_gemm")
    monkeypatch.setattr(tuned_gemm.tgemm, "mm", _kv_mm)
    n = 20
    gen = torch.Generator().manual_seed(7)
    hidden = (torch.randn(n, HQ * D + 2 * HKV * D, generator=gen) * 2).to(
        torch.bfloat16
    )
    cos, sin = _rope_tables(gen, 512)
    positions = torch.randint(0, 512, (n,), generator=gen)
    slots = torch.arange(n, dtype=torch.int64) * 3
    metadata = types.SimpleNamespace(slot_mapping=slots)
    forward_context = importlib.import_module("atom.utils.forward_context")
    monkeypatch.setattr(
        forward_context,
        "get_forward_context",
        lambda: types.SimpleNamespace(attn_metadata=metadata, kv_cache_data=None),
    )
    return types.SimpleNamespace(
        module=module,
        hidden=hidden,
        positions=positions,
        slots=slots,
        metadata=metadata,
        rope=types.SimpleNamespace(cos_cache=cos, sin_cache=sin),
    )


def _run_block(w, pool, qkrope):
    attn = _fake_attention(w.module, pool, qkrope, w.hidden, w.rope)
    w.module.KimiK3DFlashAttention.forward(attn, w.positions, w.hidden)
    return attn


@in_the_interpreter
@pytest.mark.parametrize("fp8_pool", [True, False])
def test_block_pass_with_the_flag_matches_the_chain(wiring, fp8_pool):
    n = wiring.hidden.shape[0]
    fused = _Pool(fp8_pool, wiring.slots, 1.0)
    chain = _Pool(fp8_pool, wiring.slots, 1.0)
    _run_block(wiring, fused, qkrope=True)
    _run_block(wiring, chain, qkrope=False)

    assert fused.read["q_stride"] == wiring.hidden.shape[1], "read gets strided views"
    _assert_close(fused.read["q"].reshape(n, HQ, D), chain.read["q"].reshape(n, HQ, D))
    _assert_close(
        fused.read["k"].reshape(n, HKV, D), chain.read["k"].reshape(n, HKV, D)
    )
    assert torch.equal(fused.read["v"], chain.read["v"])
    k_pool = fused.k.view(-1, HKV, D)[wiring.slots].float()
    k_want = chain.k.view(-1, HKV, D)[wiring.slots].float()
    assert float((k_pool - k_want).abs().max()) <= float(k_want.abs().max()) * 2**-3
    assert torch.equal(
        (
            fused.v.view(-1, HKV, D)[wiring.slots].contiguous().view(torch.uint8)
            if fp8_pool
            else fused.v.view(-1, HKV, D)[wiring.slots]
        ),
        (
            chain.v.view(-1, HKV, D)[wiring.slots].contiguous().view(torch.uint8)
            if fp8_pool
            else chain.v.view(-1, HKV, D)[wiring.slots]
        ),
    )


@in_the_interpreter
def test_the_flag_off_never_reaches_the_kernel(wiring, monkeypatch):
    kernels = _kernels()

    def boom(*args, **kwargs):
        raise AssertionError("the kernel ran with the flag off")

    monkeypatch.setattr(kernels, "draft_qk_norm_rope_cache", boom)
    pool = _Pool(True, wiring.slots, 1.0)
    _run_block(wiring, pool, qkrope=False)
    assert pool.read


@in_the_interpreter
def test_an_unsupported_pool_falls_back_to_the_chain(wiring, monkeypatch):
    kernels = _kernels()
    monkeypatch.setattr(
        kernels,
        "draft_qk_norm_rope_cache",
        lambda *a, **k: pytest.fail("the kernel ran on a pool it does not support"),
    )
    pool = _Pool(False, wiring.slots, 1.0)
    pool.k, pool.v = pool.k.to(torch.float16), pool.v.to(torch.float16)
    attn = _fake_attention(wiring.module, pool, True, wiring.hidden, wiring.rope)
    wiring.module.KimiK3DFlashAttention.forward(attn, wiring.positions, wiring.hidden)
    assert pool.read


@in_the_interpreter
def test_without_a_pool_the_warmup_pass_keeps_the_chain(wiring):
    attn_cls = wiring.module.KimiK3DFlashAttention
    pool = _Pool(True, wiring.slots, 1.0)
    attn = _fake_attention(wiring.module, pool, True, wiring.hidden, wiring.rope)
    attn._caches = lambda: (None, None)
    attn_cls.forward(attn, wiring.positions, wiring.hidden)
    assert not pool.read  # nothing to attend: the shaped zero output is returned


@in_the_interpreter
@pytest.mark.parametrize("fp8_pool", [True, False])
def test_context_write_leaves_q_and_fills_the_pool(wiring, fp8_pool):
    n = wiring.hidden.shape[0]
    fused = _Pool(fp8_pool, wiring.slots, 1.0)
    chain = _Pool(fp8_pool, wiring.slots, 1.0)
    for pool, flag in ((fused, True), (chain, False)):
        attn = _fake_attention(wiring.module, pool, flag, wiring.hidden, wiring.rope)
        wiring.module.KimiK3DFlashAttention.write_context_kv(
            attn, wiring.hidden, wiring.positions, wiring.slots
        )
    k_pool = fused.k.view(-1, HKV, D)[wiring.slots].float()
    k_want = chain.k.view(-1, HKV, D)[wiring.slots].float()
    assert float((k_pool - k_want).abs().max()) <= float(k_want.abs().max()) * 2**-3
    v_pool = fused.v.view(-1, HKV, D)[wiring.slots].float()
    v_want = chain.v.view(-1, HKV, D)[wiring.slots].float()
    assert torch.equal(v_pool, v_want)
    assert n == len(wiring.slots)


@pytest.mark.skipif(
    os.environ.get("RUN_QKROPE_GPU_TESTS") != "1"
    or not torch.cuda.is_available()
    or interpreting(),
    reason="GPU check against the real aiter chain (RUN_QKROPE_GPU_TESTS=1, "
    "compiled Triton, not the interpreter child)",
)
def test_gpu_against_the_real_aiter_chain():
    import aiter
    from aiter.rotary_embedding import get_rope

    from atom.model_ops.draft_qk_rope_cache import draft_qk_norm_rope_cache
    from atom.model_ops.layernorm import rmsnorm2d_fwd_

    dev, n = "cuda", 20
    torch.manual_seed(0)
    torch.set_default_dtype(torch.bfloat16)
    rope = get_rope(
        D, rotary_dim=D, max_position=262144, base=10000.0, is_neox_style=True
    )
    torch.set_default_dtype(torch.float32)
    rope = rope.to(dev)  # its cos / sin tables are built on the host
    # the op `RMSNorm` runs (its module needs a TP group, absent here)
    q_norm, k_norm = (
        types.SimpleNamespace(weight=(torch.rand(D) + 0.5).to(torch.bfloat16).to(dev))
        for _ in range(2)
    )
    width = HQ * D + 2 * HKV * D
    qkv = (torch.randn(n, width, device=dev) * 2).to(torch.bfloat16)
    pos = torch.randint(0, 200000, (n,), device=dev)
    slots = torch.randperm(64 * 16, device=dev)[:n].to(torch.int64)
    k0 = torch.zeros(64, 16, HKV, D, device=dev).to(torch.float8_e4m3fn)
    one = torch.ones(1, device=dev)

    q, k, v = qkv.clone().split([HQ * D, HKV * D, HKV * D], dim=-1)
    q = rmsnorm2d_fwd_(q, q_norm.weight, EPS, D)
    k = rmsnorm2d_fwd_(k, k_norm.weight, EPS, D)
    q, k = rope(pos, q, k)
    k_ref, v_ref = k0.clone(), torch.zeros_like(k0)
    aiter.reshape_and_cache_flash(
        k.reshape(n, HKV, D), v.reshape(n, HKV, D), k_ref, v_ref, slots, "fp8", one, one
    )
    fused = qkv.clone()
    k_got, v_got = k0.clone(), torch.zeros_like(k0)
    draft_qk_norm_rope_cache(
        fused, q_norm.weight, k_norm.weight, EPS, pos, rope.cos_cache, rope.sin_cache,
        slots, k_got, v_got, 1.0, 1.0, HQ, HKV, D,
    )  # fmt: skip
    torch.cuda.synchronize()
    _assert_close(fused[:, : HQ * D].reshape(n, -1), q.reshape(n, -1))
    _assert_close(fused[:, HQ * D : HQ * D + HKV * D], k.reshape(n, -1))
    assert (k_got.view(torch.uint8) == k_ref.view(torch.uint8)).float().mean() > 0.99
    assert torch.equal(v_got.view(torch.uint8), v_ref.view(torch.uint8))
