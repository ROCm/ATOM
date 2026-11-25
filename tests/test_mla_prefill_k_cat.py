# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""CPU bit checks for the shared K copy and native BF16 dispatch.

Kernel cases run in child processes so TRITON_INTERPRET precedes the Triton
import. No AITER import or architecture-cache mutation is needed. Integration
checks execute the production method ASTs with projection/attention stand-ins.
"""

import ast
import os
import subprocess
import sys
from pathlib import Path
from types import MethodType, ModuleType, SimpleNamespace

import pytest
import torch

pytest.importorskip(
    "triton", reason="the dedicated CPU interpreter CI step installs Triton"
)

from atom.model_ops import triton_mla_prefill_k_cat as kcat

ROOT = Path(__file__).resolve().parents[1]


def _load_methods(path, class_name, names, scope):
    """Execute the production methods without importing the GPU dependency tree."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    cls.bases, cls.keywords, cls.decorator_list = [], [], []
    cls.body = [
        n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    assert {n.name for n in cls.body} == set(names)
    for method in cls.body:
        method.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    # Only repository-owned method ASTs are executed.
    exec(  # noqa: S102
        compile(ast.fix_missing_locations(module), str(path), "exec"), scope
    )
    return scope[class_name]


# Projection and attention are stand-ins; the dispatch/method bodies are the
# repository source. AITER is absent on the CPU CI runner.
mla = ModuleType("mla_dispatch_probe")
mla.__dict__.update(
    torch=torch,
    cat_prefill_k=kcat.cat_prefill_k,
    use_triton_gemm=lambda: False,
    flash_attn_varlen_func=None,
    fused_gemm_afp4wfp4_preshuffle_split_cat=None,
    fused_gemm_a8w8_blockscale_preshuffle_split_cat=None,
)
_load_methods(
    ROOT / "atom/model_ops/attention_mla.py",
    "MLAAttention",
    [
        "_check_flydsl_fp8_mha",
        "_prepare_prefill_k",
        "_drop_rope_pad",
        "_flash_attn_prefill",
        "_forward_prefill_mha",
    ],
    mla.__dict__,
)


def _interpreter(code):
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            "import importlib.util; "
            f"s = importlib.util.spec_from_file_location('probe', {__file__!r}); "
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); "
            "m.kcat._prefill_k_cat_supported = lambda *a: True; " + code,
        ],
        env={**os.environ, "TRITON_INTERPRET": "1", "HIP_VISIBLE_DEVICES": ""},
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert out.returncode == 0, out.stdout + out.stderr[-4000:]


SPECIAL = [0x8000, 0x7FC1, 0xFFC3, 0x7F80, 0xFF80, 0x0001, 0x807F, 0x0000]


def _special_bits():
    return torch.tensor(
        [v - 0x10000 if v >= 0x8000 else v for v in SPECIAL], dtype=torch.int16
    )


def _bits(x):
    return x.contiguous().view(torch.int16)


def _inputs(tokens, heads, nope=128, v=128, rope=64, row=2112, seed=0):
    """k_nope / v as the kv_b_proj split views, k_rope as a [T, 1, rope] view of
    the qkv_a row, like the DSR1 prefill hands them over."""
    g = torch.Generator().manual_seed(seed)
    kv = torch.randn(tokens, heads * (nope + v), generator=g).to(torch.bfloat16)
    qkv_a = torch.randn(tokens, row, generator=g).to(torch.bfloat16)
    s = _special_bits()
    kv.view(torch.int16)[0, : s.numel()] = s
    last_nope_end = (heads - 1) * (nope + v) + nope
    kv.view(torch.int16)[-1, last_nope_end - s.numel() : last_nope_end] = s
    qkv_a.view(torch.int16)[0, row - rope : row - rope + s.numel()] = s
    qkv_a.view(torch.int16)[-1, row - s.numel() :] = s
    k_nope, v_ = kv.view(tokens, heads, nope + v).split([nope, v], dim=-1)
    k_rope = qkv_a[:, row - rope :].unsqueeze(1)
    return k_nope, v_, k_rope


def _ref(k_nope, k_rope):
    return torch.cat((k_nope, k_rope.expand((*k_nope.shape[:-1], -1))), dim=-1)


def _assert_same(new, ref):
    assert new.shape == ref.shape and new.dtype == ref.dtype
    assert new.stride() == ref.stride() and new.is_contiguous()
    assert torch.equal(_bits(new), _bits(ref))


@pytest.fixture
def kernel_on_cpu(monkeypatch):
    monkeypatch.setattr(kcat, "_prefill_k_cat_supported", lambda *a: True)


@pytest.mark.parametrize(
    "tokens,heads,nope,rope",
    [
        (1, 128, 128, 64),
        (7, 128, 128, 64),
        (33, 16, 128, 64),
        (5, 20, 128, 64),
        (3, 1, 128, 64),
        (4, 64, 256, 64),
    ],
)
def test_kernel_bitwise_equals_cat(kernel_on_cpu, tokens, heads, nope, rope):
    if os.environ.get("TRITON_INTERPRET") != "1":
        _interpreter(
            f"m.test_kernel_bitwise_equals_cat(None, {tokens}, {heads}, {nope}, {rope})"
        )
        return
    k_nope, _, k_rope = _inputs(tokens, heads, nope=nope, v=128, rope=rope)
    assert not k_nope.is_contiguous()
    _assert_same(kcat.cat_prefill_k(k_nope, k_rope), _ref(k_nope, k_rope))


@pytest.mark.parametrize("head_block", [1, 4, 32, 64])
def test_kernel_head_block_independent(kernel_on_cpu, monkeypatch, head_block):
    if os.environ.get("TRITON_INTERPRET") != "1":
        _interpreter(
            f"m.test_kernel_head_block_independent(None, m.pytest.MonkeyPatch(), {head_block})"
        )
        return
    monkeypatch.setattr(kcat, "_PREFILL_K_CAT_HEAD_BLOCK", head_block)
    k_nope, _, k_rope = _inputs(9, 48, seed=head_block)
    _assert_same(kcat.cat_prefill_k(k_nope, k_rope), _ref(k_nope, k_rope))


def test_kernel_contiguous_and_expanded_rope(kernel_on_cpu):
    if os.environ.get("TRITON_INTERPRET") != "1":
        _interpreter("m.test_kernel_contiguous_and_expanded_rope(None)")
        return
    k_nope = torch.randn(6, 8, 128).to(torch.bfloat16)
    k_rope = torch.randn(1, 1, 64).to(torch.bfloat16).expand(6, 1, 64)
    _assert_same(kcat.cat_prefill_k(k_nope, k_rope), _ref(k_nope, k_rope))


def test_supported_gate(monkeypatch):
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    arch = {"gfx1250": True}
    monkeypatch.setattr(kcat, "_kernel_arch", lambda index: arch["gfx1250"])
    tokens = kcat._PREFILL_K_CAT_MIN_TOKEN_HEADS // 16
    k_nope, _, k_rope = _inputs(tokens, 16)
    ok = kcat._prefill_k_cat_supported
    assert ok(k_nope, k_rope)
    arch["gfx1250"] = False
    assert not ok(k_nope, k_rope)
    arch["gfx1250"] = True
    # The gate counts token-heads: fewer tokens, or fewer heads, decline.
    assert not ok(k_nope[:-1], k_rope[:-1])
    assert not ok(k_nope[:, :8], k_rope)
    wide, _, wide_rope = _inputs(tokens // 8, 128)
    assert ok(wide, wide_rope)
    assert not ok(k_nope[:0], k_rope[:0])
    assert not ok(k_nope[:, :0], k_rope)
    assert not ok(k_nope[..., :0], k_rope)
    assert not ok(k_nope, k_rope[..., :0])
    assert not ok(k_nope, k_rope.expand(-1, 16, -1))
    assert not ok(k_nope, k_rope.squeeze(1))
    assert not ok(k_nope, k_rope.float())
    assert not ok(k_nope[..., :96], k_rope)
    assert not ok(k_nope, k_rope[..., :48])
    assert not ok(k_nope.transpose(1, 2).contiguous().transpose(1, 2), k_rope)
    assert not ok(k_nope, k_rope.transpose(0, 2).contiguous().transpose(0, 2))


@pytest.mark.parametrize("copies", [False, True])
def test_fallback_is_the_sites_own(monkeypatch, copies):
    """Below the gate a site keeps what it did on main: torch.cat, or with
    `copies` the two-copy write."""
    k_nope, _, k_rope = _inputs(5, 16)
    assert not kcat._prefill_k_cat_supported(k_nope, k_rope)
    wide = k_rope.expand(-1, 16, -1)
    refs = [_ref(k_nope, k_rope), _ref(k_nope, wide)]
    calls = []
    real_cat = torch.cat
    monkeypatch.setattr(
        kcat.torch, "cat", lambda *a, **k: calls.append(1) or real_cat(*a, **k)
    )
    outs = [kcat.cat_prefill_k(k_nope, r, copies=copies) for r in (k_rope, wide)]
    monkeypatch.undo()
    assert len(calls) == (0 if copies else 2)
    for out, ref in zip(outs, refs):
        _assert_same(out, ref)


def test_vllm_plugin_k_sites_keep_main_off_gfx1250():
    """The plugin is not imported on the CPU job, so check its source: every
    MLA prefill K is `cat_prefill_k` on a gfx1250 layer (`_prefill_k_cat_arch`,
    decided once at init) and main's own expression on any other, so other
    arches run no per-call gate. The two sites that used two copies on main
    keep them (`copies=True`, `_concat_k_nope_k_pe`)."""
    tree = ast.parse(
        (ROOT / "atom/plugin/vllm/attention/layer_mla.py").read_text(encoding="utf-8")
    )
    calls = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and ast.unparse(n.func) == "cat_prefill_k"
    ]
    sites = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.IfExp)
        and isinstance(n.body, ast.Call)
        and ast.unparse(n.body.func) == "cat_prefill_k"
    ]
    assert len(calls) == len(sites) == 4
    expanded = "torch.cat((k_nope, k_pe.expand((*k_nope.shape[:-1], -1))), dim=-1)"
    kinds = []
    for site in sorted(sites, key=lambda n: n.lineno):
        assert ast.unparse(site.test) == "self._prefill_k_cat_arch"
        copies = any(
            kw.arg == "copies" and ast.literal_eval(kw.value)
            for kw in site.body.keywords
        )
        main = "self._concat_k_nope_k_pe(k_nope, k_pe)" if copies else expanded
        assert ast.unparse(site.orelse) == main
        kinds.append(copies)
    # The two prefill-context sites used two copies on main, the two
    # `_forward_prefill` sites torch.cat.
    assert kinds == [True, True, False, False]


def test_layers_decide_the_kernel_arch_once():
    """`MLAAttention.__init__` decides the arch once; `kernel_arch` is False
    without a CUDA device."""
    init = next(
        n
        for n in ast.walk(
            ast.parse(
                (ROOT / "atom/model_ops/attention_mla.py").read_text(encoding="utf-8")
            )
        )
        if isinstance(n, ast.FunctionDef)
        and n.name == "__init__"
        and "self._flydsl_fp8_mha_ok" in ast.unparse(n)
    )
    assert "self._prefill_k_cat_arch = kernel_arch()" in ast.unparse(init)
    if not torch.cuda.is_available():
        assert kcat.kernel_arch() is False


class _KvB:
    """BF16 kv_b_proj stand-in: the MLPerf checkpoint leaves it unquantized."""

    weight = torch.empty(1, dtype=torch.bfloat16)
    weight_scale = None

    def __init__(self, kv):
        self.kv = kv

    def __call__(self, x):
        return self.kv


def _impl(fp8_ok, heads, kv=None):
    impl = SimpleNamespace(
        _flydsl_fp8_mha_ok=fp8_ok,
        _prefill_k_cat_arch=False,
        use_flydsl_fp8_prefill_attn=False,
        num_heads=heads,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rope_is_zero_pad=False,
        scale=192**-0.5,
        kv_b_proj=_KvB(kv),
        o_proj=lambda x: x,
    )
    for name in (
        "_check_flydsl_fp8_mha",
        "_prepare_prefill_k",
        "_drop_rope_pad",
        "_flash_attn_prefill",
    ):
        setattr(impl, name, MethodType(getattr(mla.MLAAttention, name), impl))
    return impl


@pytest.mark.parametrize("gfx1250", [True, False])
def test_prepare_prefill_k_zero_rope_pad(gfx1250):
    # On gfx1250 the pad lanes are sliced off before attention, so no copy is
    # made; other arches concatenate K as on main.
    k_nope, _, k_rope = _inputs(3, 16)
    impl = _impl(False, 16)
    impl._prefill_k_cat_arch = gfx1250
    impl.rope_is_zero_pad = True
    pad = torch.zeros_like(k_rope)
    k, rope = impl._prepare_prefill_k(k_nope, pad)
    assert rope is None
    if gfx1250:
        assert k is k_nope
    else:
        _assert_same(k, _ref(k_nope, pad))


def test_prepare_prefill_k_fp8_split_unchanged():
    k_nope, _, k_rope = _inputs(3, 16)
    k, rope = _impl(True, 16)._prepare_prefill_k(k_nope, k_rope)
    assert k is k_nope and rope is k_rope


@pytest.mark.parametrize("force_kernel", [False, True])
@pytest.mark.parametrize("gfx1250", [False, True])
def test_forward_prefill_mha_hands_attention_the_cat(
    monkeypatch, force_kernel, gfx1250
):
    if force_kernel:
        if os.environ.get("TRITON_INTERPRET") != "1":
            _interpreter(
                f"m.test_forward_prefill_mha_hands_attention_the_cat(m.pytest.MonkeyPatch(), True, {gfx1250})"
            )
            return
        monkeypatch.setattr(kcat, "_prefill_k_cat_supported", lambda *a: True)
    monkeypatch.setattr(mla, "use_triton_gemm", lambda: True)
    tokens, heads = 11, 16
    g = torch.Generator().manual_seed(3)
    kv = torch.randn(tokens, heads * 256, generator=g).to(torch.bfloat16)
    qkv_a = torch.randn(tokens, 2112, generator=g).to(torch.bfloat16)
    rope2d = qkv_a[:, 2048:]
    q = torch.randn(tokens, heads, 192, generator=g).to(torch.bfloat16)
    impl = _impl(False, heads, kv)
    impl._prefill_k_cat_arch = gfx1250
    seen = {}

    def attend(**kw):
        seen.update(kw)
        return torch.ones(tokens, heads, 128, dtype=torch.bfloat16)

    monkeypatch.setattr(mla, "flash_attn_varlen_func", attend)
    built = []
    monkeypatch.setattr(
        mla, "cat_prefill_k", lambda *a: built.append(1) or kcat.cat_prefill_k(*a)
    )
    cu = torch.tensor([0, 4, 5, tokens], dtype=torch.int32)
    meta = SimpleNamespace(
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=6,
        max_seqlen_k=6,
        min_seqlen_q=1,
        dropout_p=0.0,
    )
    out = mla.MLAAttention._forward_prefill_mha(impl, q, None, rope2d, None, meta)
    assert out.shape == (tokens, heads * 128)
    # gfx1250 layers build K with cat_prefill_k; others with main's torch.cat
    assert built == ([1] if gfx1250 else [])
    k_nope, v = kv.view(tokens, heads, 256).split([128, 128], dim=-1)
    _assert_same(seen["k"], _ref(k_nope, rope2d.unsqueeze(1)))
    assert seen["v"].data_ptr() == v.data_ptr() and seen["v"].stride() == v.stride()
    assert seen["q"] is q and seen["causal"] is True
