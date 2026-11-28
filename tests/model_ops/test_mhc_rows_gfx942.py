# SPDX-License-Identifier: MIT
"""The gfx942 rows seam (mhc_rows_gfx942.py): when it applies and that
v41_mhc_pre_delayed hands it the seam, on CPU with the kernels stubbed; and its
outputs against the Triton seam on a gfx942 GPU."""

import pytest
import torch
from aiter_stub import atom_without_aiter

ROWS_MIN, HC, MIX = 1024, 4, 24


@pytest.fixture(scope="module")
def mhc():
    with atom_without_aiter() as load:
        yield load("atom.model_ops.deepseek_v41.mhc_pre_delayed")


def _common():
    return {
        "rms_eps": 1e-6,
        "hc_eps": 1e-6,
        "sinkhorn_iters": 20,
        "post_mult": 2.0,
        "norm_eps": 1e-6,
    }


def _applies_case(rows, hidden):
    return {
        "residual": torch.zeros(rows, HC, hidden, dtype=torch.bfloat16),
        "hc_fn": torch.zeros(MIX, HC * hidden),
        "norm_weight": torch.ones(hidden, dtype=torch.bfloat16),
        "sublayer_output": torch.zeros(rows, hidden, dtype=torch.bfloat16),
        "w_preshuffle_bf16": False,
        "fuse_quant": False,
    }


@pytest.mark.parametrize(
    "change, expected",
    [
        ({}, True),
        ({"env": "0"}, False),  # ATOM_DSV41_MHC_ROWS=0 keeps the backend
        ({"backend": "triton"}, False),  # forces Triton, gfx942 included
        ({"rows": ROWS_MIN - 1}, False),  # decode and short chunks
        ({"hidden": 768}, False),  # hidden not a multiple of 512
        ({"w_preshuffle_bf16": True}, False),  # packed BF16 fn: HIP backend
        ({"fuse_quant": True}, False),
        ({"fn_dtype": torch.bfloat16}, False),  # the GEMM splits an FP32 fn
        ({"sublayer_output": None}, True),  # a seam without a post
        ({"kernels": None}, False),  # hipcc failed: the backend decides
    ],
)
def test_rows_seam_applies(change, expected, monkeypatch, mhc):
    """applies() with the arch probe and the hipcc build stubbed."""
    rows_mod = mhc.mhc_rows_gfx942
    monkeypatch.setenv("ATOM_DSV41_MHC_ROWS", change.get("env", "1"))
    monkeypatch.setenv("ATOM_DSV41_MHC_BACKEND", change.get("backend", "auto"))
    monkeypatch.setattr(rows_mod, "_arch_ok", lambda: True)
    kernels = change.get("kernels", (object(), {}))
    monkeypatch.setattr(rows_mod, "_kernels", lambda: kernels)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    case = _applies_case(change.get("rows", ROWS_MIN), change.get("hidden", 512))
    for key in ("w_preshuffle_bf16", "fuse_quant", "sublayer_output"):
        if key in change:
            case[key] = change[key]
    if "fn_dtype" in change:
        case["hc_fn"] = case["hc_fn"].to(change["fn_dtype"])
    assert rows_mod.applies(**case) is expected


@pytest.mark.parametrize("with_post", [True, False])
def test_seam_takes_rows_path_when_it_applies(with_post, monkeypatch, request, mhc):
    """v41_mhc_pre_delayed calls the rows seam ahead of the ASM and Triton
    seams, and returns its outputs in the seam's shapes."""
    rows, hidden = ROWS_MIN, 512
    calls = []

    def rows_seam(residual, *args, sublayer_output=None, **kwargs):
        calls.append("rows")
        return (
            residual.clone(),
            torch.zeros(rows, hidden),
            torch.zeros(rows, HC),
            torch.zeros(rows, HC),
            torch.zeros(rows, HC, HC),
        )

    def other(name):
        def kernel(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"{name} seam called")

        return kernel

    monkeypatch.setattr(mhc.mhc_rows_gfx942, "applies", lambda *a: True)
    monkeypatch.setattr(mhc.mhc_rows_gfx942, "pre_delayed_rows", rows_seam)
    monkeypatch.setattr(mhc, "mega_mhc", other("mega"))
    monkeypatch.setattr(mhc, "mhc_fused_post_pre_delayed_rmsnorm", other("triton"))
    monkeypatch.setattr(
        mhc, "mhc_fused_post_pre_delayed_asm_supported", lambda *a: True
    )
    mhc.use_asm_seam.cache_clear()
    request.addfinalizer(mhc.use_asm_seam.cache_clear)

    residual = torch.randn(1, rows, HC, hidden).bfloat16()
    pre = torch.full((1, rows, HC), 1 / HC)
    post_kw = (
        {
            "sublayer_output": torch.randn(1, rows, hidden).bfloat16(),
            "post_mix": torch.ones_like(pre),
            "combination": torch.eye(HC).expand(1, rows, HC, HC).contiguous(),
        }
        if with_post
        else {}
    )
    seam = getattr(mhc.v41_mhc_pre_delayed, "__wrapped__", mhc.v41_mhc_pre_delayed)
    out = seam(
        residual,
        pre,
        torch.randn(MIX, HC * hidden),
        torch.ones(3),
        torch.zeros(MIX),
        torch.ones(hidden),
        **_common(),
        **post_kw,
    )
    assert calls == ["rows"]
    shapes = [(1, rows, hidden), (1, rows, HC), (1, rows, HC), (1, rows, HC, HC)]
    if with_post:
        shapes.insert(0, tuple(residual.shape))
    assert [tuple(t.shape) for t in out] == shapes


def _gfx942_rows_ready():
    if not torch.cuda.is_available():
        return False
    try:
        from atom.model_ops.deepseek_v41 import mhc_rows_gfx942 as rows_mod

        return rows_mod._arch_ok() and rows_mod._kernels() is not None
    except ImportError:
        return False


@pytest.mark.skipif(not _gfx942_rows_ready(), reason="gfx942 rows seam kernels")
@pytest.mark.parametrize("with_post", [True, False])
@pytest.mark.parametrize("rows", [ROWS_MIN, 12288])  # both GEMM orientations
def test_rows_seam_matches_triton_seam(rows, with_post, monkeypatch, request):
    """The rows seam against the Triton seam at the model's hidden size. Same
    math in another order, so BF16 outputs may differ by an ULP, the gates
    barely."""
    from atom.model_ops.deepseek_v41 import mhc_pre_delayed as mhc

    request.addfinalizer(mhc.use_asm_seam.cache_clear)
    torch.manual_seed(7)
    hidden, dev = 5120, "cuda"
    residual = torch.randn(1, rows, HC, hidden, device=dev).bfloat16()
    pre = torch.softmax(torch.randn(1, rows, HC, device=dev), -1)
    fn = torch.randn(MIX, HC * hidden, device=dev) * 0.01
    scale = torch.rand(3, device=dev) + 0.5
    base = torch.randn(MIX, device=dev) * 0.2
    norm_weight = (1 + 0.1 * torch.randn(hidden, device=dev)).bfloat16()
    post_kw = (
        {
            "sublayer_output": torch.randn(1, rows, hidden, device=dev).bfloat16(),
            "post_mix": torch.rand(1, rows, HC, device=dev) * 2,
            "combination": torch.softmax(torch.randn(1, rows, HC, HC, device=dev), -1),
        }
        if with_post
        else {}
    )
    args = (residual, pre, fn, scale, base, norm_weight)

    monkeypatch.setenv("ATOM_DSV41_MHC_ROWS", "1")
    monkeypatch.setenv("ATOM_DSV41_MHC_BACKEND", "auto")
    assert mhc.mhc_rows_gfx942.applies(
        residual, fn, norm_weight, post_kw.get("sublayer_output"), False, False
    )
    got = mhc.v41_mhc_pre_delayed(*args, **_common(), **post_kw)
    monkeypatch.setenv("ATOM_DSV41_MHC_BACKEND", "triton")
    mhc.use_asm_seam.cache_clear()
    assert not mhc.mhc_rows_gfx942.applies(
        residual, fn, norm_weight, post_kw.get("sublayer_output"), False, False
    )
    want = mhc.v41_mhc_pre_delayed(*args, **_common(), **post_kw)

    n_bf16 = 2 if with_post else 1  # new residual (with a post) and normed
    for g, w in zip(got[:n_bf16], want[:n_bf16]):
        torch.testing.assert_close(g, w, rtol=2**-7, atol=2**-7)
    for g, w in zip(got[n_bf16:], want[n_bf16:]):
        torch.testing.assert_close(g, w, rtol=1e-3, atol=1e-4)
