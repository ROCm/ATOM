# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The sparse PA -> o_proj MXFP8 input (ATOM_MINIMAX_M3_PA_MXFP8_OUT) with more
than one local KV head (PP4 x TP1: 4 KV heads, 64 query heads a GPU). Runs on a
CPU runner: aiter, Triton and FlyDSL are stand-ins."""

import importlib
import importlib.abc
import importlib.util
import sys
from types import SimpleNamespace

import pytest
import torch

from tests.aiter_stub import _ExternalModuleLoader, atom_under_stubs

GROUP, D = 16, 128


class _FlydslFinder(importlib.abc.MetaPathFinder):
    """model.py imports the mono runner's FlyDSL kernels; serve them stand-ins."""

    def find_spec(self, fullname, path, target=None):
        if fullname == "flydsl" or fullname.startswith("flydsl."):
            return importlib.util.spec_from_loader(
                fullname, _ExternalModuleLoader(), is_package=True
            )
        return None


@pytest.fixture(scope="module")
def mods():
    finder = _FlydslFinder()
    saved = {n: m for n, m in sys.modules.items() if n.split(".")[0] == "flydsl"}
    sys.meta_path.insert(0, finder)
    try:
        with atom_under_stubs() as load:
            yield SimpleNamespace(
                model=load("atom.models.minimax_m3.model"),
                nvfp4=load("atom.model_ops.minimax_m3.sparse_attn_nvfp4"),
                base=load("atom.model_ops.base_attention"),
                sparse=load("atom.model_ops.minimax_m3.sparse_attn"),
                load=load,
            )
    finally:
        sys.meta_path.remove(finder)
        for name in [n for n in sys.modules if n.split(".")[0] == "flydsl"]:
            sys.modules.pop(name)
        sys.modules.update(saved)


def _conditions(mods, monkeypatch, kv_heads):
    monkeypatch.setattr(mods.model.envs, "ATOM_MINIMAX_M3_PA_MXFP8_OUT", True)
    monkeypatch.setattr(mods.model, "nvfp4_reads_page_table", lambda: True)
    attn = SimpleNamespace(
        num_kv_heads=kv_heads, impl=SimpleNamespace(kv_format="nvfp4")
    )
    o_proj = SimpleNamespace(native_a8_group_rows=1)
    return mods.model._sparse_pa_mxfp8_conditions(attn, o_proj)


@pytest.mark.parametrize("kv_heads", [1, 4])
def test_mxfp8_out_conditions_take_any_local_kv_heads(mods, monkeypatch, kv_heads):
    """Decided at construction; the attention splits the output by kv head."""
    assert all(_conditions(mods, monkeypatch, kv_heads).values())


def _mxfp8(x):
    """o_proj's group-32 MXFP8 quant of row-major BF16 rows ``[n, w]``:
    (E4M3 codes as uint8 [n, w], E8M0 scales as uint8 [n, w // 32])."""
    g = x.float().reshape(x.shape[0], -1, 32)
    m, e = torch.frexp(g.abs().amax(-1).clamp(min=1e-10) / 448.0)
    code = (torch.where(m == 0.5, e - 1, e) + 127).clamp(0, 254)
    scaled = g * torch.exp2((127 - code).float())[..., None]
    q = scaled.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
    return q.reshape(x.shape[0], -1), code.to(torch.uint8)


def _call(mods, monkeypatch, *, rows, bucket, kv_heads, mxfp8, reuse=False):
    """``_flydsl_attention`` against stand-in kernels that write, per (row, kv
    head) sequence, what aiter writes: the MXFP8 quant of that sequence's BF16
    rows (q itself here) into the views it is handed, pad sequences inert."""
    nvfp4 = mods.nvfp4
    seen = {}

    def kernel(*args, **kw):
        seen.update(kw)
        q_view = args[1]
        out, scale = kw.get("quantized_output"), kw.get("output_scale")
        seen["q_view"], seen["out"], seen["scale"] = q_view, out, scale
        if out is None or scale.dtype == torch.float32:
            return
        n = q_view.shape[0]
        codes, scales = _mxfp8(q_view.reshape(n, -1))
        out.view(torch.uint8)[:n] = codes.view(n, GROUP, D)
        scale.view(torch.uint8)[:n] = scales
        out.view(torch.uint8)[n:] = 0
        scale.view(torch.uint8)[n:] = 127

    pa = mods.load("aiter.ops.flydsl.pa_decode")
    monkeypatch.setattr(pa, "pa_decode", lambda *a, **kw: kernel(*a, **kw))
    gluon = mods.load("aiter.ops.triton.gluon.pa_decode_gluon")
    monkeypatch.setattr(gluon, "get_recommended_splits", lambda *_: 3)
    monkeypatch.setattr(
        mods.base, "_flydsl_prepare_explicit_plan", lambda **kw: seen.update(plan=kw)
    )
    monkeypatch.setattr(
        mods.base, "_flydsl_plan_scratch", lambda *a, **kw: (None, None, None)
    )
    monkeypatch.setattr(mods.sparse, "_cluster_resident_rows", lambda _: 240)
    monkeypatch.setattr(nvfp4.envs, "ATOM_M3_SPARSE_PREFILL_REUSE", reuse)
    monkeypatch.setattr(nvfp4.envs, "ATOM_M3_SPARSE_PREFILL_REUSE_MIN_ROWS", 1)
    monkeypatch.setattr(nvfp4.envs, "ATOM_M3_PA_FUSED_REDUCE", False)
    reuse_calls = []

    def reuse_kernel(*args, **kw):
        reuse_calls.append(kw)
        kernel(*args, **kw)

    monkeypatch.setattr(
        nvfp4, "_prefill_reuse_kernel", lambda: (reuse_kernel, lambda *a: True)
    )
    monkeypatch.setattr(nvfp4, "_prefill_reuse_impl", dict)
    heads = GROUP * kv_heads
    q = torch.randn(rows, heads, D, dtype=torch.bfloat16)
    pages = 4
    payload = torch.zeros(pages, kv_heads, 128, 64, dtype=torch.uint8)
    scales = torch.zeros(pages, kv_heads, 128, 8, dtype=torch.uint8)
    out = torch.full((bucket, heads * D), 0x55, dtype=torch.uint8)
    width = heads * D // 32 if mxfp8 else 1
    sdtype = torch.float8_e8m0fnu if mxfp8 else torch.float32
    out_scale = torch.full((bucket, width), 0x33, dtype=torch.uint8)
    out_scale = out_scale.view(sdtype) if mxfp8 else out_scale.float()
    seqs = rows * kv_heads
    nvfp4._flydsl_attention(
        q, payload, scales, payload, scales,
        torch.zeros(seqs, 1, dtype=torch.int32), torch.ones(seqs, dtype=torch.int32),
        D**-0.5, None, None,
        quantized_output=out.view(torch.float8_e4m3fn), output_scale=out_scale,
        quantized_direct=True,
    )  # fmt: skip
    return SimpleNamespace(
        q=q, out=out, scale=out_scale, seen=seen, reuse_calls=reuse_calls
    )


@pytest.mark.parametrize("kv_heads", [1, 4])
def test_mxfp8_output_is_row_major_o_proj_input(mods, monkeypatch, kv_heads):
    """Per-(row, kv head) sequence writes through the views are bit for bit
    o_proj's own quant of the row-major ``[rows, heads * D]`` output, and the
    padded bucket rows are inert."""
    rows, bucket = 6, 8
    r = _call(
        mods, monkeypatch, rows=rows, bucket=bucket, kv_heads=kv_heads, mxfp8=True
    )
    assert r.seen["quantized_pad_rows"] == (bucket - rows) * kv_heads
    assert r.seen["out"].shape == (bucket * kv_heads, GROUP, D)
    assert r.seen["scale"].shape == (bucket * kv_heads, GROUP * D // 32)
    assert r.seen["out"].data_ptr() == r.out.data_ptr()
    assert r.seen["scale"].data_ptr() == r.scale.data_ptr()
    codes, scales = _mxfp8(r.q.reshape(rows, -1))
    assert torch.equal(r.out[:rows], codes)
    assert torch.equal(r.scale.view(torch.uint8)[:rows], scales)
    assert (r.out[rows:] == 0).all() and (r.scale.view(torch.uint8)[rows:] == 127).all()


def test_per_token_fp8_output_still_needs_one_kv_head(mods, monkeypatch):
    """A per-token scale spans every head of a row; the model never asks for it
    with more than one kv head (``sparse_pa_output_quant_supported``)."""
    with pytest.raises(AssertionError, match="one kv head"):
        _call(mods, monkeypatch, rows=4, bucket=4, kv_heads=4, mxfp8=False)


@pytest.mark.parametrize("kv_heads", [1, 4])
def test_prefill_reuse_takes_the_mxfp8_output_for_one_kv_head(
    mods, monkeypatch, kv_heads
):
    """The image's aiter pads the reuse kernel's MXFP8 bucket for one kv head
    only; with more the prefill keeps the paged decode."""
    r = _call(
        mods, monkeypatch, rows=6, bucket=8, kv_heads=kv_heads, mxfp8=True, reuse=True
    )
    assert bool(r.reuse_calls) == (kv_heads == 1)
    codes, _ = _mxfp8(r.q.reshape(6, -1))
    assert torch.equal(r.out[:6], codes)
