# SPDX-License-Identifier: MIT
"""FP8 prefill dispatch preserves output slices across AITER API versions."""

import importlib

import pytest
import torch

pytest.importorskip("aiter")
pytest.importorskip("triton")

from atom.model_ops.v4_kernels import paged_2src


@pytest.mark.parametrize("backend", ["opus", "asm", "asm_fallback"])
@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("supplied_output", [False, True])
def test_fp8_prefill_accepts_legacy_output_api(
    monkeypatch, backend, legacy, supplied_output
):
    calls = []
    expected = torch.full((2, 128, 4), 3.0)

    def old_kernel(*args):
        calls.append("legacy")
        return expected.clone()

    def new_kernel(*args, out=None):
        calls.append("current")
        if out is None:
            return expected.clone()
        out.copy_(expected)
        return out

    def unavailable_asm(*args, **kwargs):
        calls.append("fallback")
        raise RuntimeError("unsupported ASM shape")

    kernel = old_kernel if legacy else new_kernel
    opus = importlib.import_module("aiter.ops.pa_sparse_prefill_opus")
    monkeypatch.setattr(opus, "pa_sparse_prefill_fp8_opus", kernel)
    monkeypatch.setattr(paged_2src, "_HAS_PREFILL_ASM", backend != "opus")
    monkeypatch.setattr(paged_2src, "get_gfx_runtime", lambda: "gfx1250")
    monkeypatch.setattr(paged_2src.envs, "ATOM_FORCE_V4_PREFILL_OPUS", False)
    monkeypatch.setattr(
        paged_2src,
        "mla_sparse_prefill_fp8_asm",
        unavailable_asm if backend == "asm_fallback" else kernel,
    )
    parent = torch.full((4, 128, 4), -9.0)
    out = parent[1:3] if supplied_output else None
    q = torch.zeros_like(expected)
    k = torch.zeros(2, 4)
    indices = torch.zeros(0, dtype=torch.int32)
    indptr = torch.zeros(3, dtype=torch.int32)
    result = paged_2src.sparse_attn_v4_paged_2src(
        q,
        k,
        indices,
        indptr,
        k,
        indices,
        indptr,
        torch.zeros(128),
        0.5,
        out=out,
        unified_kv_rope=k,
        q_packed=q,
        q_rope=q,
        k_packed=k,
        k_rope=k,
    )
    torch.testing.assert_close(result, expected)
    if supplied_output:
        assert result is out
        assert torch.all(parent[[0, 3]] == -9)
    assert calls == (["fallback"] if backend == "asm_fallback" else []) + [
        "legacy" if legacy else "current"
    ]


def test_prefill_does_not_swallow_unrelated_type_errors():
    def broken(*args, **kwargs):
        raise TypeError("invalid packed tensor dtype")

    with pytest.raises(TypeError, match="invalid packed tensor dtype"):
        paged_2src._prefill_with_output(broken, (), torch.empty(1))
