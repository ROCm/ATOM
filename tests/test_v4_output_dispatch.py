# SPDX-License-Identifier: MIT
"""Exercise the production dispatch functions without importing GPU kernels."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def _dispatch(kind, kernel, gfx="gfx1250", **overrides):
    name = f"sparse_attn_v4_paged_{kind}"
    path = (
        Path(__file__).resolve().parents[1]
        / f"atom/model_ops/v4_kernels/paged_{kind}.py"
    )
    node = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == name
    )
    node.decorator_list = []
    namespace = {
        "torch": torch,
        "envs": SimpleNamespace(ATOM_FORCE_ATTN_TRITON=False),
        "get_gfx": lambda: gfx,
        "_HAS_OPUS": False,
        "pa_prefill_sparse": kernel,
        "pa_sparse_prefill_opus": kernel,
        "pa_decode_sparse": kernel,
        "_sparse_attn_v4_paged_2src_triton": kernel,
        "V4_DIM_QK": 512,
    }
    namespace.update(overrides)
    exec(  # noqa: S102 -- execute the checked-in dispatcher, without GPU imports
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace[name]


def _args(kind, q):
    indices = torch.empty(0, dtype=torch.int32)
    indptr = torch.zeros(q.shape[0] + 1, dtype=torch.int32)
    prefix = (q, q[0], indices, indptr)
    if kind == "2src":
        prefix += (q[0], indices, indptr)
    return (*prefix, torch.zeros(q.shape[1]), 0.5)


@pytest.mark.parametrize("kind", ["2src", "decode"])
@pytest.mark.parametrize("supplied_output", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_opaque_kernel_receives_output(kind, supplied_output, compiled):
    calls = []

    def kernel(*args, **kwargs):
        destination = kwargs["out"]
        calls.append(destination)
        return (
            args[0] + 1
            if destination is None
            else torch.add(args[0], 1, out=destination)
        )

    run = _dispatch(kind, kernel)
    if compiled:
        run = torch.compile(run, backend="eager", fullgraph=True)
    q = torch.arange(24.0).reshape(3, 2, 4)
    parent = torch.full((5, 2, 4), -99.0)
    out = parent[1:-1] if supplied_output else None
    result = run(*_args(kind, q), out=out)
    assert len(calls) == 1 and calls[0] is out
    if out is not None:
        assert result is out
    torch.testing.assert_close(result, q + 1)
    assert torch.all(parent[[0, -1]] == -99)


@pytest.mark.parametrize("kind", ["2src", "decode"])
def test_strided_destination_is_rejected_before_launch(kind):
    def kernel(*args, **kwargs):
        pytest.fail("invalid output reached the kernel")

    run = _dispatch(kind, kernel)
    q = torch.zeros(3, 2, 4)
    parent = torch.full((3, 2, 8), -99.0)
    with pytest.raises(ValueError, match="output must be contiguous"):
        run(*_args(kind, q), out=parent[:, :, ::2])
    assert torch.all(parent == -99)


@pytest.mark.parametrize("gfx", ["gfx950", "gfx1250"])
def test_bf16_prefill_uses_independent_strides_when_query_differs(gfx):
    def opus(*args, **kwargs):
        pytest.fail("OPUS must not store with incompatible query strides")

    calls = []

    def triton(q, *args, **kwargs):
        calls.append(kwargs["out"])
        return torch.add(q, 1, out=kwargs["out"])

    run = _dispatch("2src", triton, gfx, _HAS_OPUS=True, pa_sparse_prefill_opus=opus)
    q = torch.arange(48.0).reshape(6, 2, 4)[::2]
    parent = torch.full((5, 2, 4), -99.0)
    out = parent[1:-1]
    result = run(*_args("2src", q), out=out)
    assert result is out and len(calls) == 1 and calls[0] is out
    torch.testing.assert_close(out, q + 1)
    assert torch.all(parent[[0, -1]] == -99)


@pytest.mark.parametrize("path", ["bf16", "fp8", "hca", "prefill_asm"])
@pytest.mark.parametrize("invalid", ["rows", "heads", "dim", "dtype", "device"])
def test_decode_rejects_invalid_output_before_dispatch(path, invalid):
    def kernel(*args, **kwargs):
        pytest.fail("invalid output reached the kernel")

    run, q, kwargs = _decode_case(path, kernel)
    shape = list(q.shape)
    dtype, device = torch.bfloat16, q.device
    if invalid in ("rows", "heads", "dim"):
        shape[("rows", "heads", "dim").index(invalid)] -= 1
    elif invalid == "dtype":
        dtype = torch.float32
    else:
        device = "meta"
    out = torch.empty(shape, dtype=dtype, device=device)
    assert out.is_contiguous()
    with pytest.raises(ValueError, match="decode output must have shape"):
        run(*_args("decode", q), out=out, **kwargs)


def _decode_case(path, kernel):
    fp8 = path != "bf16"
    q = torch.zeros(
        3, 128 if path in ("hca", "prefill_asm") else 8, 512, dtype=torch.bfloat16
    )
    run = _dispatch(
        "decode",
        kernel,
        gfx="gfx950" if path == "bf16" else "gfx1250",
        os=SimpleNamespace(environ={"ATOM_USE_TRITON_ATTN": "1"}),
        envs=SimpleNamespace(ATOM_USE_V4_PREFILL_ASM_FOR_DECODE=path == "prefill_asm"),
        hca_persist=SimpleNamespace(
            wanted=lambda **kw: path == "hca",
            layout_ok=lambda *a: True,
            workspace_ready=lambda *a: True,
            hca_persist_decode=kernel,
        ),
        _sparse_attn_v4_paged_decode_asm=kernel,
        _sparse_attn_v4_paged_decode_triton=kernel,
        _sparse_attn_v4_paged_decode_prefill_asm=kernel,
    )
    kwargs = {}
    if fp8:
        kwargs = {
            "q_packed_in": torch.zeros_like(q, dtype=torch.uint8),
            "q_rope_in": torch.zeros(*q.shape[:2], 64, dtype=torch.bfloat16),
            "unified_kv_rope": torch.zeros(1, 64, dtype=torch.bfloat16),
            "empty_kv_indptr": torch.zeros(4, dtype=torch.int32),
        }
    return run, q, kwargs


@pytest.mark.parametrize("path", ["fp8", "hca", "prefill_asm"])
def test_fp8_decode_validates_unpadded_bf16_output(path):
    calls = []

    def kernel(*args, **kwargs):
        calls.append(kwargs["out"])
        return kwargs["out"]

    run, q, kwargs = _decode_case(path, kernel)
    parent = torch.full((5, *q.shape[1:]), -99.0, dtype=torch.bfloat16)
    out = parent[1:-1]
    args = (None, *_args("decode", q)[1:])
    assert run(*args, out=out, **kwargs) is out
    assert calls == [out]
    assert torch.all(parent == -99)
