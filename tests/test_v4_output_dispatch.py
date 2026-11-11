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
@pytest.mark.parametrize("old_api", [False, True])
def test_kernel_typeerror_propagates_without_retry(kind, old_api):
    calls = []

    def kernel(*args, **kwargs):
        calls.append(kwargs)
        if old_api:
            raise TypeError("unexpected keyword argument 'out'")
        raise TypeError("invalid packed tensor dtype")

    run = _dispatch(kind, kernel)
    q = torch.zeros(3, 2, 4)
    out = torch.empty_like(q)
    with pytest.raises(TypeError, match="out" if old_api else "packed tensor"):
        run(*_args(kind, q), out=out)
    assert len(calls) == 1 and calls[0]["out"] is out


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
