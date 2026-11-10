# SPDX-License-Identifier: MIT
"""AITER wrappers must not turn an old output API into an advertised new one."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

# This helper has no AITER dependency; keep its compatibility tests in CPU CI.
_path = (
    Path(__file__).resolve().parents[1] / "atom/model_ops/v4_kernels/output_buffer.py"
)
_spec = importlib.util.spec_from_file_location("v4_output_buffer_test", _path)
output_buffer = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = output_buffer
_spec.loader.exec_module(output_buffer)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("expose_body", [False, True])
def test_opaque_wrapper_preserves_output_slice(legacy, expose_body):
    calls = []

    def old(x):
        calls.append(None)
        return x + 1

    def new(x, *, out=None):
        calls.append(out)
        return x + 1 if out is None else torch.add(x, 1, out=out)

    body = old if legacy else new

    def wrapper(*args, **kwargs):
        return body(*args, **kwargs)

    if expose_body:
        wrapper._eager_body = body
    parent = torch.full((6,), -9.0)
    out = parent[1:-1]
    actual = output_buffer.call_with_output(wrapper, (torch.ones(4),), out)
    assert actual is out
    torch.testing.assert_close(out, torch.full((4,), 2.0))
    torch.testing.assert_close(parent[[0, -1]], torch.full((2,), -9.0))
    assert len(calls) == 1
    assert calls[0] is (out if expose_body and not legacy else None)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("expose_body", [False, True])
def test_opaque_output_api_compiles(legacy, expose_body):
    def old(x):
        return x + 1

    def new(x, *, out=None):
        return x + 1 if out is None else torch.add(x, 1, out=out)

    body = old if legacy else new

    def wrapper(*args, **kwargs):
        return body(*args, **kwargs)

    if expose_body:
        wrapper._eager_body = body

    def run(x, out):
        return output_buffer.call_with_output(wrapper, (x,), out)

    out = torch.empty(4)
    actual = torch.compile(run, fullgraph=True, backend="eager")(torch.ones(4), out)
    assert actual is out
    torch.testing.assert_close(out, torch.full((4,), 2.0))


@pytest.mark.parametrize("accepts_output", [False, True])
def test_kernel_type_errors_propagate(accepts_output):
    def legacy(x):
        raise TypeError("invalid packed tensor dtype")

    def current(x, *, out=None):
        raise TypeError("invalid packed tensor dtype")

    kernel = current if accepts_output else legacy
    with pytest.raises(TypeError, match="invalid packed tensor dtype"):
        output_buffer.call_with_output(kernel, (torch.empty(1),), torch.empty(1))


@pytest.mark.parametrize("expose_body", [False, True])
def test_real_aiter_guard_accepts_legacy_output_api(monkeypatch, expose_body):
    guard = pytest.importorskip("aiter.jit.utils.torch_guard").torch_compile_guard

    def legacy_body(x: torch.Tensor) -> torch.Tensor:
        return x + 1

    # Each parametrization owns its registered torch operator.
    legacy_body.__name__ = f"test_v4_legacy_output_{expose_body}"
    kernel = guard(mutates_args=[])(legacy_body)
    if not expose_body:
        monkeypatch.delattr(kernel, "_eager_body", raising=False)
    out = torch.empty(4)
    actual = output_buffer.call_with_output(kernel, (torch.ones(4),), out)
    assert actual is out
    torch.testing.assert_close(out, torch.full((4,), 2.0))


@pytest.mark.parametrize("compiled", [False, True])
def test_real_aiter_guard_writes_directly_to_output(compiled):
    guard = pytest.importorskip("aiter.jit.utils.torch_guard").torch_compile_guard
    destinations = []

    def body(x: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        assert out is not None, "output-capable guard took the allocating fallback"
        destinations.append(out.data_ptr())
        return torch.add(x, 1, out=out)

    body.__name__ = f"test_v4_mutating_output_{compiled}"
    kernel = guard(mutates_args=["out"], gen_fake=lambda x, out=None: out)(body)

    def run(x, out):
        return output_buffer.call_with_output(kernel, (x,), out)

    if compiled:
        run = torch.compile(run, fullgraph=True, backend="eager")
    parent = torch.full((6,), -9.0)
    out = parent[1:-1]
    result = run(torch.ones(4), out)
    assert result is out
    assert destinations == [out.data_ptr()]
    torch.testing.assert_close(out, torch.full((4,), 2.0))
    assert torch.all(parent[[0, -1]] == -9)


@pytest.mark.parametrize("query_rows_strided", [False, True])
@pytest.mark.parametrize("layout", ["contiguous", "rows", "features", "transposed"])
def test_shared_query_output_strides(query_rows_strided, layout):
    q = torch.arange(48.0).reshape(6, 2, 4)
    q = q[::2] if query_rows_strided else q[:3].contiguous()
    if layout == "rows":
        parent = torch.full((8, 2, 4), -99.0)
        out = parent[1:7:2]
    elif layout == "features":
        parent = torch.full((5, 2, 8), -99.0)
        out = parent[1:-1, :, ::2]
    elif layout == "transposed":
        parent = torch.full((5, 4, 2), -99.0)
        out = parent[1:-1].transpose(1, 2)
    else:
        parent = torch.full((5, 2, 4), -99.0)
        out = parent[1:-1]
    out.fill_(0)
    untouched = parent == -99
    seen = []

    def shared_strides(x, *, out=None):
        assert out is not None
        assert out.stride() == x.stride()
        seen.append(out.data_ptr())
        return out.copy_(x + 1)

    result = output_buffer.call_with_output(
        shared_strides, (q,), out, output_strides=q.stride()
    )
    assert result is out
    assert (seen[0] == out.data_ptr()) == (out.stride() == q.stride())
    torch.testing.assert_close(out, q + 1)
    assert torch.all(parent[untouched] == -99)


def test_contiguous_backend_stages_strided_output():
    parent = torch.full((8,), -9.0)
    out = parent[1:7:2]
    seen = []

    def contiguous_only(x, *, out=None):
        assert out is None or out.is_contiguous()
        seen.append(out)
        return x + 1 if out is None else out.copy_(x + 1)

    result = output_buffer.call_with_output(contiguous_only, (torch.ones(3),), out)
    assert result is out and seen == [None]
    torch.testing.assert_close(out, torch.full((3,), 2.0))
    assert torch.all(parent[[0, 2, 4, 6, 7]] == -9)


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("compiled", [False, True])
def test_independent_output_strides_preserve_api_compatibility(legacy, compiled):
    parent = torch.full((8,), -9.0)
    out = parent[1:7:2]
    seen = []

    def old(x):
        seen.append(None)
        return x + 1

    def current(x, *, out=None):
        assert out is not None, "strided output took the allocating fallback"
        seen.append(out)
        return out.copy_(x).add_(1)

    kernel = old if legacy else current

    def run(x, out):
        return output_buffer.call_with_output(
            kernel, (x,), out, supports_strided_output=True
        )

    if compiled:
        run = torch.compile(run, fullgraph=True, backend="eager")
    result = run(torch.ones(3), out)
    assert result is out and len(seen) == 1
    assert seen[0] is (None if legacy else out)
    torch.testing.assert_close(out, torch.full((3,), 2.0))
    assert torch.all(parent[[0, 2, 4, 6, 7]] == -9)
