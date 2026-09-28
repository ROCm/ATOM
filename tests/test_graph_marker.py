# SPDX-License-Identifier: MIT
"""Graph markers must preserve aliases without adding activation copies."""

import runpy
import warnings

import pytest
import torch
from torch._inductor.utils import run_and_get_code

from atom.utils.graph_marker import (
    graph_marker,
    is_graph_marker_enabled,
    set_graph_marker_enabled,
)
from atom.utils.graph_marker_instrumentation import instrument_record_functions_in_file


@torch.library.custom_op("atom_test::marker_consumer", mutates_args=())
def _consume(x: torch.Tensor) -> torch.Tensor:
    return x * 2


@_consume.register_fake
def _consume_fake(x: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(x)


@pytest.fixture(autouse=True)
def restore_marker_setting():
    previous = is_graph_marker_enabled()
    set_graph_marker_enabled(True)
    yield
    set_graph_marker_enabled(previous)
    torch._dynamo.reset()


@pytest.mark.parametrize("enabled", [False, True])
def test_marker_preserves_identity_and_views(enabled):
    set_graph_marker_enabled(enabled)
    base = torch.arange(32.0)
    x = base[1::2]
    y = graph_marker(x, "view_start")
    assert y is x
    y.add_(1)
    torch.testing.assert_close(base[1::2], torch.arange(1.0, 32, 2) + 1)


@pytest.mark.parametrize("strided", [False, True])
def test_compiled_marker_has_no_copy_and_retains_profile_ranges(tmp_path, strided):
    def fn(x):
        x = graph_marker(x, "mhc_start")
        y = _consume(x)
        return graph_marker(y, "mhc_end")

    x = torch.arange(32.0)
    if strided:
        x = x[1::2]
    before = x.clone()
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*may not alias.*")
        out, codes = run_and_get_code(torch.compile(fn, fullgraph=True), x)
    torch.testing.assert_close(out, before * 2)
    torch.testing.assert_close(x, before)
    code = "\n".join(codes)
    # Only the two marker barriers and the consumer should be emitted. Any
    # compiled pointwise kernel here is an unnecessary clone/copy/layout fixup.
    assert "async_compile.cpp" not in code
    assert "async_compile.triton" not in code
    assert "torch.ops.aiter.graph_marker.default(" in code
    assert "mhc_start" in code and "mhc_end" in code

    wrapper = tmp_path / "compiled.py"
    wrapper.write_text(code)
    assert instrument_record_functions_in_file(str(wrapper))
    instrumented = wrapper.read_text()
    assert 'with record_function("mhc"):' in instrumented
    assert "torch.ops.aiter.graph_marker.default(" not in instrumented
    compile(instrumented, str(wrapper), "exec")
    assert not instrument_record_functions_in_file(str(wrapper))


@pytest.mark.parametrize("strided", [False, True])
def test_compiled_marker_preserves_inplace_aliases(strided):
    def fn(x):
        y = x[1::2] if strided else x
        y = graph_marker(y, "mutation_start")
        y.add_(1)
        return x, y

    x = torch.arange(32.0)
    reference_input = x.clone()
    expected = fn(reference_input)
    actual = torch.compile(fn, fullgraph=True)(x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(x, reference_input)
    assert (
        actual[0].untyped_storage().data_ptr() == actual[1].untyped_storage().data_ptr()
    )


def test_instrumentation_accepts_legacy_marker_assignments(tmp_path):
    wrapper = tmp_path / "legacy.py"
    wrapper.write_text(
        "def call(x):\n"
        "    y = torch.ops.aiter.graph_marker.default(x, 'mhc_start')\n"
        "    z = y * 2\n"
        "    out = torch.ops.aiter.graph_marker.default(z, 'mhc_end')\n"
        "    return out\n"
    )
    assert instrument_record_functions_in_file(str(wrapper))
    code = wrapper.read_text()
    assert "y = x" in code and "out = z" in code
    namespace = runpy.run_path(str(wrapper))
    assert namespace["call"](3) == 6
