# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Contract for the chunk-major staging plan cache's key.

The key decides whether a cached pack/unpack plan is reused. Reusing one for a
geometry it was not built for would pack from the wrong addresses, and building
a fresh one per staging group is the cost the cache exists to remove -- so both
halves of that trade need holding down: the key must separate geometries, and
it must not walk every segment to do it.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_MODULE_NAME = "atom.kv_transfer.offload.dense.triton_kv_staging"
_REAL_TRITON = importlib.util.find_spec("triton") is not None
_CPU_MODULE = None


def _staging_module():
    """The real wrapper source, loaded with an import-only Triton stub if needed.

    ``from __future__ import annotations`` in the wrapper keeps every
    ``tl.constexpr`` annotation a string, so nothing in the stub is consulted
    past ``@triton.jit`` accepting a function.
    """
    global _CPU_MODULE
    if _REAL_TRITON:
        return importlib.import_module(_MODULE_NAME)
    if _CPU_MODULE is not None:
        return _CPU_MODULE

    fake_triton = ModuleType("triton")
    fake_language = ModuleType("triton.language")
    fake_triton.__path__ = []
    fake_triton.language = fake_language
    fake_triton.jit = lambda function: function
    isolated_name = "_dense_triton_kv_staging_cpu_contract"
    source = (
        Path(__file__).parents[1]
        / "atom/kv_transfer/offload/dense/triton_kv_staging.py"
    )
    spec = importlib.util.spec_from_file_location(isolated_name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    missing = object()
    original_triton = sys.modules.get("triton", missing)
    original_language = sys.modules.get("triton.language", missing)
    sys.modules["triton"] = fake_triton
    sys.modules["triton.language"] = fake_language
    sys.modules[isolated_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        if original_triton is missing:
            sys.modules.pop("triton", None)
        else:
            sys.modules["triton"] = original_triton
        if original_language is missing:
            sys.modules.pop("triton.language", None)
        else:
            sys.modules["triton.language"] = original_language
    _CPU_MODULE = module
    return module


class _Segment:
    """Stands in for a KV segment tensor, counting pointer reads."""

    def __init__(self, address: int) -> None:
        self.address = address
        self.reads = 0

    def data_ptr(self) -> int:
        self.reads += 1
        return self.address


def _segments(addresses):
    return [_Segment(a) for a in addresses]


@pytest.fixture(scope="module")
def plan_key():
    return _staging_module()._plan_key


def test_same_geometry_and_shape_gives_the_same_key(plan_key):
    segments = _segments(range(1000, 1180))
    assert plan_key(segments, (1, 1, 1)) == plan_key(segments, (1, 1, 1))


def test_chunk_shape_separates_keys(plan_key):
    segments = _segments(range(1000, 1180))
    assert plan_key(segments, (1, 1, 1)) != plan_key(segments, (2, 1))
    # Same block total, different distribution across chunks.
    assert plan_key(segments, (2, 2)) != plan_key(segments, (1, 3))


def test_reallocated_segments_miss(plan_key):
    """A cache rebuilt at new addresses must not reuse the old plan."""
    before = plan_key(_segments(range(1000, 1180)), (1,))
    after = plan_key(_segments(range(9000, 9180)), (1,))
    assert before != after


def test_segment_count_separates_keys(plan_key):
    assert plan_key(_segments(range(1000, 1180)), (1,)) != plan_key(
        _segments(range(1000, 1179)), (1,)
    )


@pytest.mark.parametrize("moved", [0, 90, 179])
def test_a_moved_segment_separates_keys(plan_key, moved):
    """First, middle and last are the positions the key actually reads."""
    addresses = list(range(1000, 1180))
    before = plan_key(_segments(addresses), (1,))
    addresses[moved] += 4096
    assert plan_key(_segments(addresses), (1,)) != before


def test_key_is_constant_time(plan_key):
    """The whole point of the key: it must not walk every segment.

    Rebuilding it per staging group is what cost 0.277 ms under a contended
    GIL at this geometry. A change that reintroduces the walk still passes
    every separation test above, so pin the read count directly.
    """
    segments = _segments(range(1000, 1180))
    plan_key(segments, (1,))
    assert sum(s.reads for s in segments) <= 4
    assert len(segments) == 180


def test_empty_segment_list_has_no_key(plan_key):
    """None defers to _build_meta, which owns the error for this case."""
    assert plan_key([], (1,)) is None
