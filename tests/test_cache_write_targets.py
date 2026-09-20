# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A capture must not write cache through a live batch's index buffers.

`capture_cudagraph` runs the model for real, so it writes cache wherever these
point, and `forward_vars` is not released on sleep. Where that landed decided the
two symptoms: outside a pool that came back smaller it was `Memory access fault`
during capture on 30B, and on live slots a decode row of all-NaN logprobs on 8B.

Which buffers a backend writes through is its own knowledge, so these pin the
`cache_write_targets` contract -- a builder's targets get blanked whether or not
they are in `forward_vars` -- rather than only the key-name rule behind the
default.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    from atom.model_ops.attentions.backends import (
        PAD_SLOT_ID,
        AttentionMetadataBuilder,
    )
except ImportError as _e:  # pragma: no cover - environment, not the code
    from import_guard import skip_if_dependency_missing

    skip_if_dependency_missing(_e, "the attention builders import AITER")


class _FakeBuffer:
    """A `CpuGpuBuffer` down to the two halves and the upload between them."""

    def __init__(self, size: int, fill: int = 0):
        self.np = np.full(size, fill, dtype=np.int64)
        self.gpu = np.full(size, fill, dtype=np.int64)
        self.uploads = 0

    def copy_to_gpu(self, n: int | None = None):
        self.uploads += 1
        self.gpu[:] = self.np
        return self.gpu


class _Builder:
    """The two methods under test, over a `forward_vars` a served step left.

    Not a subclass: the real methods are called unbound, so the test does not
    have to satisfy the rest of the abstract surface to reach them.
    """

    def __init__(self, extra_targets=()):
        self.model_runner = type("R", (), {})()
        self.model_runner.forward_vars = {
            "slot_mapping": _FakeBuffer(4, fill=30017),
            "ub0_slot_mapping": _FakeBuffer(2, fill=30021),
            "ub1_slot_mapping": _FakeBuffer(2, fill=30022),
            "context_lens": _FakeBuffer(4, fill=99),
            "block_tables": _FakeBuffer(4, fill=1873),
            # GDN registers bare tensors here too, so anything that scanned the
            # dict and duck-typed on `.np` would trip over them.
            "spec_sequence_masks": object(),
        }
        self._extra = list(extra_targets)

    def cache_write_targets(self):
        return AttentionMetadataBuilder.cache_write_targets(self) + self._extra

    def blank_cache_write_targets(self):
        return AttentionMetadataBuilder.blank_cache_write_targets(self)


def test_the_default_covers_the_base_buffer_and_every_ubatch_mirror():
    """TBO's capture reads per-ubatch copies rather than the base buffer."""
    b = _Builder()
    var = b.model_runner.forward_vars

    found = AttentionMetadataBuilder.cache_write_targets(b)

    assert {id(x) for x in found} == {
        id(var["slot_mapping"]),
        id(var["ub0_slot_mapping"]),
        id(var["ub1_slot_mapping"]),
    }


def test_the_default_leaves_the_read_side_alone():
    """The capture builders synthesize those; blanking them would cut across it."""
    b = _Builder()
    var = b.model_runner.forward_vars

    found = AttentionMetadataBuilder.cache_write_targets(b)

    assert var["context_lens"] not in found
    assert var["block_tables"] not in found


def test_a_capture_writes_nowhere():
    b = _Builder()

    b.blank_cache_write_targets()

    for name, buf in b.model_runner.forward_vars.items():
        if name.endswith("slot_mapping"):
            assert list(buf.np) == [PAD_SLOT_ID] * len(buf.np)


def test_the_device_half_is_what_a_capture_reads():
    """The builders hand the graph `.gpu`, so a host-only fill writes nothing."""
    b = _Builder()
    base = b.model_runner.forward_vars["slot_mapping"]

    b.blank_cache_write_targets()

    assert list(base.gpu) == [PAD_SLOT_ID] * 4
    assert base.uploads == 1


def test_a_target_outside_forward_vars_is_covered():
    """GDN's read fork is allocated on the builder and never registered there,
    yet its capture metadata hands the graph a live view of it."""
    unregistered = _FakeBuffer(4, fill=7)

    _Builder(extra_targets=[unregistered]).blank_cache_write_targets()

    assert list(unregistered.gpu) == [PAD_SLOT_ID] * 4


@pytest.mark.parametrize("stale", [0, 1, 30017, 2**31])
def test_no_row_survives_whatever_the_last_batch_held(stale):
    # 0 included on purpose: it is what a fresh `CpuGpuBuffer` holds, which is
    # why a startup capture writes row 0 rather than nothing.
    b = _Builder()
    b.model_runner.forward_vars["slot_mapping"].np[:] = stale

    b.blank_cache_write_targets()

    assert (b.model_runner.forward_vars["slot_mapping"].gpu < 0).all()
