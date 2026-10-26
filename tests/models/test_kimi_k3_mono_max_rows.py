# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Kimi-K3 mono at 16 rows.

A bs=2 x MTP7 verify is 16 rows. Under a cap of 8 every such step, and the
bs=2 CUDA graph's warmup and capture, would be declined back to the production
path; at the shipped cap of 16 they run on the kernels. These tests pin the
plain-Python parts: what the routing wrapper decides at each cap (the cap is
patched here), that the buffers follow the cap, and the kernels' own arithmetic for the wider build. Whether
the 16-row kernels are right and faster on a GPU is not something they can say.
"""

import logging
from itertools import pairwise
from types import SimpleNamespace

import pytest
import torch

from atom.models.kimi_k3.mono import config as k3_config
from atom.models.kimi_k3.mono import plan


def _metadata(requests: int, qlen: int):
    return SimpleNamespace(
        spec_sequence_masks=object(),
        num_prefills=0,
        num_decodes=0,
        num_spec_decodes=requests,
        num_actual_tokens=requests * qlen,
        spec_state_indices_tensor=torch.zeros(requests, qlen, dtype=torch.int32),
        num_accepted_tokens=torch.ones(requests, dtype=torch.int32),
        replayssm=False,
    )


def _wrapper(monkeypatch, cap: int, requests: int, qlen: int):
    """The routing wrapper over a stub model, in a decode step of that shape."""
    from atom.models.kimi_k3.mono import dispatch

    monkeypatch.setattr(plan, "MAX_ROWS", cap)
    context = SimpleNamespace(
        context=SimpleNamespace(is_prefill=False),
        ubatch_slices=None,
        attn_metadata=SimpleNamespace(kda_metadata=_metadata(requests, qlen)),
    )
    monkeypatch.setattr(dispatch, "get_forward_context", lambda: context)
    return dispatch.MonoDecodeModel(torch.nn.Identity()), torch.zeros(requests * qlen)


def test_the_bs2_step_is_routed_only_at_cap_16(monkeypatch):
    for cap, qlen_served in ((8, None), (16, 8)):
        wrapper, ids = _wrapper(monkeypatch, cap, requests=2, qlen=8)
        assert wrapper._qlen(ids, None, None) == qlen_served


def test_a_declined_width_is_logged_once_per_distinct_row_count(monkeypatch, caplog):
    """So N log lines name N different row counts, not N declined steps."""
    wrapper, ids = _wrapper(monkeypatch, 8, requests=2, qlen=8)
    with caplog.at_level(logging.INFO, logger="atom"):
        for _ in range(5):
            wrapper._qlen(ids, None, None)
    lines = [
        r.getMessage() for r in caplog.records if "declined a step" in r.getMessage()
    ]
    assert lines == ["Kimi-K3 mono decode declined a step: 16 rows > the 8-row buffers"]


# ---------------------------------------------------------------- the runner
@pytest.fixture
def runner_module():
    pytest.importorskip("flydsl", reason="importing the K3 mono runner needs FlyDSL")
    pytest.importorskip("aiter", reason="the K3 mono runner imports aiter.dist")
    from atom.models.kimi_k3.mono import runner

    return runner


def test_the_runner_s_buffers_follow_the_cap(runner_module):
    """``max_rows`` sizes the peer buffer and the scratch; both read the cap the
    config was imported with."""
    assert runner_module.K3MonoDecodeRunner.max_rows == k3_config.MAX_ROWS


@pytest.mark.parametrize(
    ("max_rows", "rows", "qlen", "prepared"),
    [
        (8, 8, 8, True),
        (8, 16, 8, False),  # past the cap: no build is even attempted
        (16, 16, 8, True),
        (16, 16, 16, True),
        (16, 24, 8, False),
        (16, 9, 4, False),  # not whole requests
        (16, 0, 1, False),
    ],
)
def test_prepare_builds_a_width_only_inside_the_cap(
    runner_module, max_rows, rows, qlen, prepared
):
    built = []
    runner = object.__new__(runner_module.K3MonoDecodeRunner)
    runner.max_rows = max_rows
    runner.widths = SimpleNamespace(prepare=lambda key: built.append(key) or True)
    assert runner.prepare(rows, qlen) is prepared
    assert built == ([plan.width_key(rows, qlen)] if prepared else [])


def test_the_peer_buffer_doubles_from_8_to_16_rows(runner_module):
    """The buffer is sized for the cap and zeroed whole at every step's start
    (``StepMailboxes.begin_step``), so a cap of 16 also costs the 8-row steps."""
    from atom.models.kimi_k3.mono.kernels import layer as k2

    assert k2.peer_bytes(16) == 2 * k2.peer_bytes(8)


def test_a_16_row_build_stages_the_x_row_in_slabs(runner_module):
    """One slab at 8 rows is the build that was always there; 16 rows' whole
    7168-wide row does not fit gfx950's LDS, so the GEMV walks the column groups
    in order through several slabs, each fitting."""
    from atom.models.kimi_k3.mono.kernels.attention import kda

    assert kda.slab_plan(8) == [(0, kda.PER_WAVE)]
    wide = kda.slab_plan(16)
    assert len(wide) > 1
    assert wide[0][0] == 0
    for (first, take), (nxt, _) in pairwise(wide):
        assert first + take == nxt
    assert sum(take for _, take in wide) == kda.PER_WAVE
    for _, take in wide:
        assert 16 * (take * kda.KGROUP + kda.LDS_PAD) * 2 + kda.LDS_FIXED <= (
            kda.LDS_LIMIT
        )
