# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM_PROBE_MEM: per-step activation highs against the startup profile."""

import ast
import logging
import pathlib
from types import SimpleNamespace

import pytest
import torch

from atom.utils import mem_probe
from atom.utils.mem_probe import MemProbe

_GiB = 1 << 30
_ROOT = pathlib.Path(__file__).resolve().parents[1]


class _FakeCuda:
    """The allocator counters the probe reads, driven by the test."""

    def __init__(self):
        self.allocated = 100 * _GiB
        self.step_peak = self.allocated
        self.resets = 0

    def install(self, monkeypatch):
        cuda = torch.cuda
        monkeypatch.setattr(cuda, "memory_allocated", lambda *a: self.allocated)
        monkeypatch.setattr(cuda, "max_memory_allocated", lambda *a: self.step_peak)
        monkeypatch.setattr(cuda, "reset_peak_memory_stats", self._reset)
        monkeypatch.setattr(
            cuda,
            "memory_stats",
            lambda *a: {
                "reserved_bytes.all.current": 120 * _GiB,
                "allocated_bytes.all.current": self.allocated,
                "num_alloc_retries": 2,
            },
        )
        monkeypatch.setattr(cuda, "mem_get_info", lambda *a: (14 * _GiB, 288 * _GiB))
        monkeypatch.setattr(cuda, "memory_summary", lambda *a, **k: "SUMMARY-TABLE")

    def _reset(self, *a):
        self.resets += 1
        self.step_peak = self.allocated


def _batch(kind="mixed", p_tok=15000, d_tok=600):
    return SimpleNamespace(
        is_dummy_run=kind == "dummy",
        is_mixed=kind == "mixed",
        total_seqs_num_prefill=0 if kind == "decode" else 3,
        total_tokens_num_prefill=0 if kind == "decode" else p_tok,
        total_seqs_num_decode=0 if kind == "prefill" else 75,
        total_tokens_num_decode=0 if kind == "prefill" else d_tok,
    )


@pytest.fixture
def cuda(monkeypatch):
    fake = _FakeCuda()
    fake.install(monkeypatch)
    return fake


@pytest.fixture
def probe(monkeypatch):
    monkeypatch.setenv("ATOM_PROBE_MEM", "1")
    return MemProbe("DP3TP0")


def _arm(probe, profile_activations=4 * _GiB):
    probe.record_warmup(2, 16384)
    probe.record_budget(
        peak=100 * _GiB + profile_activations,
        current=100 * _GiB,
        non_torch=15 * _GiB,
        budget=259 * _GiB,
        kv_bytes=91 * _GiB,
    )


def _step(probe, cuda, batch, activations):
    def forward(b):
        cuda.step_peak = cuda.allocated + activations
        return "out"

    return probe.wrap(forward)(batch)


def test_off_leaves_forward_untouched(monkeypatch, cuda):
    monkeypatch.setenv("ATOM_PROBE_MEM", "0")
    probe = MemProbe("DP0TP0")

    def forward(b):
        return "out"

    assert probe.wrap(forward) is forward
    _arm(probe)
    probe.step_begin()
    assert cuda.resets == 0


def test_it_does_not_reset_the_peak_before_the_budget_reads_it(probe, cuda):
    """Resetting during the profile would change the KV size it explains."""
    _step(probe, cuda, _batch("dummy", 16384, 0), 4 * _GiB)
    assert cuda.resets == 0


def test_profile_line_and_new_highs(probe, cuda, caplog):
    with caplog.at_level(logging.WARNING, logger="atom"):
        _arm(probe)
        assert _step(probe, cuda, _batch("mixed"), 10 * _GiB) == "out"
        _step(probe, cuda, _batch("mixed"), 10 * _GiB + (100 << 20))  # within margin
        _step(probe, cuda, _batch("mixed"), 11 * _GiB)
        _step(probe, cuda, _batch("decode"), 1 * _GiB)
    text = caplog.text
    assert "PROBE_MEM DP3TP0 profile: warmup 2 seq(s) x 16384 token(s)" in text
    assert "activations 4.00 GiB" in text
    assert text.count("new mixed high") == 2
    assert "activations 10.00 GiB (2.50x profile)" in text
    assert "prefill 3 seq / 15000 tok, decode 75 seq / 600 tok" in text
    assert "new decode high" in text
    assert cuda.resets == 4


def test_periodic_summary(probe, cuda, caplog, monkeypatch):
    monkeypatch.setattr(mem_probe, "_SUMMARY_EVERY", 3)
    with caplog.at_level(logging.WARNING, logger="atom"):
        _arm(probe)
        for _ in range(3):
            _step(probe, cuda, _batch("decode"), 0)
    assert "PROBE_MEM DP3TP0 step 3: reserved 120.00" in caplog.text
    assert "alloc_retries 2" in caplog.text


def test_oom_reports_the_allocator_and_reraises(probe, cuda, caplog):
    def forward(b):
        raise torch.OutOfMemoryError("Tried to allocate 5.91 GiB")

    with (
        caplog.at_level(logging.ERROR, logger="atom"),
        pytest.raises(torch.OutOfMemoryError),
    ):
        probe.wrap(forward)(_batch("mixed"))
    assert "PROBE_MEM DP3TP0 OOM" in caplog.text
    assert "device_free 14.00 GiB" in caplog.text
    assert "SUMMARY-TABLE" in caplog.text


def test_the_runner_arms_it_after_reading_the_profile_peak():
    """Source-level: the runner module needs aiter to import."""
    src = (_ROOT / "atom/model_engine/model_runner.py").read_text(encoding="utf-8")
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name == "get_num_blocks":
            body = ast.unparse(node)
            if "record_budget" not in body:
                continue
            assert body.index("allocated_bytes.all.peak") < body.index("record_budget")
            assert body.index("_estimate_cudagraph_overhead()") < body.index(
                "record_budget"
            )
            return
    pytest.fail("get_num_blocks never calls record_budget")
