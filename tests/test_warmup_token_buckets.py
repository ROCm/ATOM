# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`ModelRunner.warmup_model` opens every power-of-two prefill token bucket at
startup, and runs each distinct linear GEMM at those counts and between them,
so no request's first forward pays a kernel's first-use host work (aiter's
gfx950 mxfp8 GEMM JIT-compiles the tile its row count picks)."""

import sys
from types import SimpleNamespace

import pytest
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def model_runner():
    with atom_without_aiter() as load:
        yield load("atom.model_engine.model_runner")


def _runner(model_runner, monkeypatch, *, warmup_tokens, dspark=False):
    """A stand-in `self` for `warmup_model` that records each forward's batch."""
    monkeypatch.setattr(
        model_runner, "get_dp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(model_runner.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        model_runner.torch.cuda, "reset_peak_memory_stats", lambda: None
    )
    forwards = []
    spec = SimpleNamespace(use_dspark=lambda: True) if dspark else None
    runner = SimpleNamespace(
        config=SimpleNamespace(
            max_model_len=32768,
            max_num_seqs=256,
            speculative_config=spec,
            enable_dp_attention=False,
        ),
        block_size=16,
        label="test",
        warmup_max_tokens=lambda: warmup_tokens,
        forward=lambda batch: forwards.append(
            (batch.total_seqs_num, batch.total_tokens_num, batch.is_dummy_run)
        ),
        tokenID_processor=SimpleNamespace(clean=lambda: None),
    )
    runner._warmup_prefill_batch = (
        lambda n, length: model_runner.ModelRunner._warmup_prefill_batch(
            runner, n, length
        )
    )
    runner.warmed_gemms = []
    runner._warm_linear_gemms = lambda counts: runner.warmed_gemms.append(counts) or 0
    return runner, forwards


def test_buckets_are_the_powers_of_two_below_the_warmed_count(
    model_runner, monkeypatch
):
    monkeypatch.setenv("ATOM_WARMUP_PREFILL_BUCKETS", "1")
    buckets = model_runner.warmup_token_buckets
    assert buckets(16384) == [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    assert buckets(3000) == [16, 32, 64, 128, 256, 512, 1024, 2048]
    assert buckets(16) == []
    assert buckets(100, min_seqs=2) == [16, 32, 64]
    gemm = model_runner.warmup_gemm_token_counts
    assert gemm(64, [16, 32]) == [16, 20, 24, 28, 32, 40, 48, 56, 64]


def test_warmup_runs_one_forward_per_bucket_after_the_first(model_runner, monkeypatch):
    monkeypatch.setenv("ATOM_WARMUP_PREFILL_BUCKETS", "1")
    runner, forwards = _runner(model_runner, monkeypatch, warmup_tokens=4096)

    model_runner.ModelRunner.warmup_model(runner)

    assert forwards[0] == (1, 4096, True)  # the peak-sizing forward, unchanged
    assert forwards[1:] == [(1, n, True) for n in (16, 32, 64, 128, 256, 512)] + [
        (1, n, True) for n in (1024, 2048)
    ]
    # The linear GEMMs a dummy forward skips run at every bucket, the peak, and
    # 5/8 to 7/8 of each.
    (counts,) = runner.warmed_gemms
    assert {16, 32, 64, 128, 256, 512, 1024, 2048, 4096} <= set(counts)
    assert {1280, 1536, 1792, 2560, 3072, 3584} <= set(counts)
    assert counts == sorted(counts) and counts[0] == 16 and counts[-1] == 4096


def test_a_dspark_warmup_keeps_two_sequences_per_bucket(model_runner, monkeypatch):
    monkeypatch.setenv("ATOM_WARMUP_PREFILL_BUCKETS", "1")
    runner, forwards = _runner(
        model_runner, monkeypatch, warmup_tokens=128, dspark=True
    )

    model_runner.ModelRunner.warmup_model(runner)

    assert forwards == [(2, 128, True), (2, 16, True), (2, 32, True), (2, 64, True)]


@pytest.mark.parametrize("value", [None, "0"])
def test_default_and_zero_leave_the_single_warmup_forward(
    model_runner, monkeypatch, value
):
    if value is None:
        monkeypatch.delenv("ATOM_WARMUP_PREFILL_BUCKETS", raising=False)
    else:
        monkeypatch.setenv("ATOM_WARMUP_PREFILL_BUCKETS", value)
    runner, forwards = _runner(model_runner, monkeypatch, warmup_tokens=4096)

    model_runner.ModelRunner.warmup_model(runner)

    assert forwards == [(1, 4096, True)]
    assert runner.warmed_gemms == []
    assert model_runner.warmup_token_buckets(4096) == []


def test_each_distinct_linear_runs_once_per_count_without_its_reduction(
    model_runner, monkeypatch
):
    torch = model_runner.torch
    linear = sys.modules["atom.model_ops.linear"]

    class FakeLinear(torch.nn.Module):
        def __init__(self, n, k, reduce=True):
            super().__init__()
            self.weight = torch.zeros(n, k)
            self.quant_type, self.input_size = "q", k
            self.reduce_results, self.prefix, self.calls = reduce, f"l{n}x{k}", []

        def forward(self, x):
            self.calls.append((tuple(x.shape), x.dtype, self.reduce_results))
            return x

    class Broken(FakeLinear):
        def forward(self, x):
            raise RuntimeError("needs a pre-quantized input")

    monkeypatch.setattr(linear, "LinearBase", FakeLinear)
    layers = [FakeLinear(8, 4), FakeLinear(8, 4), FakeLinear(16, 4, reduce=False)]
    broken = Broken(32, 4)
    runner = SimpleNamespace(
        model=torch.nn.Sequential(*layers, broken, torch.nn.Linear(4, 4)),
        config=SimpleNamespace(torch_dtype=torch.bfloat16),
        device="cpu",
    )

    assert model_runner.ModelRunner._warm_linear_gemms.__wrapped__(runner, [2, 6]) == 3
    want = [((2, 4), torch.bfloat16, False), ((6, 4), torch.bfloat16, False)]
    assert layers[0].calls == want and layers[2].calls == want
    assert layers[1].calls == []  # same class, quantization and shape: shares
    assert layers[0].reduce_results is True and layers[2].reduce_results is False
    assert broken.reduce_results is True  # restored after the failed warm
