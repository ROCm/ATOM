# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for intra-GPU disagg constrained vs unconstrained modes.

Only the scheduler-level shm gating is exercised here; the IPC handshake
and CUDA stream pool are out of scope for the no-GPU test environment.
"""

import pytest
from conftest import MockConfig


@pytest.fixture
def prefill_scheduler_unconstrained():
    from atom.model_engine.scheduler import PrefillScheduler

    return PrefillScheduler(MockConfig(), disagg_cu_shm_name="")


@pytest.fixture
def decode_scheduler_unconstrained():
    from atom.model_engine.scheduler import DecodeScheduler

    return DecodeScheduler(MockConfig(), disagg_cu_shm_name="")


@pytest.fixture
def seq_factory():
    from atom.sampling_params import SamplingParams
    from atom.model_engine.sequence import Sequence

    def make(token_ids, block_size=4):
        return Sequence(token_ids, block_size, sampling_params=SamplingParams())

    return make


# ── Unconstrained: no shm handle attached ────────────────────────────────


def test_prefill_scheduler_skips_shm_when_name_empty(prefill_scheduler_unconstrained):
    assert prefill_scheduler_unconstrained._cu_shm is None


def test_decode_scheduler_skips_shm_when_name_empty(decode_scheduler_unconstrained):
    assert decode_scheduler_unconstrained._cu_shm is None


def test_decode_handoff_builds_full_mtp_window(seq_factory):
    class SpecConfig:
        num_speculative_tokens = 3

        @staticmethod
        def use_dspark():
            return True

    from atom.model_engine.scheduler import DecodeScheduler

    scheduler = DecodeScheduler(
        MockConfig(speculative_config=SpecConfig()), disagg_cu_shm_name=""
    )
    prompt = [11, 22, 33, 44, 55]
    t0 = 14
    seq = seq_factory(prompt)
    scheduler.block_manager.allocate(seq)
    scheduler.prefill_waiting[seq.id] = seq

    scheduler.on_prefill_done(seq.id, len(prompt), t0)

    assert seq.is_first_decode is True
    assert seq.token_ids[-4:] == [t0, 2, 2, 2]
    assert not set(seq.token_ids[-4:]) & set(prompt)

    batch, _ = scheduler.schedule()
    assert batch.is_first_decode_without_local_prefill == [True]
    assert seq.is_first_decode is False


# ── Unconstrained: batches carry cu_stream_fraction=None ─────────────────


def test_unconstrained_prefill_batch_has_none_cu_fraction(
    prefill_scheduler_unconstrained, seq_factory
):
    """Without shm, PrefillScheduler must produce batches keyed by the
    plain (None) stream — never a fractional CU mask."""
    seq = seq_factory([10, 20, 30, 40])
    seq.block_table = [0, 1]
    seq.num_cached_tokens = 0
    prefill_scheduler_unconstrained.add(seq)

    batch, _ = prefill_scheduler_unconstrained.schedule()
    assert batch is not None
    assert batch.cu_stream_fraction is None
