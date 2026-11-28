# SPDX-License-Identifier: MIT
"""Under DP with TBO, the startup profile also measures the warmup width
unsplit: a DP group runs it so whenever one rank decodes beside its peers'
prefill, and the MoE then gathers twice a micro-batch's rows."""

from types import SimpleNamespace

import numpy as np
import pytest
from aiter_stub import atom_without_aiter

from atom.utils.tbo.ubatching import sync_dp_metadata

runner_module = None  # bound by `_runner_module` for the whole file


@pytest.fixture(scope="module", autouse=True)
def _runner_module():
    """The real runner module; AITER and Triton are stood in for on a CPU job."""
    global runner_module
    with atom_without_aiter() as load:
        runner_module = load("atom.model_engine.model_runner")
        yield runner_module


def _runner(monkeypatch, *, dp_size, enable_tbo, warmup_tokens=16384):
    monkeypatch.setattr(runner_module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        runner_module.torch.cuda, "reset_peak_memory_stats", lambda: None
    )
    monkeypatch.setattr(
        runner_module, "get_dp_group", lambda: SimpleNamespace(world_size=dp_size)
    )
    monkeypatch.setenv("ATOM_TBO_PREFILL_MIN_TOKENS", "8192")
    runner = object.__new__(runner_module.ModelRunner)
    runner.config = SimpleNamespace(
        max_model_len=warmup_tokens,
        max_num_seqs=256,
        enable_dp_attention=True,
        enable_tbo=enable_tbo,
        prefill_context_parallel_size=1,
        speculative_config=None,
    )
    runner.block_size, runner.label = 16, "test"
    runner.tokenID_processor = SimpleNamespace(clean=lambda: None)
    runner.warmup_max_tokens = lambda: warmup_tokens
    forwards = []

    def forward(batch):
        # The group's decision from this rank's vote, every rank casting the
        # same one: each warms up the same batch.
        meets_min, can_split, ub0, ub1 = runner._local_tbo_eligibility(batch)
        sync = sync_dp_metadata(
            dp_group=None,
            dp_size=1,
            scheduled_tokens=batch.total_tokens_num,
            scheduled_bs=batch.total_seqs_num,
            is_prefill=batch.has_prefill,
            tbo_on=runner.config.enable_tbo,
            running_tokens_per_dp_rank=np.zeros(1, np.int32),
            local_meets_min_tokens=meets_min,
            local_can_split=can_split,
            local_ub_tokens=(ub0, ub1),
        )
        forwards.append((batch.total_tokens_num, sync.tbo_collective_active))

    runner.forward = forward
    return runner, forwards


def test_a_dp_group_profiles_its_warmup_split_and_unsplit(monkeypatch):
    """The split forward the warmup always ran, then the same width unsplit:
    the allocator's peak `get_num_blocks` reads is the larger of the two."""
    runner, forwards = _runner(monkeypatch, dp_size=8, enable_tbo=True)

    runner.warmup_model()

    assert forwards == [(16384, True), (16384, False)]
    assert not runner._warmup_unsplit


def test_the_unsplit_veto_is_lifted_when_its_forward_raises(monkeypatch):
    runner, forwards = _runner(monkeypatch, dp_size=8, enable_tbo=True)
    split_forward = runner.forward

    def forward(batch):
        split_forward(batch)
        if len(forwards) == 2:
            raise RuntimeError("out of memory")

    runner.forward = forward
    with pytest.raises(RuntimeError, match="out of memory"):
        runner.warmup_model()
    assert not runner._warmup_unsplit


@pytest.mark.parametrize(
    "dp_size,enable_tbo,warmup_tokens,expected",
    [
        # A group of one splits every prefill past the bar: it never runs
        # this width unsplit.
        (1, True, 16384, [(16384, True)]),
        # Below the bar the warmup already ran unsplit.
        (8, True, 4096, [(4096, False)]),
        (8, False, 16384, [(16384, False)]),
    ],
)
def test_one_warmup_forward_when_serving_never_runs_it_unsplit(
    monkeypatch, dp_size, enable_tbo, warmup_tokens, expected
):
    runner, forwards = _runner(
        monkeypatch,
        dp_size=dp_size,
        enable_tbo=enable_tbo,
        warmup_tokens=warmup_tokens,
    )

    runner.warmup_model()

    assert forwards == expected
