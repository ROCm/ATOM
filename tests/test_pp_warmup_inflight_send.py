# SPDX-License-Identifier: MIT
"""Under PP the warmup that sizes the KV cache must see a stage's previous
send still in flight, as every serving forward does."""

from types import SimpleNamespace

import pytest
from aiter_stub import atom_without_aiter


@pytest.fixture(scope="module")
def runner_module():
    with atom_without_aiter() as load:
        yield load("atom.model_engine.model_runner")


class _PPStageMemory:
    """A PP stage's allocations as `run_model` makes them: the forward's
    activations, then the commit of the previous step's send and the post of
    this one's, whose payload stays live until the next forward sends."""

    ACTIVATIONS, PAYLOAD = 10, 3

    def __init__(self):
        self.live = {}
        self.peak = 0

    def _alloc(self, name, nbytes):
        self.live[name] = nbytes
        self.peak = max(self.peak, sum(self.live.values()))

    def forward(self, batch):
        self._alloc("activations", self.ACTIVATIONS)
        del self.live["activations"]
        self.live.pop("send", None)  # commit_pp_send_work(previous)
        self._alloc("send", self.PAYLOAD)  # async_send_intermediate_tensors


@pytest.mark.parametrize("pp_size", [1, 4])
def test_warmup_peak_covers_the_pp_send_in_flight(monkeypatch, runner_module, pp_size):
    """The warmup peak sizes the KV cache, so it must reach the serving peak:
    under PP the previous step's send is still in flight during a forward."""
    runner = runner_module.ModelRunner.__new__(runner_module.ModelRunner)
    runner.config = SimpleNamespace(
        max_model_len=1 << 20,
        max_num_seqs=32,
        enable_dp_attention=False,
        pipeline_parallel_size=pp_size,
        speculative_config=None,
    )
    runner.block_size = 128
    runner.label = "test"
    runner.tokenID_processor = SimpleNamespace(clean=lambda: None)
    runner.warmup_max_tokens = lambda: 20352
    memory = _PPStageMemory()
    batches = []
    runner.forward = lambda batch: batches.append(batch) or memory.forward(batch)
    monkeypatch.setattr(
        runner_module, "get_dp_group", lambda: SimpleNamespace(world_size=1)
    )
    monkeypatch.setattr(runner_module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        runner_module.torch.cuda, "reset_peak_memory_stats", lambda: None
    )

    runner.warmup_model()

    serving = _PPStageMemory()
    for _ in range(3):
        serving.forward(None)
    if pp_size == 1:
        assert len(batches) == 1
    else:
        assert (
            memory.peak
            == serving.peak
            == (_PPStageMemory.ACTIVATIONS + _PPStageMemory.PAYLOAD)
        )
        # Every stage runs the same forwards, so PP sends and receives pair up;
        # each forward gets fresh dummy sequences.
        assert len(batches) == 2 and batches[0] is not batches[1]
        assert batches[0].total_tokens_num == batches[1].total_tokens_num == 20352
