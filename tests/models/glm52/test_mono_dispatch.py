# SPDX-License-Identifier: MIT
"""GLM-5.2 mono: which steps are routed."""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from atom.models.glm52.mono.config import AGENTX_BATCHES, GRAPH_BATCHES
from tests.aiter_stub import stubbed_aiter


def test_only_batch_one_and_two_are_routed():
    # Review Focus 2: batches 3..96 build but must not be routed. If this ever
    # widens, the GSM8K gate at concurrency 1 stops covering what ships.
    assert AGENTX_BATCHES == (1, 2)
    assert set(AGENTX_BATCHES) < set(GRAPH_BATCHES)
    for batch in (3, 4, 8, 96):
        assert batch in GRAPH_BATCHES and batch not in AGENTX_BATCHES


SAMPLES = 16 * 5  # a padded cudagraph of 16 slots, MTP4 (query length 5)


class _Context:
    """The forward context's per-step flags, with the per-rank one poisoned."""

    is_prefill = False

    def __init__(self, scheduled_bs):
        self.scheduled_bs = scheduled_bs

    @property
    def is_dummy_run(self):
        raise AssertionError("routing read is_dummy_run, which differs per TP rank")


def _poison_rank_local_reads(monkeypatch, dispatch):
    def rank_local(*_args, **_kwargs):
        raise AssertionError("routing read a rank-local value")

    monkeypatch.setattr(torch.distributed, "get_rank", rank_local)
    monkeypatch.setattr(torch.cuda, "current_device", rank_local)
    # a name a routing module may later bind from the parallel state; the
    # functions' own globals, because the stub teardown unlists their modules
    for routed in (
        dispatch.MonoDecode.supports,
        dispatch.step_refusal,
        dispatch.kv_caches_ok,
        dispatch.native_config,
    ):
        monkeypatch.setitem(
            routed.__globals__, "get_tensor_model_parallel_rank", rank_local
        )


@pytest.mark.parametrize(
    ("batch", "query_length", "routed"),
    [(1, 5, True), (2, 5, True), (4, 5, False), (2, 2, False)],
)
def test_routing_reads_nothing_rank_local(monkeypatch, batch, query_length, routed):
    # Review Focus 3: a TP-divergent routing decision hangs the other ranks
    # in-kernel with no error and no log. This runs `supports` with the rank-local
    # reads poisoned, so one added in any file it calls fails here -- unlike a
    # search of dispatch.py's text, which cannot see another file.
    #
    # The runner reaches Triton and `supports` never touches it.
    runner = types.ModuleType("atom.models.glm52.mono.runner")
    runner.Glm52MonoRunner = object
    monkeypatch.setitem(sys.modules, runner.__name__, runner)
    with stubbed_aiter():
        dispatch = importlib.import_module("atom.models.glm52.mono.dispatch")
    _poison_rank_local_reads(monkeypatch, dispatch)

    decode = object.__new__(dispatch.MonoDecode)
    decode._atom_config = SimpleNamespace(
        tensor_parallel_size=4,
        kv_cache_dtype="fp8",
        speculative_config=SimpleNamespace(method="mtp", num_speculative_tokens=4),
        enable_dp_attention=False,
        decode_context_parallel_size=1,
    )
    decode._lm = SimpleNamespace(
        model=SimpleNamespace(
            aux_hidden_state_layers=[], layers=[], start_layer=0, end_layer=0
        )
    )
    ready = []
    decode._mono = SimpleNamespace(
        enabled=True, ready=lambda rows: ready.append(rows) or True
    )
    active = batch * query_length
    metadata = SimpleNamespace(
        max_seqlen_q=query_length,
        slot_mapping=torch.cat(
            (
                torch.arange(active, dtype=torch.int64),
                torch.full((SAMPLES - active,), -1, dtype=torch.int64),
            )
        ),
        sparse_kv_indptr=torch.cat(
            (
                torch.arange(active + 1, dtype=torch.int32),
                torch.full((SAMPLES - active,), active, dtype=torch.int32),
            )
        ),
    )
    context = _Context(batch)
    monkeypatch.setattr(
        dispatch,
        "get_forward_context",
        lambda: SimpleNamespace(
            context=context,
            attn_metadata=metadata,
            ubatch_slices=None,
            kv_cache_data={},
        ),
    )
    monkeypatch.setattr(
        dispatch,
        "shared_sparse_buffer",
        lambda _layers: torch.arange(SAMPLES, dtype=torch.int32),
    )

    assert (
        decode.supports(
            torch.arange(SAMPLES), torch.arange(SAMPLES, dtype=torch.int64), None, None
        )
        is routed
    )
    assert ready == ([SAMPLES] if routed else [])
