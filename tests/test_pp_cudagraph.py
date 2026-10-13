# SPDX-License-Identifier: MIT
"""FULL cudagraphs under pipeline parallelism: what each stage's graphs read."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", reason="needs the AITER GPU kernel library")

from atom.config import CUDAGraphMode
from atom.model_engine import model_runner as mr
from atom.model_engine.model_runner import ModelRunner
from atom.models.utils import IntermediateTensors

K3 = pytest.importorskip(
    "atom.model_ops.attentions.kimi_mla_gdn_attn",
    reason="needs the hybrid KDA/MLA backend",
    exc_type=ImportError,
)


def _runner(**attrs) -> ModelRunner:
    runner = ModelRunner.__new__(ModelRunner)
    runner.device = torch.device("cpu")
    runner._pp_recv_layout = None
    runner._pp_graph_inputs = None
    for name, value in attrs.items():
        setattr(runner, name, value)
    return runner


@pytest.mark.parametrize(
    "forks, eager",
    [((), False), ((-1, -1), False), ((-1, 7), True), (None, False)],
)
def test_a_forked_decode_runs_eagerly(forks, eager):
    builder = K3.KimiAiterMLAGDNMetadataBuilder.__new__(
        K3.KimiAiterMLAGDNMetadataBuilder
    )
    batch = (
        SimpleNamespace() if forks is None else SimpleNamespace(state_fork_srcs=forks)
    )
    assert builder.step_needs_eager(batch) is eager


def test_the_ring_refuses_piecewise_replay_under_pp():
    runner = _runner(
        enforce_eager=False,
        config=SimpleNamespace(
            pipeline_parallel_size=2,
            compilation_config=SimpleNamespace(cudagraph_mode=CUDAGraphMode.PIECEWISE),
        ),
    )
    with pytest.raises(NotImplementedError, match="PIECEWISE"):
        runner._init_forward_vars_ring()


def test_receive_buffers_take_the_layout_the_warmup_received():
    layout = {
        "hidden_states": (torch.Size([16]), torch.bfloat16),
        "block_residual": (torch.Size([3, 16]), torch.bfloat16),
    }
    buffers = _runner(_pp_recv_layout=layout)._alloc_pp_graph_inputs(8)
    assert buffers["hidden_states"].shape == (8, 16)
    assert buffers["block_residual"].shape == (8, 3, 16)
    assert all(t.dtype == torch.bfloat16 for t in buffers.values())


def test_capture_without_a_warmup_receive_says_so():
    with pytest.raises(RuntimeError, match="warmup"):
        _runner()._alloc_pp_graph_inputs(8)


def _staged(monkeypatch, sent: dict[str, torch.Tensor]) -> ModelRunner:
    runner = _runner(
        _pp_graph_inputs={
            "hidden_states": torch.zeros(8, 4),
            "block_residual": torch.zeros(8, 2, 4),
        }
    )
    monkeypatch.setattr(
        mr, "recv_intermediate_tensors", lambda: IntermediateTensors(sent)
    )
    return runner


def test_a_replay_reads_the_rows_just_received(monkeypatch):
    sent = {
        "hidden_states": torch.full((5, 4), 3.0),
        "block_residual": torch.full((5, 2, 4), 4.0),
    }
    runner = _staged(monkeypatch, sent)
    runner._pp_graph_inputs["hidden_states"].fill_(-1.0)

    runner._recv_pp_graph_inputs(5)

    hidden = runner._pp_graph_inputs["hidden_states"]
    assert torch.all(hidden[:5] == 3.0) and torch.all(hidden[5:] == -1.0)
    assert torch.all(runner._pp_graph_inputs["block_residual"][:5] == 4.0)


def test_a_receive_that_does_not_fit_the_graph_is_refused(monkeypatch):
    short = {
        "hidden_states": torch.ones(4, 4),
        "block_residual": torch.ones(4, 2, 4),
    }
    with pytest.raises(RuntimeError, match="replays 5"):
        _staged(monkeypatch, short)._recv_pp_graph_inputs(5)
    missing = _staged(monkeypatch, {"hidden_states": torch.ones(5, 4)})
    with pytest.raises(RuntimeError, match="captured reading"):
        missing._recv_pp_graph_inputs(5)
