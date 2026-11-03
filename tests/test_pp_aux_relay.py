# SPDX-License-Identifier: MIT
"""DSpark on a PP prefill, where only the last stage builds the drafter.

Its tapped target layers are carried across the stages: three stages over real
hooks and a real payload dict, and the last stage must receive exactly what one
stage holding the whole stack would have captured. Every stage also warms up on
the rows the last stage does.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

# `exc_type`: a test collected after one that stubs `sys.modules["aiter"]`
# (tests/test_pd_pp.py) imports the name and dies on a symbol.
relay_mod = pytest.importorskip(
    "atom.spec_decode.pp_aux_relay",
    reason="the relay reaches the drafter, which imports aiter at module load",
    exc_type=ImportError,
)
drafter_mod = pytest.importorskip("atom.spec_decode.drafter", exc_type=ImportError)

from atom.models.utils import IntermediateTensors, PPMissingLayer

TAPS, HIDDEN = [1, 3, 5], 8
CONFIG = SimpleNamespace(
    speculative_config=SimpleNamespace(
        draft_model_hf_config=SimpleNamespace(dspark_target_layer_ids=TAPS)
    ),
    max_num_batched_tokens=16,
    hf_config=SimpleNamespace(hidden_size=HIDDEN),
    torch_dtype=torch.float32,
)


class _Layer(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, x):
        return x * 1.5 + self.idx


class _Stage(nn.Module):
    def __init__(self, held):
        super().__init__()
        self.layers = nn.ModuleList(
            _Layer(i) if i in held else PPMissingLayer() for i in range(6)
        )

    def run(self, x):
        for layer in self.layers:
            if not isinstance(layer, PPMissingLayer):
                x = layer(x)
        return x


class _Drafter:
    model = None

    def __init__(self):
        self.got = {}

    def accept_relayed_aux(self, layer_id, tensor):
        self.got[layer_id] = tensor.clone()


@pytest.fixture(autouse=True)
def _target_forward(monkeypatch):
    ctx = SimpleNamespace(
        context=SimpleNamespace(is_draft=False, ubatch_token_offset=0)
    )
    monkeypatch.setattr(drafter_mod, "get_forward_context", lambda: ctx)


def _send(stage, relay, h):
    h = stage.run(h)
    payload = IntermediateTensors({"hidden_states": h, "residual": h})
    relay.attach(payload)
    return payload


def test_last_stage_receives_every_tap_it_does_not_run():
    x = torch.randn(5, HIDDEN)
    expect, h = {}, x
    for i in range(6):
        h = h * 1.5 + i
        expect[i] = h.clone()
    drafter = _Drafter()
    stage0, stage1, stage2 = _Stage({0, 1}), _Stage({2, 3}), _Stage({4, 5})
    relay0 = relay_mod.PPAuxRelay(CONFIG, stage0, None, "cpu")
    relay1 = relay_mod.PPAuxRelay(CONFIG, stage1, None, "cpu")
    relay2 = relay_mod.PPAuxRelay(CONFIG, stage2, drafter, "cpu")

    payload = _send(stage0, relay0, x)
    relay1.take_received(payload)
    assert set(payload.tensors) == {"hidden_states", "residual"}
    payload = _send(stage1, relay1, payload["hidden_states"])
    relay2.take_received(payload)
    relay2.deliver(drafter)
    assert set(drafter.got) == {1, 3}  # layer 5 is the drafter's own hook
    for i in (1, 3):
        torch.testing.assert_close(drafter.got[i], expect[i])


def test_last_stage_refuses_a_forward_missing_a_tap():
    relay = relay_mod.PPAuxRelay(CONFIG, _Stage({4, 5}), _Drafter(), "cpu")
    relay.take_received(
        IntermediateTensors(
            {"hidden_states": torch.zeros(2, 8), "draft_aux_1": torch.zeros(2, 8)}
        )
    )
    with pytest.raises(RuntimeError, match=r"expected taps \[1, 3\]"):
        relay.deliver(_Drafter())


def test_pp_payload_keeps_relayed_taps_and_drops_strangers():
    pp_comm = pytest.importorskip("atom.distributed.pp_comm")
    payload = IntermediateTensors(
        {
            "hidden_states": 1,
            "residual": 2,
            "draft_aux_17": 3,
            "other": 4,
            "sparse_kv_indices": 5,
        }
    )
    assert set(pp_comm._stage_payload(payload)) == {
        "hidden_states",
        "residual",
        "draft_aux_17",
        "sparse_kv_indices",
    }


def test_every_stage_warms_up_on_the_rows_the_last_stage_does(monkeypatch):
    """A block drafter warms up on >= 2 sequences. Read off the drafter, a
    stage without one sent 1 x 5000 rows to a last stage expecting 2 x 4096."""
    runner_mod = pytest.importorskip(
        "atom.model_engine.model_runner", exc_type=ImportError
    )
    monkeypatch.setattr(runner_mod.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(runner_mod.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(
        runner_mod, "get_dp_group", lambda: SimpleNamespace(world_size=1)
    )
    rows = []
    for drafter in (None, SimpleNamespace(is_block_drafter=True)):
        runner = object.__new__(runner_mod.ModelRunner)
        runner.config = SimpleNamespace(
            max_model_len=5000,
            max_num_seqs=8,
            max_num_batched_tokens=8192,
            enable_dp_attention=False,
            prefill_context_parallel_size=1,
            speculative_config=SimpleNamespace(use_dspark=lambda: True),
        )
        if drafter is not None:
            runner.drafter = drafter
        runner.block_size, runner.label = 16, "stage"
        runner.tokenID_processor = SimpleNamespace(clean=lambda: None)
        runner.forward = lambda batch: rows.append(
            (batch.total_seqs_num, batch.total_tokens_num)
        )
        runner.warmup_model()
    assert rows == [(2, 8192), (2, 8192)]
