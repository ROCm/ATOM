# SPDX-License-Identifier: MIT
"""A DSpark drafter's tapped target layers, carried across pipeline stages.

Three stages over real hooks and a real payload dict; the last stage must
receive exactly what one stage holding the whole stack would have captured.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

relay_mod = pytest.importorskip("atom.spec_decode.pp_aux_relay")
drafter_mod = pytest.importorskip("atom.spec_decode.drafter")

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
        {"hidden_states": 1, "residual": 2, "draft_aux_17": 3, "other": 4}
    )
    assert set(pp_comm._stage_payload(payload)) == {
        "hidden_states",
        "residual",
        "draft_aux_17",
    }
