# SPDX-License-Identifier: MIT
# Tests for atom/model_ops/eplb.py: MoE layers outside the EPLB placement maps
# (e.g. a DSpark/MTP drafter layer whose layer_id is past the target model's
# last MoE layer) pass through unmapped and untracked.

from types import SimpleNamespace

import pytest
from import_guard import skip_if_dependency_missing

torch = pytest.importorskip("torch")

try:
    import atom.config
    from atom.model_ops import eplb
except ImportError as _e:  # aiter/triton absent under bare non-GPU pytest
    skip_if_dependency_missing(_e, "requires full atom import env")

NUM_LAYERS = 2
NUM_LOGICAL = 4


@pytest.fixture
def live_meta(monkeypatch):
    meta = eplb.ExpertLocationMetadata.from_trivial(
        num_layers=NUM_LAYERS, num_logical_experts=NUM_LOGICAL, ep_size=1, ep_rank=0
    )
    monkeypatch.setattr(eplb, "get_live_expert_location_metadata", lambda: meta)
    return meta


def _topk():
    return torch.tensor([[0, 3], [1, 2]], dtype=torch.int32)


def test_map_passes_through_layer_past_placement(live_meta):
    topk = _topk()
    # layer_id == num_layers indexed one past the maps before the fix.
    out = eplb.eplb_map_logical_to_physical(SimpleNamespace(layer_id=NUM_LAYERS), topk)
    assert out is topk


def test_fused_map_passes_through_layer_past_placement(live_meta):
    topk = _topk()
    out = eplb.eplb_map_and_record_fused(SimpleNamespace(layer_id=NUM_LAYERS), topk)
    assert out is topk


def test_record_skips_layer_past_placement(live_meta, monkeypatch):
    monkeypatch.setattr(
        atom.config,
        "get_current_atom_config",
        lambda: SimpleNamespace(
            eplb_enable=True, eplb_config=SimpleNamespace(load_window_size=1)
        ),
    )

    def _no_monitor(**_kwargs):
        raise AssertionError("load must not be recorded for an unplaced layer")

    monkeypatch.setattr(eplb, "get_expert_load_monitor", _no_monitor)
    eplb.record_eplb_expert_load(SimpleNamespace(layer_id=NUM_LAYERS), _topk())


def test_map_still_applies_to_placed_layer(live_meta):
    topk = _topk()
    out = eplb.eplb_map_logical_to_physical(
        SimpleNamespace(layer_id=NUM_LAYERS - 1), topk
    )
    # Trivial placement with one rank: logical ids map to the same physical ids.
    assert out is not topk
    assert out.tolist() == topk.tolist()
