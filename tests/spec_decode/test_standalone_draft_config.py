# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`Drafter._standalone_draft_config`: a draft that loads its own checkpoint
(eagle3, DSpark with --draft-model) takes its static quant spec from that
checkpoint, not from the target's.

Driven with a stand-in proposer carrying only what the method reads.
"""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter")  # atom.spec_decode.drafter imports aiter at module scope

from aiter import QuantType

from atom.spec_decode.drafter import Drafter

DRAFT_PATH = "/ckpt/draft"
ONLINE = {"global_quant_config": "ptpc_fp8"}


def _drafter_double(online):
    """What Drafter._standalone_draft_config reads from a proposer."""
    target_qc = SimpleNamespace(online_quant_config_raw={"stale": "target copy"})
    return SimpleNamespace(
        speculative_config=SimpleNamespace(
            draft_model_hf_config=SimpleNamespace(
                model_type="kimi_k3_dspark",
                torch_dtype=torch.bfloat16,
                quantization_config=None,
            ),
            model=DRAFT_PATH,
        ),
        config=SimpleNamespace(
            hf_config=SimpleNamespace(name="target"),
            quant_config=target_qc,
            online_quant_config=online,
            compilation_config=SimpleNamespace(level=3),
        ),
    )


def test_standalone_draft_does_not_inherit_the_target_static_scheme():
    """A bf16 draft beside an MXFP4 target is built unquantized, from its own
    config, with compilation off and the target config untouched."""
    drafter = _drafter_double(None)
    cfg = Drafter._standalone_draft_config(drafter, object)
    assert cfg.hf_config is drafter.speculative_config.draft_model_hf_config
    assert cfg.quant_config.global_spec.quant_type == QuantType.No
    assert not cfg.quant_config.online_quant
    assert cfg.compilation_config.level != 3
    assert drafter.config.compilation_config.level == 3


def test_standalone_draft_gets_the_runs_online_override():
    """The online override is the run's (Config.online_quant_config), not a
    copy kept on the target's quant config."""
    cfg = Drafter._standalone_draft_config(_drafter_double(ONLINE), object)
    assert cfg.quant_config.online_quant
    assert cfg.quant_config.online_quant_config_raw == ONLINE


def test_a_draft_class_can_keep_its_weights_out_of_the_online_override():
    draft_class = SimpleNamespace(draft_skips_online_quant=True)
    cfg = Drafter._standalone_draft_config(_drafter_double(ONLINE), draft_class)
    assert not cfg.quant_config.online_quant


def test_standalone_draft_quant_config_probes_the_draft_checkpoint(monkeypatch):
    """model= is the draft's path, so layers it stores in bf16 can fall back."""
    import atom.config as config_mod

    seen = []
    monkeypatch.setattr(
        config_mod,
        "unquantized_layer_keys",
        lambda model: seen.append(model) or frozenset(),
    )
    Drafter._standalone_draft_config(_drafter_double(None), object)
    assert seen == [DRAFT_PATH]


def test_standalone_draft_quant_config_is_remapped_to_the_draft_modules():
    """Exclude names in HF form are rewritten onto the draft class's packed
    modules, as ModelRunner does for the target."""
    draft_class = SimpleNamespace(
        packed_modules_mapping={"gate_proj": ("gate_up_proj", 0)}
    )
    online = {**ONLINE, "exclude_layer": ["model.layers.0.mlp.gate_proj"]}
    cfg = Drafter._standalone_draft_config(_drafter_double(online), draft_class)
    assert cfg.quant_config.online_exclude_layers == ["model.layers.0.mlp.gate_up_proj"]
