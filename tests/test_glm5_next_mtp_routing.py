# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from types import SimpleNamespace

import pytest

from atom.config import SpeculativeConfig, _glm5_next_unsupported_features


def _aiter_unavailable(error):
    missing_aiter = (
        isinstance(error, ModuleNotFoundError)
        and error.name is not None
        and (error.name == "aiter" or error.name.startswith("aiter."))
    )
    missing_device = isinstance(error, RuntimeError) and "rocminfo" in str(error)
    return missing_aiter or missing_device


def _mtp_symbols():
    try:
        from atom.models.glm5_next_mtp import (
            Glm5NextMTP,
            _add_mtp_quant_excludes,
        )
    except (ModuleNotFoundError, RuntimeError) as error:
        if not _aiter_unavailable(error):
            raise
        pytest.skip("AITER imports require a visible ROCm device")
    return Glm5NextMTP, _add_mtp_quant_excludes


def _eagle_proposer():
    try:
        from atom.spec_decode.eagle_proposer import EagleProposer
    except (ModuleNotFoundError, RuntimeError) as error:
        if not _aiter_unavailable(error):
            raise
        pytest.skip("AITER imports require a visible ROCm device")
    return EagleProposer


def test_glm5_next_text_routes_to_glm_mtp_model():
    config = SimpleNamespace(
        model_type="glm5_next_text",
        architectures=["Glm5NextForConditionalGeneration"],
        num_nextn_predict_layers=1,
    )
    config.update = lambda values: [
        setattr(config, key, value) for key, value in values.items()
    ]

    SpeculativeConfig.hf_config_override(config)

    assert config.model_type == "glm5_next_mtp"
    assert config.architectures == ["Glm5NextMTPModel"]
    assert config.n_predict == 1


def test_glm5_next_allows_mtp_but_rejects_unimplemented_parallel_modes():
    config = SimpleNamespace(
        speculative_config=object(),
        prefill_context_parallel_size=1,
        decode_context_parallel_size=1,
        enable_tbo=False,
        enable_tbo_decode=False,
    )

    assert _glm5_next_unsupported_features(config) == []

    config.prefill_context_parallel_size = 2
    config.decode_context_parallel_size = 2
    config.enable_tbo = True
    assert _glm5_next_unsupported_features(config) == ["PCP", "DCP", "TBO"]


def test_glm5_next_mtp_remaps_language_model_checkpoint_layer():
    Glm5NextMTP, _ = _mtp_symbols()
    model = object.__new__(Glm5NextMTP)
    model.config = SimpleNamespace(
        num_hidden_layers=45,
        num_nextn_predict_layers=1,
    )

    assert (
        model.remap_mtp_weight_name(
            "model.language_model.layers.45.self_attn.q_a_proj.weight"
        )
        == "model.layers.45.mtp_block.self_attn.q_a_proj.weight"
    )
    assert (
        model.remap_mtp_weight_name("model.language_model.layers.45.eh_proj.weight")
        == "model.layers.45.eh_proj.weight"
    )
    assert (
        model.remap_mtp_weight_name(
            "model.language_model.layers.45.shared_head.norm.weight"
        )
        == "model.layers.45.shared_head.norm.weight"
    )
    assert (
        model.remap_mtp_weight_name(
            "model.language_model.layers.44.self_attn.q_a_proj.weight"
        )
        is None
    )
    assert Glm5NextMTP.weights_mapping == {
        "index_kpool_compress_gate": "index_kpool_compress_gate.weight"
    }


def test_glm5_next_mtp_mirrors_block_quant_excludes():
    _, _add_mtp_quant_excludes = _mtp_symbols()
    quant_config = SimpleNamespace(
        exclude_layers=[
            "model.layers.45.self_attn.q_a_proj",
            "model.layers.45.input_layernorm",
            "model.layers.45.eh_proj",
            "model.layers.44.self_attn.q_a_proj",
        ],
        online_exclude_layers=["model.layers.45.mlp.gate"],
    )
    atom_config = SimpleNamespace(
        hf_config=SimpleNamespace(num_hidden_layers=45),
        quant_config=quant_config,
    )

    _add_mtp_quant_excludes(atom_config)

    assert "model.layers.45.mtp_block.self_attn.q_a_proj" in (
        quant_config.exclude_layers
    )
    assert "model.layers.45.mtp_block.input_layernorm" in (quant_config.exclude_layers)
    assert "model.layers.45.mtp_block.mlp.gate" in (quant_config.online_exclude_layers)
    assert "model.layers.45.mtp_block.eh_proj" not in quant_config.exclude_layers


@pytest.mark.parametrize(
    "architecture, expected",
    [
        ("Glm5NextMTPModel", (4096,)),
        ("DeepSeekV4MTPModel", (4, 4096)),
        ("DeepSeekMTPModel", (4096,)),
    ],
)
def test_mtp_graph_stages_actual_residual_shape(architecture, expected):
    import torch

    EagleProposer = _eagle_proposer()

    hf = SimpleNamespace(architectures=[architecture], hidden_size=4096)
    if architecture != "DeepSeekMTPModel":
        hf.hc_mult = 4
    proposer = SimpleNamespace(
        runner=SimpleNamespace(use_mrope=False),
        mtp_k=3,
        speculative_config=SimpleNamespace(draft_model_hf_config=hf),
        dtype=torch.bfloat16,
        _step_forward=lambda *a, **kw: None,
        _step_head=lambda *a, **kw: None,
        _step_warmup_inputs=lambda *a, **kw: None,
    )
    (graph,) = EagleProposer._declare_draft_graphs(proposer)
    assert graph.inputs["hidden_states"].shape == expected
    graph.bind(SimpleNamespace(max_num_seqs=8), "cpu")
    source = torch.ones((1, *expected), dtype=torch.bfloat16)
    staged = graph.stage(4, {"hidden_states": source})["hidden_states"]
    assert staged.shape == (4, *expected)
    torch.testing.assert_close(staged, source.expand_as(staged))
