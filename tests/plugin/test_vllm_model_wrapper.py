# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`atom.plugin.vllm.model_wrapper`: the ATOM config a standalone draft is
built with under vLLM takes its static quant spec from the draft's own
checkpoint, not the target's."""


def _standalone_draft_case(monkeypatch, online_raw):
    from types import SimpleNamespace

    import torch
    from aiter import QuantType, dtypes

    from atom.config import QuantizationConfig
    from atom.plugin.vllm import model_wrapper
    from atom.quant_spec import LayerQuantConfig

    # The target's config, as _generate_atom_config_from_vllm_config builds it:
    # a statically MXFP4 target.
    target_qc = QuantizationConfig()
    target_qc.global_spec = LayerQuantConfig(
        quant_type=QuantType.per_1x32, quant_dtype=dtypes.fp4x2
    )
    target_qc.online_quant_config_raw = {"stale": "target copy"}
    draft_hf = SimpleNamespace(torch_dtype=torch.bfloat16, quantization_config=None)
    target_atom_config = SimpleNamespace(
        hf_config=SimpleNamespace(name="target"),
        quant_config=target_qc,
        online_quant_config=online_raw,
        speculative_config=SimpleNamespace(
            draft_model_hf_config=draft_hf, model="/ckpt/draft"
        ),
    )
    monkeypatch.setattr(
        model_wrapper,
        "_generate_atom_config_from_vllm_config",
        lambda vllm_config: target_atom_config,
    )
    vllm_draft_hf = SimpleNamespace(name="vllm's un-normalized copy")
    cfg = model_wrapper._standalone_draft_atom_config(
        object(), vllm_draft_hf, is_dspark_draft=True
    )
    return cfg, draft_hf, QuantType


def test_vllm_standalone_draft_does_not_inherit_the_target_static_quant(monkeypatch):
    """A bf16 DSpark draft beside an MXFP4 target is built unquantized, from
    the normalized draft config."""
    cfg, draft_hf, QuantType = _standalone_draft_case(monkeypatch, None)
    assert cfg.hf_config is draft_hf, "the normalized draft config must be used"
    assert cfg.quant_config.global_spec.quant_type.value == QuantType.No.value
    assert not cfg.quant_config.online_quant


def test_vllm_standalone_draft_keeps_the_online_override(monkeypatch):
    online = {"global_quant_config": "ptpc_fp8"}
    cfg, _, _ = _standalone_draft_case(monkeypatch, online)
    assert cfg.quant_config.online_quant
    assert cfg.quant_config.online_quant_config_raw == online
