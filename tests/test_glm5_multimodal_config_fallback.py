# SPDX-License-Identifier: MIT

"""Scope the raw multimodal-config fallback to GLM-5.3."""

import pytest
from transformers import PretrainedConfig

from atom import config as atom_config


@pytest.fixture
def failed_full_config_load(monkeypatch):
    monkeypatch.setattr(
        atom_config.AutoConfig,
        "from_pretrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("unknown config")),
    )
    monkeypatch.setattr(
        atom_config.AutoConfig,
        "for_model",
        lambda *_args, **_kwargs: PretrainedConfig,
    )


def _config_dict(model_type: str, text_model_type: str) -> dict:
    return {
        "model_type": model_type,
        "architectures": ["ForConditionalGeneration"],
        "text_config": {
            "model_type": text_model_type,
            "hidden_size": 16,
        },
        "vision_config": {
            "hidden_size": 8,
            "depth": 2,
        },
    }


def test_glm5_rebuilds_its_known_raw_multimodal_schema(
    monkeypatch, failed_full_config_load
):
    raw = _config_dict("glm5_next", "glm5_next_text")
    monkeypatch.setattr(
        PretrainedConfig,
        "get_config_dict",
        lambda *_args, **_kwargs: (raw, {}),
    )

    config = atom_config.get_hf_config("unused")

    assert isinstance(config._multimodal_config, PretrainedConfig)
    assert config._multimodal_config.vision_config.hidden_size == 8


def test_other_multimodal_models_keep_the_failure_sentinel(
    monkeypatch, failed_full_config_load
):
    raw = _config_dict("qwen3_5", "qwen3_5")
    monkeypatch.setattr(
        PretrainedConfig,
        "get_config_dict",
        lambda *_args, **_kwargs: (raw, {}),
    )

    config = atom_config.get_hf_config("unused")

    assert config._multimodal_config is None
