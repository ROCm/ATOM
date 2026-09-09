# SPDX-License-Identifier: MIT

"""A failed full multimodal config load keeps the None sentinel."""

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


@pytest.mark.parametrize(
    ("model_type", "text_model_type"),
    [
        ("glm5_next", "glm5_next_text"),
        ("qwen3_5", "qwen3_5"),
    ],
)
def test_failed_multimodal_config_load_keeps_the_failure_sentinel(
    monkeypatch, failed_full_config_load, model_type, text_model_type
):
    raw = _config_dict(model_type, text_model_type)
    monkeypatch.setattr(
        PretrainedConfig,
        "get_config_dict",
        lambda *_args, **_kwargs: (raw, {}),
    )

    config = atom_config.get_hf_config("unused")

    assert config._multimodal_config is None
