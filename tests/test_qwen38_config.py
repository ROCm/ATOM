"""The published Qwen3.8 text config loads without a Transformers config class."""

import json

from transformers import PretrainedConfig

from atom.config import get_hf_config


def test_qwen4_exp_text_config_uses_plain_schema(tmp_path):
    config = {
        "model_type": "qwen4_exp",
        "architectures": ["Qwen4ExpForConditionalGeneration"],
        "text_config": {
            "model_type": "qwen4_exp_text",
            "hidden_size": 2560,
            "num_hidden_layers": 48,
            "layer_types": ["linear_attention"] * 48,
        },
    }
    (tmp_path / "config.json").write_text(json.dumps(config))

    loaded = get_hf_config(str(tmp_path))

    assert type(loaded) is PretrainedConfig
    assert loaded.model_type == "qwen4_exp_text"
    assert loaded.hidden_size == 2560
    assert loaded.architectures == config["architectures"]
