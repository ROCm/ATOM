from types import SimpleNamespace as NS

from atom.entrypoints.openai.chat_encoder_adapters import build_message_encoder_adapter
from atom.entrypoints.openai.model_capabilities import deployment_metadata


def test_dspark_reports_runtime_context_and_text_only():
    result = deployment_metadata(
        NS(
            max_model_len=32768,
            method="dspark",
            hf_config=NS(
                text_config=NS(max_position_embeddings=1048576), vision_config=NS()
            ),
        ),
        build_message_encoder_adapter("encoding_dsv41", lambda messages: ""),
    )
    assert result["context_window"] == 32768
    assert result["max_output_tokens"] == 32768
    assert result["input_modalities"] == ["text"]
    assert result["effort"] == {
        "supported_levels": ["low", "high", "max"],
        "default_level": "high",
    }


def test_checkpoint_limit_and_output_cap():
    result = deployment_metadata(
        NS(max_model_len=2097152, hf_config=NS(max_position_embeddings=1048576)),
        build_message_encoder_adapter("encoding_dsv41", lambda messages: ""),
    )
    assert result["context_window"] == 1048576
    assert result["max_output_tokens"] == 393216


def test_unknown_model_does_not_invent_capabilities():
    result = deployment_metadata(None, None)
    assert result == {"output_modalities": ["text"]}


def test_metadata_effort_levels_share_the_request_contract():
    from atom.entrypoints.openai.deepseek_v41_encoder import normalize_effort

    caps = deployment_metadata(
        None, build_message_encoder_adapter("encoding_dsv41", lambda messages: "")
    )
    for level in caps["effort"]["supported_levels"]:
        assert normalize_effort(level) == level
    assert normalize_effort(caps["effort"]["default_level"]) == "high"


def test_unknown_encoder_only_reports_deployment_limits():
    result = deployment_metadata(
        NS(max_model_len=4096, hf_config=NS(max_position_embeddings=8192)),
        NS(name="another_encoder"),
    )
    assert result == {
        "output_modalities": ["text"],
        "context_window": 4096,
        "max_output_tokens": 4096,
    }


def test_generic_contract_import_does_not_load_model_modules():
    import subprocess
    import sys

    code = """
import sys
from atom.entrypoints.openai.api_compatibility import APICompatibility, api_compatibility_for
assert api_compatibility_for(None) == APICompatibility()
assert not any('deepseek' in name for name in sys.modules)
"""
    subprocess.run(
        [sys.executable, "-c", code], check=True, capture_output=True, text=True
    )


def test_contract_selection_uses_metadata_not_encoder_name():
    from atom.entrypoints.openai.api_compatibility import (
        DEFAULT_API_COMPATIBILITY,
        APICompatibility,
        api_compatibility_for,
    )

    contract = APICompatibility(display_name="Custom model", effort_levels=("custom",))
    assert (
        api_compatibility_for(NS(name="unregistered", api_compatibility=contract))
        is contract
    )
    assert api_compatibility_for(NS(name="encoding_dsv41")) is DEFAULT_API_COMPATIBILITY
