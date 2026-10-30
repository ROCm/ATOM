from types import SimpleNamespace as NS

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
        NS(name="encoding_dsv41"),
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
        NS(name="encoding_dsv41"),
    )
    assert result["context_window"] == 1048576
    assert result["max_output_tokens"] == 393216


def test_unknown_model_does_not_invent_capabilities():
    result = deployment_metadata(None, None)
    assert result == {"output_modalities": ["text"]}
