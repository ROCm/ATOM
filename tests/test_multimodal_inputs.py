# SPDX-License-Identifier: MIT
"""Tests for media normalization, ordering and native image preparation."""

import base64
import io
from types import SimpleNamespace

import numpy as np
import pytest

from atom.multimodal import (
    AudioData,
    MediaLoader,
    VideoData,
    normalize_multimodal_data,
)
from atom.multimodal.media import load_multimodal_data
from atom.multimodal.processing import prepare_multimodal_inputs


@pytest.fixture
def image_module():
    return pytest.importorskip("PIL.Image")


def chat(*parts):
    return [{"role": "user", "content": list(parts)}]


def config(architecture="Qwen3_5ForConditionalGeneration"):
    return SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[architecture]),
        multimodal_config=SimpleNamespace(
            media_placeholder_token_id=42,
            vision_config=SimpleNamespace(merge_kernel_size=2),
        ),
    )


@pytest.mark.parametrize("modality", ["image", "video", "audio"])
def test_single_media_and_lists_share_a_protocol_independent_representation(modality):
    url = "https://example.invalid/media"
    single = normalize_multimodal_data({modality: url})
    batch = normalize_multimodal_data({modality: [url]})
    assert single == batch == {modality: [url]}


def test_empty_modalities_are_omitted_without_mutating_input_lists():
    images = ["first.png", "second.png"]
    data = {"image": images, "video": [], "audio": None}
    normalized = normalize_multimodal_data(data)
    assert normalized == {"image": images}
    assert normalized["image"] is not images
    assert data["video"] == []
    assert data["audio"] is None


def test_audio_tuples_are_single_items_and_lists_contain_separate_recordings():
    first = np.zeros(8, dtype=np.float32)
    second = np.ones(4, dtype=np.float32)
    single = normalize_multimodal_data({"audio": (first, 16000)})
    batch = normalize_multimodal_data({"audio": [(first, 16000), (second, 8000)]})
    assert len(single["audio"]) == 1
    assert single["audio"][0].waveform is first
    assert [item.sampling_rate for item in batch["audio"]] == [16000, 8000]
    assert batch["audio"][1].waveform is second


def test_decoded_audio_and_video_keep_timing_metadata():
    waveform = np.zeros(8, dtype=np.float32)
    video = VideoData(np.zeros((2, 2, 2, 3), dtype=np.uint8), timestamps=(0.0, 0.2))
    inputs = normalize_multimodal_data({"audio": (waveform, 16000), "video": video})
    media = load_multimodal_data(inputs, MediaLoader())
    assert isinstance(media["audio"][0], AudioData)
    assert media["audio"][0].waveform is waveform
    assert media["audio"][0].sampling_rate == 16000
    assert media["video"][0] is video
    assert media["video"][0].timestamps == (0.0, 0.2)


@pytest.mark.parametrize(
    "data",
    [
        {"image_url": "image.png"},  # Protocol fields belong to the entrypoint.
        {"messages": []},
        [{"role": "user", "content": []}],
        {"video": ""},
        {"audio": np.zeros(8)},  # No sampling rate.
        {"video": np.zeros((2, 2, 2, 3))},  # No timing metadata.
        {"audio": [None]},
    ],
)
def test_invalid_or_ambiguous_media_data_is_rejected(data):
    with pytest.raises(ValueError):
        normalize_multimodal_data(data)


@pytest.mark.parametrize("sampling_rate", [0, -1, True, 1.5])
def test_audio_requires_positive_integer_sampling_rate(sampling_rate):
    with pytest.raises(ValueError, match="sampling_rate"):
        AudioData(np.zeros(2), sampling_rate)


@pytest.mark.parametrize(
    "timing",
    [
        {},
        {"fps": 0},
        {"fps": float("nan")},
        {"timestamps": (0.0,)},
        {"timestamps": (0.2, 0.1)},
    ],
)
def test_video_requires_valid_frame_timing(timing):
    with pytest.raises(ValueError):
        VideoData(np.zeros((2, 2, 2, 3), dtype=np.uint8), **timing)


@pytest.mark.parametrize(
    "source_kind", ["path", "file_url", "data_url", "bytes", "pil"]
)
def test_image_sources_produce_the_same_rgb_image(tmp_path, source_kind, image_module):
    original = image_module.new("RGBA", (2, 3), (10, 20, 30, 255))
    encoded = io.BytesIO()
    original.save(encoded, format="PNG")
    path = tmp_path / "image with spaces.png"
    path.write_bytes(encoded.getvalue())
    sources = {
        "path": path,
        "file_url": path.as_uri(),
        "data_url": "data:image/png;base64,"
        + base64.b64encode(encoded.getvalue()).decode(),
        "bytes": encoded.getvalue(),
        "pil": original,
    }
    inputs = normalize_multimodal_data({"image": sources[source_kind]})
    media = load_multimodal_data(inputs, MediaLoader())
    loaded = media["image"][0]
    assert loaded.mode == "RGB"
    assert loaded.size == (2, 3)
    assert loaded.getpixel((0, 0)) == (10, 20, 30)


def test_http_image_loading_uses_the_shared_connector(monkeypatch, image_module):
    from atom.multimodal.media import connector

    data = io.BytesIO()
    image_module.new("RGB", (2, 3), "red").save(data, format="PNG")
    seen = []

    def urlopen(url, timeout):
        seen.append((url, timeout))
        return io.BytesIO(data.getvalue())

    monkeypatch.setattr(connector.urllib.request, "urlopen", urlopen)
    inputs = normalize_multimodal_data({"image": "https://example.invalid/image"})
    media = load_multimodal_data(inputs, MediaLoader(timeout=5))
    assert seen == [("https://example.invalid/image", 5)]
    assert media["image"][0].getpixel((0, 0)) == (255, 0, 0)


@pytest.mark.parametrize("source", [b"not an image", "data:image/png;base64,%%"])
def test_invalid_encoded_images_raise_input_validation_errors(source, image_module):
    inputs = normalize_multimodal_data({"image": source})
    with pytest.raises(ValueError, match="Invalid"):
        load_multimodal_data(inputs, MediaLoader())


@pytest.mark.parametrize("modality", ["video", "audio"])
def test_native_unsupported_modalities_fail_before_loading_any_media(modality):
    class NoLoad(MediaLoader):
        def load(self, modality, data):
            pytest.fail("Unsupported request must be rejected before loading media")

    with pytest.raises(ValueError, match="supported modalities: image"):
        prepare_multimodal_inputs(
            config(),
            None,
            "unused prompt",
            {"image": "a.png", modality: "https://example.invalid/media"},
            media_loader=NoLoad(),
        )


def test_qwen_preparation_preserves_template_options_and_hf_tokens_and_tensors(
    image_module,
):
    first = image_module.new("RGB", (2, 2), "red")
    second = image_module.new("RGB", (2, 2), "blue")
    hf_output = {
        "input_ids": np.array([[1, 2, 3]]),
        "pixel_values": np.ones((4, 3)),
        "image_grid_thw": np.array([[1, 2, 2], [1, 2, 2]]),
    }
    seen = {}

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            seen["messages"], seen["template_kwargs"] = messages, kwargs
            return "<|image_pad|><|image_pad|> prompt"

        def __call__(self, **kwargs):
            seen["processor_kwargs"] = kwargs
            return hf_output

    messages = chat(
        {"type": "text", "text": "before"},
        {"type": "image"},
        {"type": "text", "text": "after"},
        {"type": "image"},
    )
    messages[0]["name"] = "user_1"
    kwargs = {
        "tokenize": True,
        "add_generation_prompt": False,
        "enable_thinking": False,
    }
    token_ids, model_inputs = prepare_multimodal_inputs(
        config(), Processor(), messages, {"image": [first, second]}, kwargs
    )
    assert token_ids == [1, 2, 3]
    assert model_inputs["pixel_values"] is hf_output["pixel_values"]
    assert model_inputs["image_grid_thw"] is hf_output["image_grid_thw"]
    assert seen["messages"][0]["name"] == "user_1"
    parts = seen["messages"][0]["content"]
    assert [part["type"] for part in parts] == ["image", "image", "text"]
    assert parts[-1]["text"] == "before\nafter"
    assert [im.getpixel((0, 0)) for im in seen["processor_kwargs"]["images"]] == [
        (255, 0, 0),
        (0, 0, 255),
    ]
    assert seen["template_kwargs"] == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": False,
    }
    assert kwargs["tokenize"] is True  # Caller-owned options remain unchanged.


def test_kimi_preparation_expands_each_image_and_keeps_interleaving(image_module):
    seen = {}
    pixel_values = np.ones((3, 4))
    grid = np.array([[1, 4, 4], [3, 4, 8]])

    def processor(**kwargs):
        seen.update(kwargs)
        return {
            "input_ids": np.array([[9, 42, 7, 42, 8]]),
            "pixel_values": pixel_values,
            "grid_thws": grid,
        }

    messages = chat(
        {"type": "image"},
        {"type": "text", "text": "between"},
        {"type": "image"},
    )
    token_ids, model_inputs = prepare_multimodal_inputs(
        config("KimiK3ForConditionalGeneration"),
        processor,
        messages,
        {"image": [image_module.new("RGB", (2, 2)), image_module.new("RGB", (4, 2))]},
    )
    assert token_ids == [9] + [42] * 4 + [7] + [42] * 8 + [8]
    assert model_inputs["image_grid_thw"] is grid
    assert model_inputs["pixel_values"] is pixel_values
    assert [part["type"] for part in seen["messages"][0]["content"]] == [
        "image",
        "text",
        "image",
    ]
    assert [entry["image"].size for entry in seen["medias"]] == [(2, 2), (4, 2)]


@pytest.mark.parametrize(
    "architecture",
    ["Qwen3_5ForConditionalGeneration", "KimiK3ForConditionalGeneration"],
)
def test_separate_images_bind_in_order_across_chat_turns(image_module, architecture):
    seen = {}

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            seen["messages"] = messages
            return "<|image_pad|> first turn <|image_pad|> second turn"

        def __call__(self, **kwargs):
            if "messages" in kwargs:
                seen["messages"] = kwargs["messages"]
            images = kwargs.get("images")
            if images is None:
                images = [item["image"] for item in kwargs["medias"]]
            return {
                "input_ids": np.array([[42, 7, 42]]),
                "pixel_values": np.stack([np.asarray(image) for image in images]),
                "image_grid_thw": np.array([[1, 2, 2], [1, 2, 2]]),
                "grid_thws": np.array([[1, 2, 2], [1, 2, 2]]),
            }

    conversation = [
        {"role": "user", "content": [{"type": "image", "detail": "high"}]},
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]},
        {"role": "tool", "tool_call_id": "call_1", "content": [{"type": "image"}]},
    ]
    images = [image_module.new("RGB", (2, 2), color) for color in ["red", "blue"]]
    _, media = prepare_multimodal_inputs(
        config(architecture), Processor(), conversation, {"image": images}
    )
    assert media["pixel_values"][:, 0, 0].tolist() == [[255, 0, 0], [0, 0, 255]]
    assert seen["messages"][0]["content"][0]["image"].getpixel((0, 0)) == (255, 0, 0)
    assert seen["messages"][2]["content"][0]["image"].getpixel((0, 0)) == (0, 0, 255)
    assert seen["messages"][0]["content"][0]["detail"] == "high"
    assert seen["messages"][1]["tool_calls"] == [{"id": "call_1"}]
    assert seen["messages"][1]["content"] == ""
    assert seen["messages"][2]["tool_call_id"] == "call_1"
    assert conversation[0]["content"] == [{"type": "image", "detail": "high"}]
    assert conversation[1]["content"] is None


def test_rendered_qwen_prompt_accepts_media_without_chat_messages(image_module):
    seen = {}

    def processor(**kwargs):
        seen.update(kwargs)
        return {
            "input_ids": np.array([[1, 42, 2]]),
            "pixel_values": np.asarray(kwargs["images"][0]),
            "image_grid_thw": np.array([[1, 2, 2]]),
        }

    prompt = "<|vision_start|><|image_pad|><|vision_end|> Describe this image."
    tokens, _ = prepare_multimodal_inputs(
        config(), processor, prompt, {"image": image_module.new("RGB", (2, 2))}
    )
    assert tokens == [1, 42, 2]
    assert seen["text"] == [prompt]


@pytest.mark.parametrize("num_markers", [0, 2])
def test_conversation_image_count_mismatch_fails_before_processing(
    image_module, num_markers
):
    conversation = chat(*({"type": "image"} for _ in range(num_markers)))
    with pytest.raises(ValueError, match="image markers"):
        prepare_multimodal_inputs(
            config(), None, conversation, {"image": image_module.new("RGB", (2, 2))}
        )


@pytest.mark.parametrize("rendered", [False, True])
@pytest.mark.parametrize("num_placeholders", [0, 2])
def test_qwen_placeholder_count_mismatch_fails_before_processing(
    image_module, rendered, num_placeholders
):
    text = "<|image_pad|>" * num_placeholders + " prompt"
    processor = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: text)
    prompt = text if rendered else chat({"type": "image"})
    with pytest.raises(ValueError, match="image placeholders"):
        prepare_multimodal_inputs(
            config(), processor, prompt, {"image": image_module.new("RGB", (2, 2))}
        )


def test_kimi_rendered_prompt_reports_required_conversation_before_loading():
    with pytest.raises(ValueError, match="requires a chat conversation"):
        prepare_multimodal_inputs(
            config("KimiK3ForConditionalGeneration"),
            None,
            "prompt",
            {"image": "unused.png"},
        )
