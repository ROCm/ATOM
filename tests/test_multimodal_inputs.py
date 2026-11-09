# SPDX-License-Identifier: MIT
"""Tests for media normalization, ordering and native image preparation."""

import base64
import io
import os
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
from atom.multimodal.processing import media_token_spans, prepare_multimodal_inputs


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
def test_image_sources_produce_the_same_image(tmp_path, source_kind, image_module):
    original = image_module.new("RGBA", (2, 3), (10, 20, 30, 128))
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
    # The decoded mode is kept: color conversion is the processor's.
    assert loaded.mode == "RGBA"
    assert loaded.size == (2, 3)
    assert loaded.getpixel((0, 0)) == (10, 20, 30, 128)


def _png(image, **save_kwargs):
    data = io.BytesIO()
    image.save(data, format="PNG", **save_kwargs)
    return data.getvalue()


def _transparency_cases(image_module):
    """Encoded PNGs that carry transparency in each way PNG can."""
    rgba = image_module.new("RGBA", (40, 30), (0, 0, 0, 0))
    rgba.paste((20, 20, 20, 255), (10, 10, 30, 20))
    keyed_rgb = image_module.new("RGB", (40, 30), (0, 0, 0))
    keyed_rgb.paste((200, 50, 50), (10, 10, 30, 20))
    palette = image_module.new("P", (40, 30), 0)
    palette.putpalette([0, 0, 0, 20, 20, 20] + [0, 0, 0] * 254)
    palette.paste(1, (10, 10, 30, 20))
    return {
        "rgba": _png(rgba),
        "rgb_trns": _png(keyed_rgb, transparency=(0, 0, 0)),
        "palette_transparency": _png(palette, transparency=0),
        "opaque_rgba": _png(image_module.new("RGBA", (40, 30), (9, 8, 7, 255))),
        "grayscale": _png(image_module.new("L", (40, 30), 77)),
    }


@pytest.mark.parametrize(
    "case, mode",
    [
        ("rgba", "RGBA"),
        ("rgb_trns", "RGB"),
        ("palette_transparency", "P"),
        ("opaque_rgba", "RGBA"),
        ("grayscale", "L"),
    ],
)
def test_loaded_images_keep_their_mode_and_transparency(image_module, case, mode):
    encoded = _transparency_cases(image_module)[case]
    loaded = MediaLoader().load_image(encoded)
    with image_module.open(io.BytesIO(encoded)) as reference:
        assert loaded.mode == reference.mode == mode
        assert loaded.info.get("transparency") == reference.info.get("transparency")
        assert loaded.tobytes() == reference.tobytes()


def test_kimi_k3_processor_sees_the_same_image_as_the_reference(image_module):
    """The checkpoint processor gives the same pixels for an ATOM-loaded image
    as for the encoded image it would open itself, transparency included.

    Set ATOM_KIMI_K3_REFERENCE to a Kimi-K3 checkpoint directory (only the
    processor files are read).
    """

    directory = os.environ.get("ATOM_KIMI_K3_REFERENCE")
    if not directory:
        pytest.skip("Set ATOM_KIMI_K3_REFERENCE for the checkpoint processor")
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(directory, trust_remote_code=True)
    for case, encoded in _transparency_cases(image_module).items():
        loaded = MediaLoader().load_image(encoded)
        actual = processor.preprocess([{"type": "image", "image": loaded}])
        expected = processor.preprocess([{"type": "image", "image": encoded}])
        assert np.array_equal(
            np.asarray(actual["pixel_values"]), np.asarray(expected["pixel_values"])
        ), case


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
    # One span per expanded run, in image order, so the prefill can chunk.
    assert model_inputs["embedding_spans"] == ((1, 4), (6, 8))
    assert isinstance(model_inputs["cache_seed"], int)
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


IMG = 248056  # Qwen3.5 <|image_pad|>


def qwen3_5_config(model_type="qwen3_5_moe"):
    return SimpleNamespace(
        hf_config=SimpleNamespace(architectures=["Qwen3_5MoeForConditionalGeneration"]),
        multimodal_config=SimpleNamespace(
            model_type=model_type,
            image_token_id=IMG,
            vision_config=SimpleNamespace(spatial_merge_size=2),
        ),
    )


def prepare_qwen3_5(image_module, input_ids, image_grid_thw, model_type="qwen3_5_moe"):
    hf_output = {
        "input_ids": np.array([input_ids]),
        "pixel_values": np.ones((sum(t * h * w for t, h, w in image_grid_thw), 3)),
        "image_grid_thw": np.array(image_grid_thw),
    }

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            return "<|image_pad|>" * len(image_grid_thw) + " prompt"

        def __call__(self, **kwargs):
            return hf_output

    images = [image_module.new("RGB", (2, 2)) for _ in image_grid_thw]
    conversation = chat(*({"type": "image"} for _ in images))
    return prepare_multimodal_inputs(
        qwen3_5_config(model_type), Processor(), conversation, {"image": images}
    )


@pytest.mark.parametrize("model_type", ["qwen3_5", "qwen3_5_moe"])
def test_qwen3_5_image_prompts_carry_one_span_per_image(image_module, model_type):
    # Two images: a 4x4 grid (4 tokens after the 2x2 merge) and a 2x4 grid
    # (2 tokens), the second directly after the first with no separator.
    input_ids = [1, 2, IMG, IMG, IMG, IMG, IMG, IMG, 3]
    token_ids, data = prepare_qwen3_5(
        image_module, input_ids, [[1, 4, 4], [1, 2, 4]], model_type
    )
    assert token_ids == input_ids
    assert data["embedding_spans"] == ((2, 4), (6, 2))
    assert isinstance(data["cache_seed"], int)


def test_qwen3_5_cache_seed_tells_images_apart(image_module):
    _, first = prepare_qwen3_5(image_module, [IMG] * 4, [[1, 4, 4]])
    _, same = prepare_qwen3_5(image_module, [IMG] * 4, [[1, 4, 4]])
    _, other_grid = prepare_qwen3_5(image_module, [IMG] * 4, [[1, 2, 8]])
    assert first["cache_seed"] == same["cache_seed"]
    assert first["cache_seed"] != other_grid["cache_seed"]


@pytest.mark.parametrize(
    "input_ids",
    [
        [1, IMG, IMG, IMG, 2],  # one token short of the grid's 4
        [1, IMG, IMG, 2, IMG, IMG],  # the run is split
        [1, IMG, IMG, IMG, IMG, IMG],  # one image token too many
    ],
)
def test_qwen3_5_image_tokens_that_disagree_with_the_grid_are_rejected(
    image_module, input_ids
):
    with pytest.raises(ValueError, match="placeholder tokens"):
        prepare_qwen3_5(image_module, input_ids, [[1, 4, 4]])


def test_other_qwen_model_types_keep_the_whole_image_merge(image_module):
    _, data = prepare_qwen3_5(image_module, [IMG] * 4, [[1, 4, 4]], "qwen4_exp")
    assert "embedding_spans" not in data and "cache_seed" not in data


P = 9  # a placeholder id for the shared span scan


@pytest.mark.parametrize(
    "input_ids, counts, spans",
    [
        # MiniMax-M3: each run wrapped in start / end-of-image tokens.
        ([1, 5, P, P, P, 6, 2, 5, P, P, 6], [3, 2], ((2, 3), (8, 2))),
        # Qwen3.5: adjacent runs with no separator, split by their counts.
        ([1, P, P, P, P, P, 2], [3, 2], ((1, 3), (4, 2))),
        ([1, 2, 3], [], ()),
    ],
)
def test_media_token_spans_claims_each_items_run_in_order(input_ids, counts, spans):
    assert media_token_spans(input_ids, counts, P) == spans


@pytest.mark.parametrize(
    "input_ids, counts",
    [
        ([1, P, P, 2], [3]),  # run shorter than the item needs
        ([1, P, P, 2, P], [3]),  # run split by another token
        ([1, P, P, P, P], [3]),  # placeholder left over after the last item
        ([1, P, P, P], [3, 1]),  # an item with no run at all
    ],
)
def test_media_token_spans_rejects_runs_that_disagree_with_the_items(input_ids, counts):
    with pytest.raises(ValueError, match="placeholder tokens"):
        media_token_spans(input_ids, counts, P)
