# SPDX-License-Identifier: MIT
"""Protocol parsing keeps chat structure separate from media normalization."""

import pytest

from atom.entrypoints.chat_utils import has_multimodal_content, parse_chat_messages
from atom.multimodal import EncodedMedia


def chat(*parts):
    return [{"role": "user", "content": list(parts)}]


@pytest.mark.parametrize("modality", ["image", "video", "audio"])
def test_url_aliases_extract_the_same_media_without_loading(modality):
    url = "https://example.invalid/media"
    url_part = f"{modality}_url"
    messages = chat({"type": url_part, url_part: {"url": url}})
    aliased = parse_chat_messages(messages)
    direct = parse_chat_messages(chat({"type": modality, modality: url}))
    assert aliased == direct == (chat({"type": modality}), {modality: [url]})
    assert has_multimodal_content(messages)


def test_interleaved_media_keep_message_order_and_metadata_across_turns():
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1"}]},
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": [
                {"type": "text", "text": "before"},
                {"type": "image_url", "image_url": {"url": "a.png", "detail": "high"}},
                {"type": "text", "text": "between"},
                {"type": "video_url", "video_url": {"url": "b.mp4"}},
                {"type": "audio_url", "audio_url": {"url": "c.wav"}},
            ],
        },
        {"role": "user", "content": [{"type": "image", "image": "d.png"}]},
    ]
    conversation, media = parse_chat_messages(messages)
    assert conversation[0] == messages[0]
    assert conversation[1]["tool_call_id"] == "call_1"
    assert conversation[1]["content"] == [
        {"type": "text", "text": "before"},
        {"type": "image", "detail": "high"},
        {"type": "text", "text": "between"},
        {"type": "video"},
        {"type": "audio"},
    ]
    assert conversation[2]["content"] == [{"type": "image"}]
    assert media == {
        "image": ["a.png", "d.png"],
        "video": ["b.mp4"],
        "audio": ["c.wav"],
    }
    assert messages[1]["content"][1]["type"] == "image_url"


def test_inline_audio_extracts_encoded_bytes_and_format():
    messages = chat(
        {"type": "input_audio", "input_audio": {"data": "YWJj", "format": "wav"}}
    )
    conversation, media = parse_chat_messages(messages)
    assert has_multimodal_content(messages)
    assert conversation == chat({"type": "audio"})
    assert media == {"audio": [EncodedMedia(b"abc", "wav")]}


@pytest.mark.parametrize(
    "part",
    [
        {"type": "image_url", "image_url": {}},
        {"type": "image"},
        {"type": "input_audio", "input_audio": {"data": "%%", "format": "wav"}},
        {"type": "input_audio", "input_audio": {"data": "YWJj"}},
        {"type": "text", "text": None},
        {"type": "unknown"},
        {"type": []},
        "image.png",
    ],
)
def test_invalid_content_is_rejected_instead_of_silently_dropped(part):
    with pytest.raises(ValueError):
        parse_chat_messages(chat(part))
