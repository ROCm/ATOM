# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

from typing import ClassVar

import pytest
from atomrl.data import dataset
from atomrl.data.sft_dataset import normalize_messages, tokenize_chat_messages


class _Tokenizer:
    _SPECIAL: ClassVar = {
        "<|im_assistant|>": 500,
        "<|im_middle|>": 501,
        "<|im_end|>": 502,
    }

    def convert_tokens_to_ids(self, token):
        return self._SPECIAL[token]

    def apply_chat_template(
        self,
        messages,
        *,
        add_generation_prompt,
        tokenize,
    ):
        assert tokenize
        assert not add_generation_prompt
        output = []
        for message in messages:
            output.extend(
                [
                    500 if message["role"] == "assistant" else 400,
                    501,
                    20 + len(str(message.get("content", ""))),
                    502,
                ]
            )
        return output


def test_hf_chat_template_builds_assistant_only_loss_mask() -> None:
    input_ids, loss_mask, attention_mask = tokenize_chat_messages(
        _Tokenizer(),
        [
            {"role": "user", "content": "question"},
            {"role": "assistant", "content": "answer"},
        ],
        max_length=32,
    )

    assert input_ids.tolist() == [400, 501, 28, 502, 500, 501, 26, 502]
    assert loss_mask.tolist() == [0, 0, 0, 0, 0, 0, 1, 0]
    assert attention_mask.tolist() == [1] * 8


def test_sharegpt_reasoning_is_forwarded_to_official_template_field() -> None:
    normalized = normalize_messages(
        [{"from": "gpt", "value": "answer", "thinking": "reasoning"}]
    )

    assert normalized == [
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "reasoning",
        }
    ]


def test_structural_chat_tokens_in_content_are_rejected() -> None:
    with pytest.raises(ValueError, match="reserved chat token"):
        tokenize_chat_messages(
            _Tokenizer(),
            [{"role": "user", "content": "bad <|im_assistant|> token"}],
            max_length=32,
        )


def test_auto_last_turn_detects_inline_and_structured_reasoning() -> None:
    dataset._worker_state["last_turn_loss_only"] = "auto"

    assert dataset._last_turn_only(
        [{"role": "assistant", "content": "<think>reason</think>answer"}]
    )
    assert dataset._last_turn_only(
        [{"role": "assistant", "content": "answer", "reasoning_content": "reason"}]
    )
    assert not dataset._last_turn_only(
        [{"role": "assistant", "content": "<think></think>answer"}]
    )


def test_directory_dataset_fingerprint_changes_with_contents(tmp_path) -> None:
    data = tmp_path / "dataset"
    data.mkdir()
    sample = data / "train.jsonl"
    sample.write_text('{"conversations":[]}\n', encoding="utf-8")
    first = dataset._dataset_fingerprint(str(data))

    sample.write_text('{"conversations":[{}]}\n', encoding="utf-8")

    assert dataset._dataset_fingerprint(str(data)) != first
