# SPDX-License-Identifier: Apache-2.0
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
#
# Modified by Advanced Micro Devices, Inc.
# Source: https://github.com/verl-project/verl/blob/8357f1e0a7ce4c06dc970c2c637d49b33815f5cc/verl/utils/dataset/multiturn_sft_dataset.py

#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from verl's MultiTurnSFTDataset:
# https://github.com/volcengine/verl

"""Model-agnostic SFT dataset using HF chat templates for loss mask construction.

Tokenizes each conversation turn independently via ``apply_chat_template``,
marks assistant tokens with loss_mask=1, and concatenates into a single sequence.
Supports Parquet and JSONL inputs. Adapted from verl's MultiTurnSFTDataset.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

logger = logging.getLogger(__name__)


def _normalize_token_ids(tokenized_output) -> list[int]:
    """Normalize ``apply_chat_template`` output to a flat ``list[int]``.

    Handles Transformers 4/5 differences: the return value may be
    ``list[int]``, ``BatchEncoding``/dict with ``input_ids``, a tensor,
    a nested ``list[list[int]]``, or a tuple.
    """
    token_ids = tokenized_output
    if isinstance(tokenized_output, dict):
        if "input_ids" in tokenized_output:
            token_ids = tokenized_output["input_ids"]
    elif hasattr(tokenized_output, "input_ids"):
        token_ids = tokenized_output.input_ids

    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()

    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)

    if (
        isinstance(token_ids, list)
        and len(token_ids) == 1
        and isinstance(token_ids[0], (list, tuple))
    ):
        token_ids = list(token_ids[0])

    return [int(t.item() if hasattr(t, "item") else t) for t in token_ids]


_ROLE_MAP = {"human": "user", "gpt": "assistant", "system": "system"}
_RESERVED_CHAT_TOKENS = ("<|im_assistant|>", "<|im_middle|>", "<|im_end|>")


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize ShareGPT keys and preserve official-template metadata."""
    normalized = []
    for message in messages:
        converted = dict(message)
        if "role" not in converted:
            source_role = converted.pop("from", "")
            converted["role"] = _ROLE_MAP.get(source_role, source_role)
        if "content" not in converted:
            converted["content"] = converted.pop("value", None)
        reasoning = converted.get("reasoning_content")
        for field in ("thinking", "thinking_content", "reasoning"):
            candidate = converted.pop(field, None)
            if reasoning is None and candidate:
                reasoning = candidate
        if reasoning is not None:
            converted["reasoning_content"] = reasoning
        normalized.append(converted)
    return normalized


def _contains_reserved_chat_token(value: Any) -> bool:
    if isinstance(value, str):
        return any(token in value for token in _RESERVED_CHAT_TOKENS)
    if isinstance(value, dict):
        return any(_contains_reserved_chat_token(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_reserved_chat_token(item) for item in value)
    return False


def tokenize_chat_messages(
    tokenizer: PreTrainedTokenizer,
    messages: list[dict[str, Any]],
    max_length: int,
    *,
    last_turn_only: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Render one official HF template and mark its assistant-content spans."""
    messages = normalize_messages(messages)
    if _contains_reserved_chat_token(messages):
        raise ValueError("message content contains a reserved chat token")

    token_ids = _normalize_token_ids(
        tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=False,
            tokenize=True,
        )
    )
    assistant_id = tokenizer.convert_tokens_to_ids("<|im_assistant|>")
    middle_id = tokenizer.convert_tokens_to_ids("<|im_middle|>")
    end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    spans = []
    cursor = 0
    while cursor < len(token_ids):
        if token_ids[cursor] != assistant_id:
            cursor += 1
            continue
        try:
            content_start = token_ids.index(middle_id, cursor + 1) + 1
            content_end = token_ids.index(end_id, content_start)
        except ValueError as error:
            raise ValueError(
                "malformed assistant segment in HF chat template"
            ) from error
        spans.append((content_start, content_end))
        cursor = content_end + 1
    if last_turn_only and spans:
        spans = spans[-1:]

    input_ids = torch.tensor(token_ids[:max_length], dtype=torch.long)
    loss_mask = torch.zeros(len(input_ids), dtype=torch.long)
    for start, end in spans:
        loss_mask[start : min(end, max_length)] = 1
    attention_mask = torch.ones(len(input_ids), dtype=torch.long)
    return input_ids, loss_mask, attention_mask


def _extract_system_and_gen_prompt(
    tokenizer: PreTrainedTokenizer,
    **apply_chat_template_kwargs,
) -> tuple[list[int], list[int]]:
    """Detect system-prompt prefix and assistant generation-prompt token lengths."""
    empty_user = {"role": "user", "content": ""}
    token1 = _normalize_token_ids(
        tokenizer.apply_chat_template(
            [empty_user],
            add_generation_prompt=False,
            tokenize=True,
            **apply_chat_template_kwargs,
        )
    )
    token2 = _normalize_token_ids(
        tokenizer.apply_chat_template(
            [empty_user, empty_user],
            add_generation_prompt=False,
            tokenize=True,
            **apply_chat_template_kwargs,
        )
    )
    system_prompt = token1[: -(len(token2) - len(token1))]
    token3 = _normalize_token_ids(
        tokenizer.apply_chat_template(
            [empty_user],
            add_generation_prompt=True,
            tokenize=True,
            **apply_chat_template_kwargs,
        )
    )
    return system_prompt, token3[len(token1) :]


class SFTDataset(Dataset):
    """Single/multi-turn SFT dataset with model-agnostic chat-template loss masking.

    Args:
        data_files: Path(s) to Parquet or JSONL files.
        tokenizer: HuggingFace tokenizer (or path string).
        max_length: Maximum sequence length after padding/truncation.
        messages_key: Column name for the conversation messages list.
        pad_mode: ``"right"`` for fixed-length padding, ``"no_padding"`` for variable.
        truncation: ``"right"``, ``"left"``, or ``"error"``.
        shuffle: Whether to shuffle samples on load.
        seed: Random seed for shuffling.
        max_samples: Limit the number of samples (-1 for all).
    """

    def __init__(
        self,
        data_files: str | list[str],
        tokenizer: PreTrainedTokenizer | str,
        max_length: int = 2048,
        messages_key: str = "messages",
        pad_mode: str = "right",
        truncation: str = "right",
        shuffle: bool = False,
        seed: int = 42,
        max_samples: int = -1,
    ) -> None:
        if isinstance(tokenizer, str):
            tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.messages_key = messages_key
        self.pad_mode = pad_mode
        self.truncation = truncation

        if isinstance(data_files, str):
            data_files = [data_files]

        self.system_prompt, self.gen_prompt = _extract_system_and_gen_prompt(tokenizer)
        self._load_data(data_files, shuffle, seed, max_samples)

    def _load_data(
        self,
        data_files: list[str],
        shuffle: bool,
        seed: int,
        max_samples: int,
    ) -> None:
        import numpy as np
        import pandas as pd

        frames = []
        for path in data_files:
            if path.endswith(".parquet"):
                frames.append(pd.read_parquet(path, dtype_backend="pyarrow"))
            elif path.endswith((".jsonl", ".json")):
                frames.append(pd.read_json(path, lines=path.endswith(".jsonl")))
            else:
                frames.append(pd.read_parquet(path, dtype_backend="pyarrow"))
        df = pd.concat(frames, ignore_index=True)

        total = len(df)
        if 0 < max_samples < total:
            rng = np.random.default_rng(seed)
            indices = (
                rng.choice(total, size=max_samples, replace=False)
                if shuffle
                else np.arange(max_samples)
            )
            df = df.iloc[indices.tolist()]
        elif shuffle:
            df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

        self.messages_list = df[self.messages_key].tolist()
        logger.info(
            "SFTDataset: loaded %d samples from %d files",
            len(self.messages_list),
            len(data_files),
        )

    def __len__(self) -> int:
        return len(self.messages_list)

    def _process_single_message(
        self,
        index: int,
        message: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Tokenize one turn and build its loss mask."""
        inputs = self.tokenizer.apply_chat_template(
            [message],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        if index != 0 and message.get("role") != "system":
            input_ids = input_ids[len(self.system_prompt) :]
            attention_mask = attention_mask[len(self.system_prompt) :]
        if message.get("role") == "assistant":
            loss_mask = torch.ones_like(attention_mask)
            loss_mask[: len(self.gen_prompt)] = 0
        else:
            loss_mask = torch.zeros_like(attention_mask)
        return input_ids, loss_mask, attention_mask

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        messages = self.messages_list[item]
        if isinstance(messages, str):
            import json

            messages = json.loads(messages)

        if messages and "from" in messages[0]:
            messages = [
                {
                    "role": _ROLE_MAP.get(message["from"], message["from"]),
                    "content": message["value"],
                }
                for message in messages
            ]
        input_ids_parts, loss_mask_parts, attention_mask_parts = [], [], []
        for index, message in enumerate(messages):
            input_ids, loss_mask, attention_mask = self._process_single_message(
                index, message
            )
            input_ids_parts.append(input_ids)
            loss_mask_parts.append(loss_mask)
            attention_mask_parts.append(attention_mask)
        input_ids = torch.cat(input_ids_parts)
        loss_mask = torch.cat(loss_mask_parts)
        attention_mask = torch.cat(attention_mask_parts)
        position_ids = torch.arange(len(input_ids), dtype=torch.long)

        seq_len = input_ids.shape[0]

        if self.pad_mode == "right":
            if seq_len < self.max_length:
                pad_id = self.tokenizer.pad_token_id or 0
                pad_len = self.max_length - seq_len
                input_ids = torch.cat(
                    [input_ids, torch.full((pad_len,), pad_id, dtype=input_ids.dtype)]
                )
                attention_mask = torch.cat(
                    [attention_mask, torch.zeros(pad_len, dtype=attention_mask.dtype)]
                )
                loss_mask = torch.cat(
                    [loss_mask, torch.zeros(pad_len, dtype=loss_mask.dtype)]
                )
                position_ids = F.pad(position_ids, (0, pad_len), value=0)
            elif seq_len > self.max_length:
                if self.truncation == "right":
                    input_ids = input_ids[: self.max_length]
                    attention_mask = attention_mask[: self.max_length]
                    loss_mask = loss_mask[: self.max_length]
                    position_ids = position_ids[: self.max_length]
                elif self.truncation == "left":
                    input_ids = input_ids[-self.max_length :]
                    attention_mask = attention_mask[-self.max_length :]
                    loss_mask = loss_mask[-self.max_length :]
                    position_ids = position_ids[-self.max_length :]
                else:
                    raise ValueError(
                        f"Sequence length {seq_len} exceeds max_length {self.max_length}"
                    )
        elif self.pad_mode == "no_padding":
            if seq_len > self.max_length:
                if self.truncation == "error":
                    raise ValueError(
                        f"Sequence length {seq_len} exceeds max_length {self.max_length}"
                    )
                input_ids = input_ids[: self.max_length]
                attention_mask = attention_mask[: self.max_length]
                loss_mask = loss_mask[: self.max_length]
                position_ids = position_ids[: self.max_length]
        else:
            raise ValueError(f"Unknown pad_mode: {self.pad_mode}")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
            "position_ids": position_ids,
        }
