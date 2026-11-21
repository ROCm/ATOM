# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2026 Advanced Micro Devices, Inc.
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


def _extract_system_and_gen_prompt(
    tokenizer: PreTrainedTokenizer,
    **apply_chat_template_kwargs,
) -> tuple[list[int], list[int]]:
    """Detect system-prompt prefix and assistant generation-prompt token lengths."""
    token1 = _normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            add_generation_prompt=False,
            tokenize=True,
            **apply_chat_template_kwargs,
        )
    )
    token2 = _normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}] * 2,
            add_generation_prompt=False,
            tokenize=True,
            **apply_chat_template_kwargs,
        )
    )
    system_prompt = token1[: -(len(token2) - len(token1))]

    token3 = _normalize_token_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ""}],
            add_generation_prompt=True,
            tokenize=True,
            **apply_chat_template_kwargs,
        )
    )
    gen_prompt = token3[len(token1) :]
    return system_prompt, gen_prompt


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
        """Tokenize one turn and build its loss_mask."""
        inputs = self.tokenizer.apply_chat_template(
            [message],
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]

        # Strip system prompt prefix from non-first turns.
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

        # Ensure standard format.
        if messages and "from" in messages[0]:
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            messages = [
                {"role": role_map.get(m["from"], m["from"]), "content": m["value"]}
                for m in messages
            ]

        input_ids_parts, loss_mask_parts, attn_mask_parts = [], [], []
        for i, msg in enumerate(messages):
            ids, lm, am = self._process_single_message(i, msg)
            input_ids_parts.append(ids)
            loss_mask_parts.append(lm)
            attn_mask_parts.append(am)

        input_ids = torch.cat(input_ids_parts)
        loss_mask = torch.cat(loss_mask_parts)
        attention_mask = torch.cat(attn_mask_parts)
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
