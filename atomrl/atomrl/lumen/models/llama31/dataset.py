# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Framework-agnostic pretraining dataset for LLaMA 3.1.

Provides :class:`PretrainTextDataset` that reads raw text files (one document
per line, or jsonl with a ``"text"`` field) and produces fixed-length token
sequences for causal language-model pretraining.

No dependency on Megatron, HuggingFace Transformers, or any training
framework — only PyTorch and the Python standard library.
"""

import json
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset

__all__ = ["PretrainTextDataset", "check_sample_budget"]

logger = logging.getLogger(__name__)


class PretrainTextDataset(Dataset):
    """Pretraining dataset that concatenates tokenized documents and chunks
    into fixed-length sequences.

    Supports two input formats:

    - **plain text** (``.txt``): one document per line.
    - **jsonl** (``.jsonl`` / ``.json``): each line is a JSON object with a
      ``"text"`` field.

    Tokens from all documents are concatenated with EOS separators, then
    split into contiguous chunks of ``seq_length + 1`` tokens (the extra
    token provides the label for the last position).

    Args:
        data_path: Path to a ``.txt``, ``.jsonl``, or ``.json`` file.
        seq_length: Context window (number of input tokens per sample).
        tokenizer: Any object with ``encode(text) -> list[int]`` and
            ``eos_token_id``, **or** a Megatron tokenizer with
            ``tokenize()`` / ``eod``.
        is_hf_tokenizer: ``True`` for HuggingFace-style tokenizers.
        max_samples: If set, cap ``__len__`` at this value.
        rank: Zero-based data-parallel rank. Each rank tokenizes only its
            assigned input lines instead of duplicating preprocessing.
        world_size: Number of data-parallel dataset shards.
        allow_repeat: Serve ``max_samples`` samples even when this shard holds
            fewer, wrapping back to chunk 0. Off by default so a dataset never
            silently trains on repeated data. Callers that turn it on own
            telling the user: the class does not warn, because it is built once
            per rank and would say it once per rank.
    """

    def __init__(
        self,
        data_path: str | None,
        seq_length: int,
        tokenizer,
        is_hf_tokenizer: bool = False,
        max_samples: int | None = None,
        rank: int = 0,
        world_size: int = 1,
        allow_repeat: bool = False,
    ):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError(
                f"Invalid dataset shard rank={rank}, world_size={world_size}"
            )
        self._allow_repeat = allow_repeat
        self.seq_length = seq_length
        self.tokenizer = tokenizer
        self.is_hf_tokenizer = is_hf_tokenizer
        self.rank = rank
        self.world_size = world_size
        self._chunks: list[list[int]] = []

        if data_path is None:
            self._max_samples = max_samples or 0
            return

        chunk_len = seq_length + 1
        # Stop reading once the requested sample count is covered: a pretraining
        # corpus can be far larger than a run needs, and tokenizing all of it up
        # front dominates startup time.
        token_budget = max_samples * chunk_len if max_samples else None
        all_ids = self._load_and_tokenize(data_path, token_budget)

        n_chunks = len(all_ids) // chunk_len
        for i in range(n_chunks):
            self._chunks.append(all_ids[i * chunk_len : (i + 1) * chunk_len])

        logger.info(
            "Built %d pretraining samples (seq_length=%d) from %s " "(%d tokens total)",
            len(self._chunks),
            seq_length,
            data_path,
            len(all_ids),
        )

        self._max_samples = max_samples

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    @property
    def n_samples_read(self) -> int:
        """Distinct samples this shard built, before ``max_samples`` or repeat.

        ``__len__`` reports what the sampler is served, which under
        ``allow_repeat`` says nothing about how much distinct data there is.

        Reading stops once ``max_samples`` worth of tokens are in hand, so this
        is samples read, not corpus size: a corpus at least as long as the
        budget yields ``n_samples_read >= max_samples``, and anything less
        means the corpus ran out.
        """
        return len(self._chunks)

    def __len__(self) -> int:
        n = len(self._chunks)
        if self._max_samples is None:
            return n
        if self._allow_repeat and n:
            return self._max_samples
        return min(n, self._max_samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        idx = idx % len(self._chunks)
        chunk = self._chunks[idx]
        return {
            "input_ids": torch.LongTensor(chunk[:-1]),
            "labels": torch.LongTensor(chunk[1:]),
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> list[int]:
        if self.is_hf_tokenizer:
            return self.tokenizer.encode(text, add_special_tokens=False)
        return self.tokenizer.tokenize(text)

    def _get_eos_id(self) -> int:
        if self.is_hf_tokenizer:
            return self.tokenizer.eos_token_id
        return self.tokenizer.eod

    def _load_and_tokenize(
        self, data_path: str, token_budget: int | None = None
    ) -> list[int]:
        """Read documents, tokenize, and concatenate with EOS separators.

        ``token_budget`` stops the scan early once enough tokens are collected
        to build the requested number of samples.
        """
        path = Path(data_path)
        eos_id = self._get_eos_id()
        is_json = path.suffix in (".jsonl", ".json")
        all_ids: list[int] = []

        with open(path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if line_idx % self.world_size != self.rank:
                    continue
                line = line.strip()
                if not line:
                    continue
                text = json.loads(line).get("text", "") if is_json else line
                if not text:
                    continue
                all_ids.extend(self._tokenize(text))
                all_ids.append(eos_id)
                if token_budget is not None and len(all_ids) >= token_budget:
                    break

        return all_ids


def check_sample_budget(
    available: int, requested: int, allow_repeat: bool
) -> str | None:
    """Settle a training corpus too short for the requested run before step 0.

    Returns the message to report when the run proceeds on repeated data, and
    ``None`` when there is nothing to say. Raises when the shortfall is real
    and repeating was not asked for: a corpus that runs out partway through
    surfaces as a ``StopIteration`` far from its cause, which reads as a crash
    rather than the configuration problem it is.

    ``available`` comes from :attr:`PretrainTextDataset.n_samples_read`, which
    undercounts a sufficient corpus but never overcounts a short one, so this
    never reports a corpus as short when it is not.

    The remedies name Megatron flags even though this module imports nothing
    from Megatron; the flag name is the actionable half of the message and
    every caller reaches here from the same CLI.
    """
    if not available or requested <= available:
        return None

    if not allow_repeat:
        raise ValueError(
            f"The training corpus holds {available} samples but --train-iters and "
            f"--global-batch-size ask for {requested}. Supply more data, shorten "
            f"the run, or pass --lumen-repeat-corpus to train on repeated data."
        )

    return (
        f"training corpus holds {available} samples but {requested} were requested; "
        f"--lumen-repeat-corpus is set, so the data repeats "
        f"{requested / available:.2f} times"
    )
