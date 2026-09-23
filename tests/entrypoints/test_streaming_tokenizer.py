# SPDX-License-Identifier: MIT
"""Merged and single-token decoding must agree without model downloads or GPUs."""

import asyncio
from itertools import cycle

import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

from atom.entrypoints.openai.streaming_dispatch import (
    IncrementalStreamDetokenizer,
    StreamBatchDispatcher,
    StreamOutputCollector,
)

_TEXTS = (
    "你好，世界！ 日本語 한국어 العربية हिन्दी русский",
    "👩🏽‍💻 👨‍👩‍👧‍👦 🙂🌍",
    "café e\u0301 naïve résumé",
    "a  b\t c \n\n trailing  ",
    "def f(x):\n    return x + 1  # comment\n",
    "Hello <special> world! <special>",
)


@pytest.fixture(scope="module")
def tokenizer():
    # Exercise the real HF wrapper and byte-level BPE backend entirely offline.
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.train_from_iterator(
        _TEXTS,
        trainers.BpeTrainer(
            vocab_size=320,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<special>"],
            show_progress=False,
        ),
    )
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        additional_special_tokens=["<special>"],
        clean_up_tokenization_spaces=False,
    )


@pytest.mark.parametrize("text", _TEXTS)
@pytest.mark.parametrize(
    "drain_pattern", ((1,), (2,), (3,), (8,), (4096,), (1, 1, 1, 8, 2, 1))
)
def test_merged_tokens_match_single_token_updates(tokenizer, text, drain_pattern):
    ids = tokenizer.encode(text, add_special_tokens=False)
    reference = IncrementalStreamDetokenizer(tokenizer)
    expected = "".join(
        reference.update([token], finished=i == len(ids) - 1)
        for i, token in enumerate(ids)
    )
    assert expected == tokenizer.decode(ids, skip_special_tokens=True)

    async def run():
        dispatcher = StreamBatchDispatcher(tokenizer)
        collector = StreamOutputCollector("merged-tokenizer")
        state = dispatcher.new_state()
        loop = asyncio.get_running_loop()
        intervals = cycle(drain_pattern)
        remaining = next(intervals)
        texts, delivered_ids, terminal = [], [], 0
        for i, token in enumerate(ids):
            finished = i == len(ids) - 1
            dispatcher.enqueue(
                loop=loop,
                collector=collector,
                state=state,
                chunk={"token_ids": [token], "finished": finished},
            )
            dispatcher.flush()
            await asyncio.sleep(0)  # Deliver without forcing the consumer to read.
            remaining -= 1
            if remaining == 0 or finished:
                chunk = await collector.get()
                texts.append(chunk["text"])
                delivered_ids.extend(chunk["token_ids"])
                terminal += bool(chunk["finished"])
                remaining = next(intervals)
        return "".join(texts), delivered_ids, terminal

    assert asyncio.run(run()) == (expected, ids, 1)
