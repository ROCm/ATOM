# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The tokenizer the engine and the API server encode prompts with.

Its own module, importing only transformers, so that a tokenize worker process
(`atom.entrypoints.openai.tokenize_pool`) loads exactly the same tokenizer
without importing the engine.
"""

import logging

from transformers import AutoTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger("atom")


def load_tokenizer(model: str, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(
        model, use_fast=True, trust_remote_code=trust_remote_code
    )
    probe = "Hello world 你好"
    if tokenizer.decode(tokenizer.encode(probe), skip_special_tokens=True) != probe:
        logger.warning(
            "AutoTokenizer round-trip failed, falling back to PreTrainedTokenizerFast"
        )
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model)
    return tokenizer
