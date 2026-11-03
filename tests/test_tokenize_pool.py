# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Prompt tokenization in worker processes (`tokenize_pool.TokenizePool`)."""

import asyncio
import random
from pathlib import Path

import pytest

tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

from tokenizers import (
    AddedToken,
    Tokenizer,
    models,
    pre_tokenizers,
    trainers,
)

from atom.entrypoints.openai.tokenize_pool import TokenizePool
from atom.model_engine.tokenizer_loader import load_tokenizer

WORDS = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "kappa",
    "omega",
    "sigma",
    "theta",
    "lambda",
    "zeta",
    "héllo",
    "你好",
]
SPECIALS = ["<|begin|>", "<|User|>", "<|Assistant|>"]


def _save_tokenizer(path, seed, vocab_size=300):
    rng = random.Random(seed)
    corpus = [" ".join(rng.choice(WORDS) for _ in range(40)) for _ in range(200)]
    backend = Tokenizer(models.BPE(unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.train_from_iterator(
        corpus,
        trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=["[UNK]"],
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    backend.add_special_tokens([AddedToken(t, special=True) for t in SPECIALS])
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=backend)
    tok.save_pretrained(str(path))
    return str(path)


def _prompts(n=8):
    rng = random.Random(3)
    out = []
    for i in range(n):
        p = "<|begin|>"
        for t in range(rng.randint(1, 6)):
            p += ("<|User|>" if t % 2 == 0 else "<|Assistant|>") + " ".join(
                rng.choice(WORDS) for _ in range(rng.randint(5, 3000))
            )
        out.append(p)
    return out


def test_pool_ids_match_the_server_tokenizer(tmp_path):
    path = _save_tokenizer(tmp_path / "tok", seed=0)
    reference = load_tokenizer(path)
    pool = TokenizePool(path, False, 2, reference)
    try:

        async def run():
            return await asyncio.gather(*(pool.encode(p) for p in _prompts()))

        results = asyncio.run(run())
        for p, ids in zip(_prompts(), results):
            assert ids.tolist() == reference.encode(p)
        assert pool.enabled
    finally:
        pool.shutdown()


def test_a_mismatch_turns_the_pool_off(tmp_path):
    pool_path = _save_tokenizer(tmp_path / "pool", seed=0)
    other = load_tokenizer(_save_tokenizer(tmp_path / "other", seed=1, vocab_size=280))
    pool = TokenizePool(pool_path, False, 1, other)
    try:
        prompt = _prompts(1)[0]
        ids = asyncio.run(pool.encode(prompt))
        assert ids.tolist() == other.encode(prompt)
        assert not pool.enabled
    finally:
        pool.shutdown()


def test_llm_engine_keeps_its_loader_name():
    from atom.model_engine import llm_engine

    assert llm_engine._load_tokenizer is load_tokenizer


def test_workers_exit_when_the_server_dies(tmp_path):
    """A server killed outright must not leave its tokenize workers behind."""
    import os
    import signal
    import subprocess
    import sys
    import time

    path = _save_tokenizer(tmp_path / "tok", seed=0)
    script = (
        "import asyncio, os, sys\n"
        "from atom.entrypoints.openai.tokenize_pool import TokenizePool\n"
        "from atom.model_engine.tokenizer_loader import load_tokenizer\n"
        f"pool = TokenizePool({path!r}, False, 2, load_tokenizer({path!r}))\n"
        "print(' '.join(str(p) for p in pool._pool._processes), flush=True)\n"
        "import time; time.sleep(600)\n"
    )
    server = subprocess.Popen(
        [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
    )
    workers = [int(p) for p in server.stdout.readline().split()]
    assert len(workers) == 2
    os.kill(server.pid, signal.SIGKILL)
    server.wait()
    deadline = time.time() + 15
    alive = workers
    while alive and time.time() < deadline:
        time.sleep(0.5)
        alive = [
            p
            for p in alive
            if os.path.exists(f"/proc/{p}")
            and "zombie" not in Path(f"/proc/{p}/status").read_text().lower()
        ]
    assert not alive, alive
