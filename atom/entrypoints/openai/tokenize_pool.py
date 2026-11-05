# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Prompt tokenization in worker processes (ATOM_TOKENIZE_PROCESSES).

The API server is one Python process. Encoding an agentic prompt of ~100k
tokens takes ~60 ms of CPU, and the server's preprocessing threads do not run
it in parallel; at ~100 requests/s every request waits seconds in the
preprocessing executor before the engine sees it. Each worker of this pool
loads the server's tokenizer (`tokenizer_loader.load_tokenizer`) and encodes
whole prompts; the server gets the ids back as int32 bytes and builds the
request from them as before (`LLMEngine.preprocess` takes ids as well as text).

The first few results are compared with the server's own tokenizer, and a
difference turns the pool off.

Given `chat_settings`, each worker can also prepare whole chat completions
(`chat_offload.prepare_chat_in_worker`, ATOM_API_OFFLOAD_PREPROCESS).
"""

import array
import asyncio
import concurrent.futures
import logging
import multiprocessing
import os
import threading
import time

logger = logging.getLogger("atom")

# Results checked against the server's tokenizer before the pool is trusted.
_VERIFY_FIRST = 4

_tokenizer = None


def _exit_with(parent: int) -> None:
    # A server killed outright (SIGKILL) leaves its workers waiting on a queue
    # nobody writes to; they would otherwise outlive it, ~1 GB each.
    while os.getppid() == parent:
        time.sleep(1.0)
    os._exit(0)


def _init_worker(
    model: str, trust_remote_code: bool, parent: int, chat_settings=None
) -> None:
    global _tokenizer
    threading.Thread(target=_exit_with, args=(parent,), daemon=True).start()
    from atom.model_engine.tokenizer_loader import load_tokenizer

    _tokenizer = load_tokenizer(model, trust_remote_code)
    if chat_settings is not None:
        from atom.entrypoints.openai.chat_offload import init_chat_worker

        init_chat_worker(_tokenizer, model, chat_settings)


def _encode(prompt: str) -> bytes:
    return array.array("i", _tokenizer.encode(prompt)).tobytes()


class TokenizePool:
    """`await encode(prompt)` -> `array("i")` ids, computed in a worker process."""

    def __init__(
        self,
        model: str,
        trust_remote_code: bool,
        processes: int,
        reference,
        chat_settings=None,
    ):
        # `chat_settings` (`chat_offload.ChatPrepareSettings`): also build the
        # state `chat_offload.prepare_chat_in_worker` renders prompts with.
        self._pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=processes,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(model, trust_remote_code, os.getpid(), chat_settings),
        )
        self._reference = reference
        self._verify_left = _VERIFY_FIRST
        self.enabled = True
        # Start every worker (a tokenizer load each) now, not on the first requests.
        list(self._pool.map(_encode, ["warm up"] * processes))

    async def encode(self, prompt: str) -> array.array:
        loop = asyncio.get_running_loop()
        ids = array.array("i")
        ids.frombytes(await loop.run_in_executor(self._pool, _encode, prompt))
        if self._verify_left > 0:
            self._verify_left -= 1
            ref = await loop.run_in_executor(None, self._reference.encode, prompt)
            if ids.tolist() != ref:
                logger.warning(
                    "tokenize pool: ids differ from the server tokenizer; "
                    "turning the pool off"
                )
                self.enabled = False
                return array.array("i", ref)
        return ids

    async def run(self, fn, *args):
        """`fn(*args)` in a worker process; `fn` must be a module-level function."""
        return await asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
