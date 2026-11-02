# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Validate on every execution rank before admitting a constrained request."""

import asyncio
from collections import OrderedDict


class WorkerGrammarPreflight:
    def __init__(self, engine, tokenizer, config):
        from atom.structured_output import (
            structured_stop_token_ids,
            tokenizer_fingerprint,
        )

        self.engine = engine
        self.tokenizer = tokenizer
        hf = config.hf_config
        self.vocab_size = (getattr(hf, "text_config", None) or hf).vocab_size
        self.fingerprint = tokenizer_fingerprint(
            tokenizer, structured_stop_token_ids(config, tokenizer)
        )
        self.ready = OrderedDict()
        self.lock = asyncio.Lock()

    async def prepare(self, schema, tool_format):
        key = (schema, tool_format)
        async with self.lock:
            if key in self.ready:
                self.ready.move_to_end(key)
                return
            manager = getattr(self.engine, "core_mgr", None)
            rpc = getattr(manager, "collective_rpc", None)
            if rpc is None:
                raise ValueError(
                    "Structured output requires worker preflight RPC support"
                )
            replies = await asyncio.to_thread(
                rpc,
                "prepare_structured_output",
                args=(schema, tool_format, self.fingerprint, self.vocab_size),
                timeout=60.0,
            )
            if not replies or any(not reply.ok for reply in replies):
                errors = [reply.error for reply in replies if not reply.ok]
                raise ValueError(
                    "Worker JSON Schema preparation failed: " + str(errors)
                )
            expected = {"fingerprint": self.fingerprint, "vocab_size": self.vocab_size}
            if any(reply.value != expected for reply in replies):
                raise ValueError(
                    "Structured output configuration differs across workers"
                )
            self.ready[key] = None
            if len(self.ready) > 128:
                self.ready.popitem(last=False)
