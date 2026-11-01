# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import asyncio
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from atom.entrypoints.openai.structured_preflight import WorkerGrammarPreflight
from atom.model_engine.structured_output import tokenizer_fingerprint


def frontend(rpc):
    tokenizer = NS(
        eos_token_id=2, backend_tokenizer=NS(to_str=lambda: '{"vocab":{"a":1}}')
    )
    config = NS(hf_config=NS(vocab_size=8), stop_token_ids=[2])
    return WorkerGrammarPreflight(
        NS(core_mgr=NS(collective_rpc=rpc)), tokenizer, config
    )


def test_all_rank_success_is_cached_and_tool_format_is_part_of_key():
    rpc = Mock()
    preflight = frontend(rpc)
    value = {"fingerprint": preflight.fingerprint, "vocab_size": 8}
    rpc.return_value = [NS(ok=True, value=value), NS(ok=True, value=value)]

    async def run():
        await asyncio.gather(*(preflight.prepare("{}", None) for _ in range(4)))
        await preflight.prepare("{}", "tool-envelope")

    asyncio.run(run())
    assert rpc.call_count == 2
    assert rpc.call_args.kwargs["args"] == (
        "{}",
        "tool-envelope",
        preflight.fingerprint,
        8,
    )


@pytest.mark.parametrize("kind", ["one_rank_failed", "empty", "mismatch", "timeout"])
def test_failed_preparation_is_never_cached(kind):
    rpc = Mock()
    preflight = frontend(rpc)
    value = {"fingerprint": preflight.fingerprint, "vocab_size": 8}
    if kind == "one_rank_failed":
        rpc.return_value = [
            NS(ok=True, value=value),
            NS(ok=False, error="invalid grammar"),
        ]
    elif kind == "empty":
        rpc.return_value = []
    elif kind == "mismatch":
        rpc.return_value = [NS(ok=True, value={**value, "vocab_size": 16})]
    else:
        rpc.side_effect = TimeoutError("timeout")
    with pytest.raises((ValueError, TimeoutError)):
        asyncio.run(preflight.prepare("{}", None))
    assert not preflight.ready
    rpc.side_effect = None
    rpc.return_value = [NS(ok=True, value=value)]
    asyncio.run(preflight.prepare("{}", None))
    assert rpc.call_count == 2


def test_fingerprint_covers_decoder_and_stops():
    a = NS(backend_tokenizer=NS(to_str=lambda: '{"vocab":{"a":1},"decoder":"A"}'))
    b = NS(backend_tokenizer=NS(to_str=lambda: '{"decoder":"A","vocab":{"a":1}}'))
    c = NS(backend_tokenizer=NS(to_str=lambda: '{"vocab":{"a":1},"decoder":"B"}'))
    assert tokenizer_fingerprint(a, [2, 3]) == tokenizer_fingerprint(b, [3, 2])
    assert tokenizer_fingerprint(a, [2]) != tokenizer_fingerprint(c, [2])
    assert tokenizer_fingerprint(a, [2]) != tokenizer_fingerprint(a, [3])


def test_unstructured_request_never_uses_worker_rpc(monkeypatch):
    from atom.entrypoints.openai import api_server

    rpc = Mock(side_effect=AssertionError("ordinary request made an RPC"))
    monkeypatch.setattr(api_server, "engine", NS(core_mgr=NS(collective_rpc=rpc)))
    asyncio.run(api_server._configure_structured_output(NS(), None, {}))
    rpc.assert_not_called()


def test_worker_failure_is_value_error_before_schema_is_attached(monkeypatch):
    from atom.entrypoints.openai import api_server

    rpc = Mock(return_value=[NS(ok=False, error="invalid schema on worker")])
    preflight = frontend(rpc)
    monkeypatch.setattr(api_server, "engine", preflight.engine)
    monkeypatch.setattr(api_server, "tokenizer", preflight.tokenizer)
    monkeypatch.setattr(
        api_server, "_get_engine_config", lambda: NS(speculative_config=None)
    )
    monkeypatch.setattr(api_server, "synthetic_token_text", None)
    monkeypatch.setattr(api_server, "tool_call_parser_cls", None)
    monkeypatch.setattr(
        api_server._configure_structured_output, "preflight", preflight, raising=False
    )
    params = NS(ignore_eos=False, stop_strings=None)
    with pytest.raises(ValueError, match="Cannot compile JSON Schema"):
        asyncio.run(api_server._configure_structured_output(params, "{}", {}))
    assert not hasattr(params, "json_schema")
