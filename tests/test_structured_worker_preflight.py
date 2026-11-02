# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import sys
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from atom.structured_output import manager as so


@pytest.fixture
def worker(monkeypatch):
    from atom.model_engine import llm_engine

    class Head:
        num_embeddings = 8

    monkeypatch.setitem(
        sys.modules, "atom.model_ops.embed_head", NS(ParallelLMHead=Head)
    )
    tokenizer = NS(
        eos_token_id=2, backend_tokenizer=NS(to_str=lambda: '{"vocab":{"a":1}}')
    )
    loader = Mock(return_value=tokenizer)
    monkeypatch.setattr(llm_engine, "_load_tokenizer", loader)
    config = NS(model="test-model", trust_remote_code=False, stop_token_ids=[2])
    runner = NS(config=config, model=NS(modules=lambda: [Head()]))
    compile_schema = Mock()
    constructor = Mock(
        side_effect=lambda tok, vocab, stops: NS(
            vocab_size=vocab, compile=compile_schema
        )
    )
    monkeypatch.setattr(so, "StructuredOutput", constructor)
    return runner, so.tokenizer_fingerprint(tokenizer, [2]), loader, compile_schema


def test_worker_reuses_execution_compiler(worker):
    runner, fingerprint, loader, compile_schema = worker
    result = so.prepare_worker_grammar(runner, "{}", None, fingerprint, 8)
    controller = runner.structured_output
    so.prepare_worker_grammar(runner, '{"type":"string"}', None, fingerprint, 8)
    assert runner.structured_output is controller
    assert result == {"fingerprint": fingerprint, "vocab_size": 8}
    loader.assert_called_once_with("test-model", False)
    assert compile_schema.call_count == 2


@pytest.mark.parametrize("fingerprint,vocab", [("different", 8), (None, 16)])
def test_mismatch_fails_before_worker_compilation(worker, fingerprint, vocab):
    runner, actual, _loader, compile_schema = worker
    with pytest.raises(ValueError):
        so.prepare_worker_grammar(runner, "{}", None, fingerprint or actual, vocab)
    compile_schema.assert_not_called()
    assert not hasattr(runner, "structured_output")


def test_compile_failure_does_not_poison_next_request(worker):
    runner, fingerprint, _loader, compile_schema = worker
    compile_schema.side_effect = [ValueError("unsupported grammar"), None]
    with pytest.raises(ValueError, match="unsupported grammar"):
        so.prepare_worker_grammar(runner, "bad", None, fingerprint, 8)
    so.prepare_worker_grammar(runner, "{}", None, fingerprint, 8)
    assert compile_schema.call_count == 2
