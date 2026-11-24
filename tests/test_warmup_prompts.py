# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""ATOM_WARMUP_PROMPTS: random-token prompts the server completes before it serves."""

from types import SimpleNamespace


def test_warmup_prompts_go_through_the_engine(monkeypatch):
    from atom.entrypoints.openai import api_server

    monkeypatch.setenv("ATOM_WARMUP_PROMPTS", "2x8,5")
    calls = []

    def generate(prompts, params):
        engine.core_mgr.keep_finished_outputs = True
        calls.append(([len(p) for p in prompts], params.max_tokens, prompts))

    engine = SimpleNamespace(
        config=SimpleNamespace(kv_transfer_config=None),
        tokenizer=[None] * 100,
        core_mgr=SimpleNamespace(keep_finished_outputs=False),
        generate=generate,
    )
    api_server._run_warmup_prompts(engine)
    assert [call[:2] for call in calls] == [([8, 8], 1), ([5], 1)]
    assert all(0 <= t < 100 for call in calls for p in call[2] for t in p)
    assert engine.core_mgr.keep_finished_outputs is False


def test_a_moriio_producer_runs_no_warmup_prompts(monkeypatch):
    """MoRI-IO claims every finished prefill for a decoder, so a warmup prompt's
    blocks would stay held; a Mooncake producer frees them and runs the prompts.
    A config that names no connector builds MoRI-IO."""
    from atom.entrypoints.openai import api_server

    monkeypatch.setenv("ATOM_WARMUP_PROMPTS", "8")
    calls = []

    def engine(connector):
        kv = {"kv_role": "kv_producer"}
        if connector is not None:
            kv["kv_connector"] = connector
        return SimpleNamespace(
            config=SimpleNamespace(kv_transfer_config=kv),
            tokenizer=[None] * 100,
            core_mgr=SimpleNamespace(keep_finished_outputs=False),
            generate=lambda prompts, params: calls.append(connector),
        )

    api_server._run_warmup_prompts(engine("moriio"))
    api_server._run_warmup_prompts(engine(None))
    api_server._run_warmup_prompts(engine("mooncake"))
    assert calls == ["mooncake"]
