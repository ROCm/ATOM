# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""R3 through the Ray ATOM rollout: capture is switched on, and every result
path of ATOMRayServer keeps the engine's ``routed_experts``."""

from __future__ import annotations

import asyncio
import itertools
import queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from atomrl.engine.inference import atom_ray_server, vllm_http_engine
from atomrl.engine.inference.atom_ray_server import ATOMRayServer
from atomrl.trainer.rl_trainer import RLTrainer


def _routes(rows: int) -> np.ndarray:
    """ATOM's R3 payload: int16 ``[rows, num_layers, top_k]``."""
    return np.arange(rows * 2 * 2, dtype=np.int16).reshape(rows, 2, 2)


class Engine:
    """``LLMEngine.generate`` plus the pump's pieces; R3 attaches routes."""

    def __init__(self, with_routes: bool = True) -> None:
        self.with_routes = with_routes
        self._ids = itertools.count()
        self._finished: queue.Queue = queue.Queue()
        self.added: list[list[SimpleNamespace]] = []
        self.io_processor = SimpleNamespace(
            preprocess_fanout=self._fanout, postprocess=self._postprocess
        )
        self.core_mgr = SimpleNamespace(
            add_request=self._add, keep_finished_outputs=False
        )

    def _output(self, prompt_len: int, token_ids: list[int]) -> dict:
        out = {"text": "", "token_ids": token_ids, "logprobs": [-0.5] * len(token_ids)}
        if self.with_routes:
            out["routed_experts"] = _routes(prompt_len + len(token_ids) - 1)
        return out

    def generate(self, prompts, params, request_ids=None):
        # One params per prompt, or one for all; `n` completions of each prompt.
        params = params if isinstance(params, list) else [params] * len(prompts)
        return [
            self._output(len(p), [7, 8])
            for p, sp in zip(prompts, params)
            for _ in range(sp.n)
        ]

    def _fanout(self, prompt_ids, sampling_params, parent_request_id):
        return [
            SimpleNamespace(id=next(self._ids), prompt_len=len(prompt_ids))
            for _ in range(sampling_params.n)
        ]

    def _add(self, seqs) -> None:
        self.added.append(seqs)
        # Finish at once: the pump reads them on its next step().
        self._finished.put(list(seqs))

    def _postprocess(self, seqs):
        return {seq.id: self._output(seq.prompt_len, [9]) for seq in seqs}

    def step(self):
        return self._finished.get(timeout=5)


def _server(engine: Engine) -> ATOMRayServer:
    server = object.__new__(ATOMRayServer)
    server.engine = engine
    server._pump = None
    server._preemptions_seen = 0
    server._build_sampling_params = lambda params, prompt_length: SimpleNamespace(n=1)
    server._log_kv_pressure = lambda: None
    return server


def test_generate_keeps_routed_experts() -> None:
    server = _server(Engine())

    result = asyncio.run(server.generate([1, 2, 3], {}))

    np.testing.assert_array_equal(result["routed_experts"], _routes(4))
    assert result["token_ids"] == [7, 8]


def test_generate_batch_keeps_routed_experts_per_completion() -> None:
    server = _server(Engine())
    # The repeated prompt is one grouped request with n=2; the other is n=1.
    prompts = [[1, 2], [1, 2], [3]]

    results = asyncio.run(server.generate_batch(prompts, {}))

    assert [r["prompt_token_ids"] for r in results] == prompts
    for prompt, result in zip(prompts, results):
        np.testing.assert_array_equal(
            result["routed_experts"], _routes(len(prompt) + 2 - 1)
        )


def test_the_group_pump_keeps_routed_experts() -> None:
    server = _server(Engine())

    async def run():
        # generate_group starts the pump, and every request goes through it.
        group = await asyncio.wait_for(server.generate_group([1, 2], 2, {}), timeout=10)
        single = await asyncio.wait_for(server.generate([5], {}), timeout=10)
        return group, single

    group, single = asyncio.run(run())

    assert len(group) == 2
    for result in group:
        np.testing.assert_array_equal(result["routed_experts"], _routes(2))
    np.testing.assert_array_equal(single["routed_experts"], _routes(1))


def test_a_result_without_routes_has_no_routed_experts_key() -> None:
    """R3 off: the result is what it was before, with no key to misread."""
    server = _server(Engine(with_routes=False))

    single = asyncio.run(server.generate([1], {}))
    batch = asyncio.run(server.generate_batch([[1], [2]], {}))

    assert "routed_experts" not in single
    assert all("routed_experts" not in r for r in batch)


def test_assemble_takes_the_results_the_server_returns() -> None:
    """The server's dicts feed `_assemble_ray_rollout` as they are."""
    server = _server(Engine())
    results = asyncio.run(server.generate_batch([[1, 2], [3]], {}))

    class Tokenizer:
        pad_token_id = 0

    trainer = object.__new__(RLTrainer)
    trainer._tokenizer = Tokenizer()
    trainer._device = torch.device("cpu")
    trainer.config = SimpleNamespace(
        policy=SimpleNamespace(
            generation=SimpleNamespace(
                vllm_cfg=SimpleNamespace(calculate_log_probs=False)
            )
        ),
        moe=SimpleNamespace(r3=SimpleNamespace(enabled=True, rollout_replay=False)),
    )

    *_, ragged = trainer._assemble_ray_rollout(results)

    routing = ragged["rollout_routing"]
    assert [tuple(r.shape) for r in routing] == [(3, 2, 2), (2, 2, 2)]
    assert routing[0].dtype == torch.int16


def test_assemble_names_the_missing_routes() -> None:
    trainer = object.__new__(RLTrainer)

    class Tokenizer:
        pad_token_id = 0

    trainer._tokenizer = Tokenizer()
    trainer._device = torch.device("cpu")
    trainer.config = SimpleNamespace(
        policy=SimpleNamespace(
            generation=SimpleNamespace(
                vllm_cfg=SimpleNamespace(calculate_log_probs=False)
            )
        ),
        moe=SimpleNamespace(r3=SimpleNamespace(enabled=True, rollout_replay=False)),
    )

    with pytest.raises(RuntimeError, match="returned no routed_experts"):
        trainer._assemble_ray_rollout(
            [{"prompt_token_ids": [1], "token_ids": [2], "logprobs": None}]
        )


def _setup(monkeypatch, r3: SimpleNamespace, atom_extra=None) -> dict:
    """Run `_setup_ray_atom_rollout` against stand-ins; return the engine kwargs."""
    seen: dict = {}

    class Manager:
        num_replicas = 1

        def __init__(self, wg, model_name, engine_kwargs, **kwargs):
            seen.update(engine_kwargs)

        def create(self):
            pass

    monkeypatch.setattr(atom_ray_server, "ATOMReplicaManager", Manager)
    monkeypatch.setattr(vllm_http_engine, "VLLMHttpEngine", lambda *a, **k: object())

    trainer = object.__new__(RLTrainer)
    trainer._actor_wg = object()
    trainer.config = SimpleNamespace(
        seed=1,
        moe=SimpleNamespace(r3=r3),
        weight_sync=SimpleNamespace(backend="ipc"),
    )
    vcfg = SimpleNamespace(
        seed=None,
        max_model_len=1024,
        quantization="",
        gpu_memory_utilization=0.4,
        max_num_batched_tokens=512,
        max_num_seqs=8,
        enforce_eager=True,
        trust_remote_code=False,
        enable_chunked_prefill=True,
        sleep_level=2,
        enable_sleep_mode=False,
    )
    atom_cfg = SimpleNamespace(engine_kwargs=atom_extra or {})
    trainer._setup_ray_atom_rollout("model", vcfg, atom_cfg)
    return seen


@pytest.mark.parametrize(
    "r3",
    [
        SimpleNamespace(enabled=True, rollout_replay=False),
        SimpleNamespace(enabled=False, rollout_replay=True),
    ],
)
def test_r3_turns_on_route_capture_in_the_atom_engine(monkeypatch, r3) -> None:
    assert _setup(monkeypatch, r3)["enable_return_routed_experts"] is True


def test_r3_off_leaves_route_capture_out_of_the_engine_kwargs(monkeypatch) -> None:
    kwargs = _setup(monkeypatch, SimpleNamespace(enabled=False, rollout_replay=False))
    assert "enable_return_routed_experts" not in kwargs
