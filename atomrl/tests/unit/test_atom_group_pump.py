# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""AtomGroupPump and ATOMRayServer.generate_group against a fake ATOM engine."""

from __future__ import annotations

import asyncio
import itertools
import queue
from types import SimpleNamespace

import pytest
from atomrl.engine.inference.atom_group_pump import AtomGroupPump
from atomrl.engine.inference.atom_ray_server import ATOMRayServer


class FakeEngine:
    """Sequences finish when the test says so, in whatever order it says."""

    def __init__(self) -> None:
        self._ids = itertools.count()
        self.added: list[list[SimpleNamespace]] = []
        self._finished: queue.Queue = queue.Queue()
        self.offline_calls = 0
        self.io_processor = SimpleNamespace(
            preprocess_fanout=self._preprocess_fanout,
            postprocess=self._postprocess,
        )
        self.core_mgr = SimpleNamespace(
            add_request=self.added.append, keep_finished_outputs=False
        )

    def _preprocess_fanout(self, prompt_ids, sampling_params, parent_request_id):
        return [
            SimpleNamespace(
                id=next(self._ids), prompt=list(prompt_ids), request=parent_request_id
            )
            for _ in range(sampling_params.n)
        ]

    def _postprocess(self, seqs):
        return {
            seq.id: {"text": "", "token_ids": [seq.id], "logprobs": [-float(seq.id)]}
            for seq in seqs
        }

    def step(self):
        item = self._finished.get()
        if isinstance(item, BaseException):
            raise item
        return item

    def finish(self, *seqs) -> None:
        # As CoreManager since ROCm/ATOM#2912: finished sequences reach step()
        # only once a reader has asked for them.
        if self.core_mgr.keep_finished_outputs:
            self._finished.put(list(seqs))

    def fail(self, exc: BaseException) -> None:
        self._finished.put(exc)

    def generate(self, prompts, params, request_ids=None):
        self.offline_calls += 1
        return [{"text": "", "token_ids": [7], "logprobs": [-7.0]} for _ in prompts]


def test_concurrent_groups_get_their_own_outputs_in_fanout_order() -> None:
    engine = FakeEngine()
    pump = AtomGroupPump(engine)

    async def run():
        first = pump.submit([1, 2], SimpleNamespace(n=3), "a")
        second = pump.submit([3], SimpleNamespace(n=2), "b")
        a, b = engine.added
        engine.finish(b[1], a[2])
        engine.finish(a[0])
        engine.finish(b[0], a[1])
        return await asyncio.wait_for(asyncio.gather(first, second), timeout=5)

    first, second = asyncio.run(run())

    a, b = engine.added
    assert [out["token_ids"] for out in first] == [[seq.id] for seq in a]
    assert [out["token_ids"] for out in second] == [[seq.id] for seq in b]
    assert [seq.request for seq in a] == ["a"] * 3


def test_the_pump_asks_for_finished_sequences() -> None:
    """Without it an ATOM since #2912 drops every finished sequence, and no
    group ever comes back."""
    engine = FakeEngine()
    AtomGroupPump(engine)
    assert engine.core_mgr.keep_finished_outputs is True


def test_a_reader_failure_reaches_every_waiting_group_and_later_submits() -> None:
    engine = FakeEngine()
    pump = AtomGroupPump(engine)

    async def run():
        waiting = pump.submit([1], SimpleNamespace(n=2), "a")
        engine.fail(RuntimeError("engine core died"))
        with pytest.raises(RuntimeError, match="engine core died"):
            await asyncio.wait_for(waiting, timeout=5)
        with pytest.raises(RuntimeError, match="output reader has stopped"):
            pump.submit([2], SimpleNamespace(n=1), "b")

    asyncio.run(run())


def _server(engine: FakeEngine) -> ATOMRayServer:
    server = object.__new__(ATOMRayServer)
    server.engine = engine
    server._pump = None
    server._preemptions_seen = 0
    server._build_sampling_params = lambda params, prompt_length: SimpleNamespace(n=1)
    server._log_kv_pressure = lambda: None
    return server


def test_generate_group_moves_every_later_request_onto_the_pump() -> None:
    engine = FakeEngine()
    server = _server(engine)

    async def run():
        before = await server.generate_batch([[5], [5]], {})
        group = asyncio.ensure_future(server.generate_group([1, 2], 2, {}))
        batch = asyncio.ensure_future(server.generate_batch([[3], [3], [4]], {}))
        await asyncio.sleep(0.05)
        for seqs in engine.added:
            engine.finish(*seqs)
        return (
            before,
            await asyncio.wait_for(group, 5),
            await asyncio.wait_for(batch, 5),
        )

    before, group, batch = asyncio.run(run())

    assert (
        engine.offline_calls == 1
    ), "only the request before generate_group ran offline"
    assert [r["prompt_token_ids"] for r in group] == [[1, 2], [1, 2]]
    assert [r["prompt_token_ids"] for r in batch] == [[3], [3], [4]]
    assert [len(seqs) for seqs in engine.added] == [
        2,
        2,
        1,
    ], "the batch's repeats stay one fan-out"
    assert group[0]["logprobs"] == [-float(engine.added[0][0].id)]
    assert before[0]["token_ids"] == [7]
