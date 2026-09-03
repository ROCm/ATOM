# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""The dense load path is split into a KV-bytes leg and a completion report.

These tests pin the seam that lets a hybrid subclass fuse the KV leg with a
second leg under exactly one completion: `_load_kv_bytes` must return the
verdict and report nothing, `_finish_load` must be the only thing that moves a
completion id and the only thing that drops the lookup pin.
"""

from types import SimpleNamespace

import pytest

from atom.kv_transfer.disaggregation.types import LoadOperationId
from atom.kv_transfer.offload.dense.connector import DenseOffloadConnector
from atom.kv_transfer.offload.metadata import LMCacheReqMeta, LoadSpec


def _config(role="kv_consumer"):
    return SimpleNamespace(
        kv_transfer_config={"kv_role": role},
        kv_cache_block_size=4,
        decode_context_parallel_size=2,
        tensor_parallel_size=1,
    )


class _Engine:
    """Minimal LMCache engine stand-in: records unpins, fakes retrieve()."""

    gpu_connector = None

    def __init__(self, hit=True):
        self._hit = hit
        self.unpinned = []

    def retrieve(self, _tokens, *, mask, **_kwargs):
        return mask.clone().fill_(self._hit)

    def lookup_unpin(self, lookup_id):
        self.unpinned.append(lookup_id)


def _worker(*, hit=True, chunk_size=8):
    worker = DenseOffloadConnector(_config())
    worker.chunk_size = chunk_size
    worker._engine = _Engine(hit=hit)
    return worker


def _req(req_id, *, hbm, lmc, generation=1, n_tokens=8):
    return LMCacheReqMeta(
        req_id=req_id,
        token_ids=list(range(n_tokens)),
        block_ids=[0],
        load_spec=LoadSpec(
            hbm_cached_tokens=hbm,
            lmcache_cached_tokens=lmc,
            can_load=True,
        ),
        load_operation=LoadOperationId(req_id=req_id, generation=generation),
    )


@pytest.fixture
def worker_factory():
    made = []

    def make(**kwargs):
        worker = _worker(**kwargs)
        made.append(worker)
        return worker

    yield make
    for worker in made:
        worker._save_executor.shutdown(wait=True)
        worker._load_executor.shutdown(wait=True)


# -- the three exit paths, end to end --------------------------------------
@pytest.mark.parametrize(
    "case,hbm,lmc,hit,expect_ok",
    [
        ("already_in_hbm", 8, 8, True, True),
        ("unaligned_hbm", 4, 8, True, False),
        ("normal_hit", 0, 8, True, True),
        ("normal_miss", 0, 8, False, False),
    ],
)
def test_exit_paths_report_one_completion_and_unpin_once(
    worker_factory, case, hbm, lmc, hit, expect_ok
):
    worker = worker_factory(hit=hit)
    req = _req(61, hbm=hbm, lmc=lmc)

    worker._do_load_req(req)

    result = worker.get_finished()
    operation = req.load_operation
    if expect_ok:
        assert result.finished_loading == {operation}
        assert result.failed_loading == set()
    else:
        assert result.finished_loading == set()
        assert result.failed_loading == {operation}
    assert worker._engine.unpinned == ["61"], case


@pytest.mark.parametrize(
    "hbm,lmc,hit,expected",
    [
        (8, 8, True, True),
        (9, 16, True, False),
        (0, 8, True, True),
        (0, 8, False, False),
    ],
)
def test_load_kv_bytes_returns_verdict_and_reports_nothing(
    worker_factory, hbm, lmc, hit, expected
):
    worker = worker_factory(hit=hit)
    req = _req(62, hbm=hbm, lmc=lmc, n_tokens=16)

    assert worker._load_kv_bytes(req) is expected

    # No completion, no unpin: the leg is silent by construction.
    assert worker._done_load == set()
    assert worker._failed_load == set()
    assert worker._engine.unpinned == []


@pytest.mark.parametrize("ok,attr", [(True, "_done_load"), (False, "_failed_load")])
def test_finish_load_is_what_moves_the_id(worker_factory, ok, attr):
    worker = worker_factory()
    req = _req(63, hbm=0, lmc=8)

    worker._finish_load(req, ok)

    other = "_failed_load" if ok else "_done_load"
    assert getattr(worker, attr) == {req.load_operation}
    assert getattr(worker, other) == set()
    assert worker._engine.unpinned == ["63"]


def test_completion_id_falls_back_to_req_id(worker_factory):
    """No load_operation armed -> the bare req_id is the completion id."""

    worker = worker_factory()
    req = LMCacheReqMeta(
        req_id=64,
        token_ids=list(range(8)),
        block_ids=[0],
        load_spec=LoadSpec(hbm_cached_tokens=0, lmcache_cached_tokens=8, can_load=True),
    )

    worker._do_load_req(req)

    assert worker.get_finished().finished_loading == {64}


def test_do_load_req_is_pure_composition(worker_factory, monkeypatch):
    worker = worker_factory()
    req = _req(65, hbm=0, lmc=8)
    calls = []

    monkeypatch.setattr(
        worker, "_load_kv_bytes", lambda r: calls.append(("bytes", r)) or True
    )
    monkeypatch.setattr(
        worker, "_finish_load", lambda r, ok: calls.append(("finish", r, ok))
    )

    worker._do_load_req(req)

    assert calls == [("bytes", req), ("finish", req, True)]
