# SPDX-License-Identifier: MIT
"""CPU checks for OFFLOAD_PROFILE missing-versus-zero timing semantics."""

from __future__ import annotations

import logging
import math
import threading
from types import SimpleNamespace

import pytest
import torch

from atom.kv_transfer.offload._offload_common import OffloadWorkerMixin
from atom.kv_transfer.offload.dense.connector import DenseOffloadConnector
from atom.kv_transfer.offload.hybrid.dsv4.connector import DSV4OffloadConnector

TIMING_FIELDS = ("pack_ms", "copy_ms", "sync_ms", "transfer_ms", "effective_gbps")
COUNT_STATS = {
    "stats_available": 1,
    "counts_available": 1,
    "transfer_succeeded": 1,
    "chunks": 1,
    "groups": 1,
    "max_chunk_bytes": 16,
    "max_group_bytes": 16,
    "gpu_staging_chunk_bytes": 16,
    "gpu_staging_buffer_chunks": 1,
    "gpu_staging_buffer_bytes": 16,
    "total_bytes": 16,
    "producer_fenced": 1,
}


class _GPUConnector:
    def __init__(self, stats):
        self.stats = dict(stats)

    def reset_transfer_stats(self):
        pass

    def last_transfer_stats(self):
        return dict(self.stats)


class _Engine:
    def __init__(self, stats):
        self.gpu_connector = _GPUConnector(stats)

    def retrieve(self, tokens, **_kwargs):
        return torch.ones(int(tokens.numel()), dtype=torch.bool)

    def store(self, _tokens, **_kwargs):
        pass

    def lookup_unpin(self, _req_id):
        pass


def _dense_worker(stats):
    worker = DenseOffloadConnector.__new__(DenseOffloadConnector)
    worker.chunk_size = 4
    worker._rank = 0
    worker._engine = _Engine(stats)
    worker._lock = threading.Lock()
    worker._done_load = set()
    worker._failed_load = set()
    worker._failed_load_blocks = set()
    worker._done_save = set()
    worker._connector_completions = set()
    worker._early_release = False
    return worker


def _dsv4_worker(stats):
    worker = DSV4OffloadConnector.__new__(DSV4OffloadConnector)
    worker.chunk_size = 4
    worker._rank = 0
    worker._engine = _Engine(stats)
    return worker


def _load_req(req_id):
    return SimpleNamespace(
        req_id=req_id,
        token_ids=[1, 2, 3, 4],
        block_ids=[0],
        load_spec=SimpleNamespace(hbm_cached_tokens=0, lmcache_cached_tokens=4),
        load_operation=None,
    )


def _save_req(req_id):
    return SimpleNamespace(
        req_id=req_id,
        token_ids=[1, 2, 3, 4],
        block_ids=[0],
        save_spec=SimpleNamespace(skip_leading_tokens=0),
        save_operation=None,
        is_last_prefill=True,
    )


def _profile_message(caplog, marker):
    messages = [record.getMessage() for record in caplog.records if marker in record.getMessage()]
    assert len(messages) == 1
    return messages[0]


def _assert_timing_fields(message, value):
    for field in TIMING_FIELDS:
        assert f"{field}={value}" in message


def test_profile_stat_distinguishes_unmeasured_from_measured_zero():
    for field in TIMING_FIELDS:
        assert math.isnan(OffloadWorkerMixin._profile_transfer_stat({}, field))
        assert OffloadWorkerMixin._profile_transfer_stat({field: 0.0}, field) == 0.0


@pytest.mark.parametrize(
    ("stats", "expected"),
    [(COUNT_STATS, "nan"), (COUNT_STATS | dict.fromkeys(TIMING_FIELDS, 0.0), "0.00")],
    ids=["unmeasured", "measured-zero"],
)
def test_dense_load_profile_preserves_unmeasured_timing_state(
    monkeypatch, caplog, stats, expected
):
    monkeypatch.setenv("OFFLOAD_PROFILE", "1")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="atom"):
        _dense_worker(stats)._do_load_req(_load_req("dense-load"))

    message = _profile_message(caplog, "[OFFLOAD-LOAD-PROF]")
    _assert_timing_fields(message, expected)
    assert "retrieve_ms=nan" not in message
    assert "total_ms=nan" not in message
    assert "total_bytes=16" in message


@pytest.mark.parametrize(
    ("stats", "expected"),
    [(COUNT_STATS, "nan"), (COUNT_STATS | dict.fromkeys(TIMING_FIELDS, 0.0), "0.00")],
    ids=["unmeasured", "measured-zero"],
)
def test_dense_save_profile_preserves_unmeasured_timing_state(
    monkeypatch, caplog, stats, expected
):
    monkeypatch.setenv("OFFLOAD_PROFILE", "1")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="atom"):
        _dense_worker(stats)._do_save_req(_save_req("dense-save"))

    message = _profile_message(caplog, "[OFFLOAD-SAVE-PROF]")
    _assert_timing_fields(message, expected)
    assert "store_ms=nan" not in message
    assert "total_ms=nan" not in message
    assert "producer_fenced=1" in message


@pytest.mark.parametrize(
    ("stats", "expected"),
    [(COUNT_STATS, "nan"), (COUNT_STATS | dict.fromkeys(TIMING_FIELDS, 0.0), "0.00")],
    ids=["unmeasured", "measured-zero"],
)
def test_dsv4_load_profile_uses_the_same_unmeasured_contract(
    monkeypatch, caplog, stats, expected
):
    monkeypatch.setenv("OFFLOAD_PROFILE", "1")
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="atom"):
        assert _dsv4_worker(stats)._load_page(_load_req("dsv4-load")) is True

    message = _profile_message(caplog, "[OFFLOAD-LOAD-PROF]")
    _assert_timing_fields(message, expected)
    assert "retrieve_ms=nan" not in message
    assert "total_ms=nan" not in message
