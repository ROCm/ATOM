# SPDX-License-Identifier: MIT
"""CPU contract for optional dense lookup initialization diagnostics."""

import logging
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
from conftest import atom_config_double

from atom.kv_transfer.offload.dense import connector as dense

ROLES = ["offload", "kv_both", "kv_producer", "kv_consumer"]
SIDES = ["scheduler", "worker"]
DETAIL = "synthetic lookup endpoint refused"


def _install_boundary_module(monkeypatch, name, **attributes):
    parts = name.split(".")
    for index in range(1, len(parts)):
        parent = ".".join(parts[:index])
        if parent not in sys.modules:
            package = ModuleType(parent)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, parent, package)
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)


@pytest.fixture
def lookup_boundary(monkeypatch):
    client = SimpleNamespace()
    server = SimpleNamespace()
    factory = SimpleNamespace(
        create_lookup_client=Mock(return_value=client),
        create_lookup_server=Mock(return_value=server),
    )
    _install_boundary_module(
        monkeypatch,
        "lmcache.v1.lookup_client.factory",
        LookupClientFactory=factory,
    )
    tp = SimpleNamespace(world_size=1, rank_in_group=0)
    get_tp_group = Mock(return_value=tp)
    _install_boundary_module(
        monkeypatch, "aiter.dist.parallel_state", get_tp_group=get_tp_group
    )
    cfg = SimpleNamespace(chunk_size=8)
    metadata = SimpleNamespace(
        engine_id="synthetic-dense-lookup", world_size=1, worker_id=0, use_mla=False
    )
    monkeypatch.setattr(dense.offcfg, "build_lmcache_config", lambda *_args: cfg)
    monkeypatch.setattr(dense.offcfg, "build_lmcache_metadata", lambda *_args: metadata)
    codec = SimpleNamespace(bytes_per_block=16)
    codec_factory = Mock(return_value=codec)
    monkeypatch.setattr(dense, "DenseKVByteCodec", codec_factory)
    engine = SimpleNamespace(
        gpu_connector=SimpleNamespace(
            gpu_staging_chunk_bytes=0,
            gpu_staging_buffer_chunks=0,
            gpu_staging_buffer_bytes=0,
            release_gpu_staging_after_transfer=False,
        )
    )
    build_engine = Mock(return_value=(engine, cfg, metadata))
    monkeypatch.setattr(dense, "build_offload_engine", build_engine)
    workers = []

    def construct(side, role):
        config = atom_config_double(
            kv_transfer_config={"kv_role": role},
            kv_cache_block_size=4,
            decode_context_parallel_size=2,
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
        )
        if side == "scheduler":
            result = dense.DenseOffloadScheduler(config)
            factory.create_lookup_client.assert_called_once_with(cfg, metadata)
        else:
            result = dense.DenseOffloadConnector(config)
            workers.append(result)
            result.register_kv_caches({}, num_blocks=2)
            get_tp_group.assert_called_once_with()
            codec_factory.assert_called_once_with(
                {}, num_blocks=2, permit_per_request_state=False
            )
            build_engine.assert_called_once()
            factory.create_lookup_server.assert_called_once_with(engine, metadata)
            assert result._engine is engine
        return result

    try:
        yield SimpleNamespace(construct=construct, factory=factory)
    finally:
        for worker in workers:
            worker._save_executor.shutdown(wait=True)
            worker._load_executor.shutdown(wait=True)


def _assert_role(result, role):
    assert result._do_save is (role != "kv_consumer")
    assert result._do_load is (role != "kv_producer")


def _lookup_warnings(caplog):
    return [
        record
        for record in caplog.records
        if record.name == "atom"
        and record.levelno >= logging.WARNING
        and "lookup" in record.getMessage()
    ]


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("role", ROLES)
@pytest.mark.parametrize("message", ["", DETAIL], ids=["empty", "message"])
def test_lookup_initialization_warning_adds_type_without_losing_message(
    lookup_boundary, caplog, side, role, message
):
    method = (
        lookup_boundary.factory.create_lookup_client
        if side == "scheduler"
        else lookup_boundary.factory.create_lookup_server
    )
    method.side_effect = RuntimeError(message)
    with caplog.at_level(logging.WARNING, logger="atom"):
        result = lookup_boundary.construct(side, role)

    _assert_role(result, role)
    attribute = "_lookup_client" if side == "scheduler" else "_lookup_server"
    assert getattr(result, attribute) is None
    warnings = _lookup_warnings(caplog)
    assert len(warnings) == 1
    rendered = warnings[0].getMessage()
    component = "lookup client" if side == "scheduler" else "lookup server"
    assert component in rendered
    assert "error_type=RuntimeError" in rendered, "LOOKUP_DIAGNOSTIC_CLASS_MISSING"
    assert rendered.endswith(f"error={message}")
    if message:
        assert message in rendered, "LOOKUP_DIAGNOSTIC_MESSAGE_LOST"
    assert warnings[0].exc_info is None
    assert warnings[0].stack_info is None


@pytest.mark.parametrize("side", SIDES)
@pytest.mark.parametrize("role", ROLES)
def test_lookup_success_preserves_object_and_role(lookup_boundary, caplog, side, role):
    with caplog.at_level(logging.INFO, logger="atom"):
        result = lookup_boundary.construct(side, role)
    _assert_role(result, role)
    attribute = "_lookup_client" if side == "scheduler" else "_lookup_server"
    method = (
        lookup_boundary.factory.create_lookup_client
        if side == "scheduler"
        else lookup_boundary.factory.create_lookup_server
    )
    assert getattr(result, attribute) is method.return_value
    assert not _lookup_warnings(caplog)


@pytest.mark.parametrize("role", ROLES)
def test_lookup_valid_nonhosting_worker_is_not_failure(lookup_boundary, caplog, role):
    lookup_boundary.factory.create_lookup_server.return_value = None
    with caplog.at_level(logging.WARNING, logger="atom"):
        result = lookup_boundary.construct("worker", role)
    _assert_role(result, role)
    assert result._lookup_server is None
    assert not _lookup_warnings(caplog)


def test_lookup_ordinary_miss_recomputes_without_startup_warning(
    lookup_boundary, caplog
):
    lookup = Mock(return_value=0)
    lookup_boundary.factory.create_lookup_client.return_value = SimpleNamespace(
        lookup=lookup, clear_lookup_status=lambda _sid: None
    )
    with caplog.at_level(logging.WARNING, logger="atom"):
        result = lookup_boundary.construct("scheduler", "kv_consumer")
        seq = SimpleNamespace(
            id=101,
            num_cached_tokens=0,
            num_prompt_tokens=24,
            token_ids=list(range(24)),
            block_table=[0, 1, 2],
        )
        assert result.get_num_new_matched_tokens(seq) == (0, False)
    lookup.assert_called_once()
    assert not _lookup_warnings(caplog)
