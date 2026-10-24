# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Pins the connector-selection logic in lmcache_connector_patch."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest

from atom.plugin.vllm.lmcache_connector_patch import (
    _LMCACHE_V3_ENV,
    enforce_lmcache_gpu_connector,
)

DSA_ARCH = "GlmMoeDsaForCausalLM"


@pytest.fixture(autouse=True)
def _clean_v3_env():
    """Keep the selection variable out of the rest of the session.

    ``monkeypatch.delenv(..., raising=False)`` records no undo when the variable
    is absent, so the value the code under test then *sets* survives teardown
    and leaks into every later test. Delete on both sides instead.
    """
    os.environ.pop(_LMCACHE_V3_ENV, None)
    yield
    os.environ.pop(_LMCACHE_V3_ENV, None)


class MlaSpec(SimpleNamespace):
    """A single-vector cache: kv_size 1, the shape GLM-5.2 uses for both families."""


class KVSpec(SimpleNamespace):
    """A separate-K-and-V cache: kv_size 2."""


def _spec(
    num_kv_heads=1,
    head_size=576,
    block_size=64,
    dtype="torch.uint8",
    cls=MlaSpec,
    **extra,
):
    return cls(
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        block_size=block_size,
        dtype=dtype,
        **extra,
    )


def _kv_cache_config(*specs):
    """One vLLM KV cache group whose members may differ in hidden size.

    This is the shape vLLM produces for GLM-5.2: MLA and indexer layers land in
    one ``UniformTypeKVCacheSpecs`` group, so they share block ids.
    """
    group = SimpleNamespace(
        kv_cache_spec=SimpleNamespace(
            kv_cache_specs={f"layer.{i}": spec for i, spec in enumerate(specs)}
        )
    )
    return SimpleNamespace(kv_cache_groups=[group])


def _vllm_config(
    *,
    kv_connector: str | None = "LMCacheConnectorV1",
    index_head_dim: int | None = 128,
    enable_prefix_caching: bool = True,
    nested_connectors: list[dict] | None = None,
    architectures: list[str] | None = None,
):
    hf_config = SimpleNamespace()
    if index_head_dim is not None:
        hf_config.index_head_dim = index_head_dim

    kv_transfer_config = None
    if kv_connector is not None:
        extra = {}
        if nested_connectors is not None:
            extra["connectors"] = nested_connectors
        kv_transfer_config = SimpleNamespace(
            kv_connector=kv_connector,
            kv_connector_extra_config=extra,
        )

    return SimpleNamespace(
        kv_transfer_config=kv_transfer_config,
        model_config=SimpleNamespace(
            hf_text_config=hf_config,
            hf_config=hf_config,
            architectures=architectures or [DSA_ARCH],
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=enable_prefix_caching),
    )


def test_dsa_model_with_lmcache_selects_v3():
    enforce_lmcache_gpu_connector(_vllm_config())

    assert os.environ[_LMCACHE_V3_ENV] == "True"


def test_lmcache_nested_in_multi_connector_is_detected():
    enforce_lmcache_gpu_connector(
        _vllm_config(
            kv_connector="multi",
            nested_connectors=[
                {"kv_connector": "mooncake", "kv_role": "kv_producer"},
                {"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"},
            ],
        )
    )

    assert os.environ[_LMCACHE_V3_ENV] == "True"


@pytest.mark.parametrize("value", ["0", "false", "False", "no", "off", ""])
def test_explicitly_disabled_v3_fails_fast(monkeypatch, value):
    monkeypatch.setenv(_LMCACHE_V3_ENV, value)

    with pytest.raises(ValueError, match=_LMCACHE_V3_ENV):
        enforce_lmcache_gpu_connector(_vllm_config())


def test_explicitly_enabled_v3_is_left_alone(monkeypatch):
    monkeypatch.setenv(_LMCACHE_V3_ENV, "true")

    enforce_lmcache_gpu_connector(_vllm_config())

    assert os.environ[_LMCACHE_V3_ENV] == "true"


def test_dense_model_is_untouched():
    enforce_lmcache_gpu_connector(
        _vllm_config(index_head_dim=None, architectures=["DeepseekV3ForCausalLM"])
    )

    assert _LMCACHE_V3_ENV not in os.environ


def test_dsa_model_without_lmcache_is_untouched():
    enforce_lmcache_gpu_connector(_vllm_config(kv_connector=None))
    enforce_lmcache_gpu_connector(_vllm_config(kv_connector="MooncakeConnector"))

    assert _LMCACHE_V3_ENV not in os.environ


def test_prefix_caching_off_warns_but_does_not_block(caplog):
    with caplog.at_level("WARNING", logger="atom"):
        enforce_lmcache_gpu_connector(_vllm_config(enable_prefix_caching=False))

    assert any("prefix caching is off" in record.message for record in caplog.records)

    assert os.environ[_LMCACHE_V3_ENV] == "True"


def test_real_kv_cache_config_drives_the_decision():
    enforce_lmcache_gpu_connector(
        _vllm_config(index_head_dim=None, architectures=["SomeOtherModel"]),
        _kv_cache_config(_spec(head_size=576), _spec(head_size=132)),
    )

    assert os.environ[_LMCACHE_V3_ENV] == "True"


def test_single_geometry_config_beats_the_architecture_heuristic():
    enforce_lmcache_gpu_connector(
        _vllm_config(),
        _kv_cache_config(_spec(head_size=576), _spec(head_size=576)),
    )

    assert _LMCACHE_V3_ENV not in os.environ


def test_factory_patch_runs_the_guard_and_is_idempotent(monkeypatch):
    factory = pytest.importorskip(
        "vllm.distributed.kv_transfer.kv_connector.factory"
    ).KVConnectorFactory
    from atom.plugin.vllm.lmcache_connector_patch import (
        apply_vllm_lmcache_connector_patch,
    )

    seen = {}
    monkeypatch.setattr(
        factory,
        "create_connector",
        classmethod(
            lambda cls, config, role, kv_cache_config=None: seen.setdefault(
                "role", role
            )
        ),
    )

    apply_vllm_lmcache_connector_patch()
    # Bound classmethods are rebuilt on every attribute access, so compare the
    # underlying function rather than the binding.
    first = factory.create_connector.__func__
    apply_vllm_lmcache_connector_patch()
    assert factory.create_connector.__func__ is first, "patch must not stack"

    factory.create_connector(
        _vllm_config(),
        "worker",
        _kv_cache_config(_spec(head_size=576), _spec(head_size=132)),
    )

    assert os.environ[_LMCACHE_V3_ENV] == "True"
    assert seen["role"] == "worker", "the original factory must still be called"


def test_register_kv_caches_primes_the_layer_groups():
    # V3 builds its layer-group map inside the first transfer, but the memory
    # object is allocated before that and sized from the map -- so the map has
    # to exist by the time registration returns.
    from atom.plugin.vllm.lmcache_connector_patch import _wrap_register_kv_caches

    calls = []

    class _FakeV3GPUConnector:
        init = False

        def initialize_kvcaches_ptr(self, **kwargs):
            calls.append(("initialize_kvcaches_ptr", len(kwargs["kvcaches"])))

        def _initialize_kv_cache_pointers(self):
            calls.append(("_initialize_kv_cache_pointers", None))
            self.init = True

    class _FakeLMCacheConnectorV1:
        def __init__(self):
            self._lmcache_engine = SimpleNamespace(
                lmcache_engine=SimpleNamespace(gpu_connector=_FakeV3GPUConnector())
            )

        def register_kv_caches(self, kv_caches):
            calls.append(("register_kv_caches", len(kv_caches)))

    connector = _FakeLMCacheConnectorV1()
    _wrap_register_kv_caches(connector)
    _wrap_register_kv_caches(connector)  # idempotent

    connector.register_kv_caches({"mla.0": object(), "indexer.0": object()})

    assert calls == [
        ("register_kv_caches", 2),
        ("initialize_kvcaches_ptr", 2),
        ("_initialize_kv_cache_pointers", None),
    ]


def test_priming_failure_does_not_take_the_server_down(caplog):
    # LMCache internals are not a stable API; if they move, fall back to its own
    # lazy discovery rather than killing startup.
    from atom.plugin.vllm.lmcache_connector_patch import _wrap_register_kv_caches

    class _ExplodingGPUConnector:
        init = False

        def initialize_kvcaches_ptr(self, **kwargs):
            raise AttributeError("LMCache moved this")

        def _initialize_kv_cache_pointers(self):  # pragma: no cover - unreachable
            raise AssertionError("should not be reached")

    class _FakeLMCacheConnectorV1:
        def __init__(self):
            self._lmcache_engine = SimpleNamespace(
                lmcache_engine=SimpleNamespace(gpu_connector=_ExplodingGPUConnector())
            )
            self.registered = None

        def register_kv_caches(self, kv_caches):
            self.registered = kv_caches

    connector = _FakeLMCacheConnectorV1()
    _wrap_register_kv_caches(connector)

    with caplog.at_level("WARNING", logger="atom"):
        connector.register_kv_caches({"mla.0": object()})

    assert connector.registered is not None, "registration itself must still happen"
    assert any("lazy discovery" in record.message for record in caplog.records)


def test_non_lmcache_connector_is_not_wrapped():
    from atom.plugin.vllm.lmcache_connector_patch import _wrap_register_kv_caches

    class MooncakeConnector:
        def register_kv_caches(self, kv_caches):
            pass

    connector = MooncakeConnector()
    original = connector.register_kv_caches
    _wrap_register_kv_caches(connector)

    assert connector.register_kv_caches == original


class _AbortedRequest:
    status = SimpleNamespace(name="FINISHED_ABORTED")
    request_id = "req-1"


class _LiveRequest:
    status = SimpleNamespace(name="FINISHED_STOPPED")
    request_id = "req-2"


def _scheduler_role_connector(seen):
    """An LMCache connector as the scheduler role builds it: no engine."""
    from vllm.v1.request import RequestStatus  # noqa: F401  (import sanity only)

    class _FakeLMCacheConnectorV1:
        def __init__(self):
            self._lmcache_engine = SimpleNamespace(
                _manager=SimpleNamespace(_lmcache_engine=None)
            )

        def request_finished(self, request, block_ids):
            # Mirrors LMCache: the aborted branch dereferences the engine.
            if request.status.name == "FINISHED_ABORTED":
                engine = self._lmcache_engine._manager._lmcache_engine
                assert engine is not None
                seen.append(("cancel_checked", engine.storage_manager))
            return False, None

    return _FakeLMCacheConnectorV1()


def test_aborted_request_does_not_kill_a_scheduler_role_connector():
    from atom.plugin.vllm.lmcache_connector_patch import _wrap_request_finished

    seen = []
    connector = _scheduler_role_connector(seen)
    _wrap_request_finished(connector)
    _wrap_request_finished(connector)  # idempotent

    assert connector.request_finished(_AbortedRequest(), [1, 2]) == (False, None)
    assert seen == [("cancel_checked", None)], "the cancel must be a no-op"
    assert (
        connector._lmcache_engine._manager._lmcache_engine is None
    ), "the null engine must not outlive the call"


def test_non_aborted_request_takes_the_untouched_path():
    from atom.plugin.vllm.lmcache_connector_patch import _wrap_request_finished

    seen = []
    connector = _scheduler_role_connector(seen)
    _wrap_request_finished(connector)

    assert connector.request_finished(_LiveRequest(), [1]) == (False, None)
    assert seen == []


# --- truthiness must match LMCache's, not merely be reasonable ---


@pytest.mark.parametrize("value", ["yes", "y", "on", "enabled", "TRUE!", "2"])
def test_values_lmcache_reads_as_off_are_treated_as_off(monkeypatch, value):
    monkeypatch.setenv(_LMCACHE_V3_ENV, value)

    with pytest.raises(ValueError, match=_LMCACHE_V3_ENV):
        enforce_lmcache_gpu_connector(_vllm_config())


@pytest.mark.parametrize("value", ["true", "True", " TRUE ", "1"])
def test_values_lmcache_reads_as_on_are_left_alone(monkeypatch, value):
    monkeypatch.setenv(_LMCACHE_V3_ENV, value)

    assert enforce_lmcache_gpu_connector(_vllm_config()) is True
    assert os.environ[_LMCACHE_V3_ENV] == value


# --- layouts V3 cannot address are refused, not enabled into ---


def test_mixed_kv_size_is_refused_rather_than_given_v3():
    with pytest.raises(ValueError, match="kv_size"):
        enforce_lmcache_gpu_connector(
            _vllm_config(architectures=["MiniMaxM3SparseForCausalLM"]),
            _kv_cache_config(
                _spec(head_size=132, cls=MlaSpec),
                _spec(head_size=128, num_kv_heads=8, cls=KVSpec),
            ),
        )

    assert _LMCACHE_V3_ENV not in os.environ


def test_mixed_physical_block_size_is_refused():
    with pytest.raises(ValueError, match="physical block size"):
        enforce_lmcache_gpu_connector(
            _vllm_config(architectures=["DeepseekV4ForCausalLM"]),
            _kv_cache_config(
                _spec(head_size=576, storage_block_size=64),
                _spec(head_size=132, storage_block_size=16),
            ),
        )

    assert _LMCACHE_V3_ENV not in os.environ


def test_physical_block_size_beats_the_logical_one():
    # Same logical block_size, different compress ratios: keying on
    # spec.block_size alone would miss this.
    with pytest.raises(ValueError, match="physical block size"):
        enforce_lmcache_gpu_connector(
            _vllm_config(),
            _kv_cache_config(
                _spec(head_size=576, block_size=64, storage_block_size=64),
                _spec(head_size=132, block_size=64, storage_block_size=32),
            ),
        )


def test_glm52_shape_is_still_accepted():
    assert (
        enforce_lmcache_gpu_connector(
            _vllm_config(),
            _kv_cache_config(_spec(head_size=576), _spec(head_size=132)),
        )
        is True
    )
    assert os.environ[_LMCACHE_V3_ENV] == "True"


# --- chunk settings that would slice a preshuffled page ---


def _dsa_config():
    return _vllm_config(), _kv_cache_config(_spec(head_size=576), _spec(head_size=132))


def test_chunk_size_not_a_multiple_of_the_page_is_refused(monkeypatch):
    monkeypatch.setenv("LMCACHE_CHUNK_SIZE", "100")
    cfg, kv = _dsa_config()

    with pytest.raises(ValueError, match="LMCACHE_CHUNK_SIZE"):
        enforce_lmcache_gpu_connector(cfg, kv)


def test_saving_unfull_chunks_is_refused(monkeypatch):
    monkeypatch.setenv("LMCACHE_SAVE_UNFULL_CHUNK", "True")
    cfg, kv = _dsa_config()

    with pytest.raises(ValueError, match="LMCACHE_SAVE_UNFULL_CHUNK"):
        enforce_lmcache_gpu_connector(cfg, kv)


def test_default_chunking_is_accepted(monkeypatch):
    monkeypatch.delenv("LMCACHE_CHUNK_SIZE", raising=False)
    monkeypatch.delenv("LMCACHE_SAVE_UNFULL_CHUNK", raising=False)
    cfg, kv = _dsa_config()

    assert enforce_lmcache_gpu_connector(cfg, kv) is True


def test_chunking_is_not_policed_for_single_geometry_models(monkeypatch):
    # No indexer cache means no preshuffled page to slice.
    monkeypatch.setenv("LMCACHE_CHUNK_SIZE", "100")

    assert (
        enforce_lmcache_gpu_connector(
            _vllm_config(index_head_dim=None, architectures=["DeepseekV3ForCausalLM"]),
            _kv_cache_config(_spec(head_size=576), _spec(head_size=576)),
        )
        is False
    )


# --- a priming failure is only survivable when priming was optional ---


class _ExplodingConnectorLMCache:
    def __init__(self):
        self._lmcache_engine = SimpleNamespace(
            lmcache_engine=SimpleNamespace(
                gpu_connector=SimpleNamespace(
                    init=False,
                    initialize_kvcaches_ptr=_boom,
                    _initialize_kv_cache_pointers=lambda: None,
                )
            )
        )
        self.registered = None

    def register_kv_caches(self, kv_caches):
        self.registered = kv_caches


def _boom(**_kwargs):
    raise AttributeError("LMCache moved this")


def test_priming_failure_is_fatal_when_v3_was_required():
    from atom.plugin.vllm.lmcache_connector_patch import _wrap_register_kv_caches

    connector = _ExplodingConnectorLMCache()
    _wrap_register_kv_caches(connector, required=True)

    with pytest.raises(AttributeError):
        connector.register_kv_caches({"mla.0": object()})

    assert connector.registered is not None, "registration still has to happen first"


def test_priming_failure_is_survivable_when_v3_was_optional(caplog):
    from atom.plugin.vllm.lmcache_connector_patch import _wrap_register_kv_caches

    connector = _ExplodingConnectorLMCache()
    _wrap_register_kv_caches(connector, required=False)

    with caplog.at_level("WARNING", logger="atom"):
        connector.register_kv_caches({"mla.0": object()})

    assert any("lazy discovery" in r.message for r in caplog.records)


# --- the null engine has to be inert, not merely have storage_manager ---


def test_null_engine_answers_the_health_probe():
    from atom.plugin.vllm.lmcache_connector_patch import _NullLMCacheEngine

    engine = _NullLMCacheEngine()
    assert engine.storage_manager is None
    assert engine.is_healthy() is True


# --- MultiConnector children are reached ---


def test_lmcache_nested_in_multi_connector_is_wrapped():
    from atom.plugin.vllm.lmcache_connector_patch import _wrap_lmcache_connectors

    class LMCacheConnectorV1:
        def __init__(self):
            self._lmcache_engine = SimpleNamespace(
                _manager=SimpleNamespace(_lmcache_engine=None)
            )

        def register_kv_caches(self, kv_caches):
            pass

        def request_finished(self, request, block_ids):
            return False, None

    class MultiConnector:
        def __init__(self, children):
            self._connectors = children

    child = LMCacheConnectorV1()
    _wrap_lmcache_connectors(MultiConnector([child]), required=True)

    assert getattr(child, "_atom_lmcache_group_prebuilt", False)
    assert getattr(child, "_atom_lmcache_abort_guard", False)


def _fake_first_rank_engine(chunk_sizes, first_rank=True):
    """An LMCache engine stub that records what the broadcast loop does."""
    import torch

    calls = SimpleNamespace(objects=[], tensors=[], delegated=0)

    def broadcast_object_fn(obj, src):
        calls.objects.append((obj, src))

    def broadcast_fn(tensor, src):
        calls.tensors.append((tensor.data_ptr(), tensor.numel(), src))

    def original(reordered_chunks, ret_mask):
        calls.delegated += 1

    original._atom_unblocked = False

    chunks = []
    for i, n in enumerate(chunk_sizes):
        raw = torch.zeros(n, dtype=torch.uint8)
        obj = SimpleNamespace(
            raw_tensor=raw, metadata=SimpleNamespace(to_dict=lambda i=i: {"i": i})
        )
        chunks.append((None, obj, i * 8, i * 8 + 8))

    engine = SimpleNamespace(
        save_only_first_rank=True,
        async_loading=False,
        broadcast_fn=broadcast_fn,
        broadcast_object_fn=broadcast_object_fn,
        _broadcast_or_receive_memory_objs=original,
        retrieve=lambda tokens, mask=None, **kw: None,
        gpu_connector=SimpleNamespace(batched_to_gpu=lambda *a, **k: None),
        metadata=SimpleNamespace(
            is_first_rank=lambda: first_rank, first_rank=0, worker_id=0
        ),
    )
    return engine, chunks, calls


def test_first_rank_delegates_to_upstream_send():
    """Rank 0 must stay on LMCache's own send path — we only window the peers."""
    from atom.plugin.vllm.lmcache_connector_patch import _bound_broadcast_staging

    engine, chunks, calls = _fake_first_rank_engine([64, 64], first_rank=True)
    _bound_broadcast_staging(engine)
    engine._broadcast_or_receive_memory_objs(chunks, None)

    assert calls.delegated == 1, "rank 0 should call through to upstream"
    assert calls.objects == [] and calls.tensors == []


def test_peer_without_retrieve_wrapper_falls_back_to_upstream():
    from atom.plugin.vllm.lmcache_connector_patch import _bound_broadcast_staging

    engine, chunks, calls = _fake_first_rank_engine([64], first_rank=False)
    _bound_broadcast_staging(engine)
    engine._broadcast_or_receive_memory_objs(chunks, None)

    assert calls.delegated == 1
    assert calls.objects == [] and calls.tensors == []


def test_first_rank_copy_patch_is_skipped_and_idempotent():
    from atom.plugin.vllm.lmcache_connector_patch import _bound_broadcast_staging

    engine, _, _ = _fake_first_rank_engine([64])
    engine.save_only_first_rank = False
    before = engine._broadcast_or_receive_memory_objs
    _bound_broadcast_staging(engine)
    assert engine._broadcast_or_receive_memory_objs is before

    engine, _, _ = _fake_first_rank_engine([64])
    _bound_broadcast_staging(engine)
    patched = engine._broadcast_or_receive_memory_objs
    _bound_broadcast_staging(engine)
    assert engine._broadcast_or_receive_memory_objs is patched


def test_peer_receive_uploads_and_releases_every_window():
    """The peers must not hold the whole restore in device memory."""
    pytest.importorskip("torch")
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")

    import lmcache.v1.memory_management as mm

    from atom.plugin.vllm.lmcache_connector_patch import _bound_broadcast_staging

    n_chunks, window = 7, 3
    uploads: list[int] = []
    released: list[int] = []
    live: list[int] = []

    class Obj:
        def __init__(self, raw_data, metadata, parent_allocator):
            self.raw_tensor = raw_data
            live.append(1)

        def ref_count_down(self):
            released.append(1)
            live.pop()

    # rank 0 sends the count first, then one metadata tuple per chunk
    feed = iter([n_chunks] + [(i * 8, i * 8 + 8, {}) for i in range(n_chunks)])
    ret_mask = torch.zeros(n_chunks * 8, dtype=torch.bool)

    def upload(objs, starts, ends, **kwargs):
        assert kwargs == {"kvcaches": "sentinel"}, "retrieve kwargs must reach to_gpu"
        uploads.append(len(objs))
        # nothing may be held beyond the window at any point
        assert len(live) <= window

    engine = SimpleNamespace(
        save_only_first_rank=True,
        async_loading=False,
        broadcast_fn=lambda t, src: None,
        broadcast_object_fn=lambda obj, src: next(feed),
        _broadcast_or_receive_memory_objs=lambda c, m: None,
        gpu_connector=SimpleNamespace(batched_to_gpu=upload),
        metadata=SimpleNamespace(
            is_first_rank=lambda: False, first_rank=0, worker_id=1
        ),
    )
    # the real retrieve is what drives the broadcast helper, so the stub must too
    engine.retrieve = lambda tokens, mask=None, **kw: (
        engine._broadcast_or_receive_memory_objs([], ret_mask)
    )

    real_obj, real_meta = mm.TensorMemoryObj, mm.MemoryObjMetadata
    mm.TensorMemoryObj = Obj
    mm.MemoryObjMetadata = SimpleNamespace(
        from_dict=staticmethod(lambda d: SimpleNamespace(get_size=lambda: 64))
    )
    try:
        _bound_broadcast_staging(engine, window=window)
        engine.retrieve([1], None, kvcaches="sentinel")
    finally:
        mm.TensorMemoryObj, mm.MemoryObjMetadata = real_obj, real_meta

    assert uploads == [3, 3, 1], uploads
    assert sum(released) == n_chunks
    assert not live, "every received chunk must be released"
    assert bool(ret_mask.all()), "every chunk's span must be marked retrieved"


def test_staging_patch_is_skipped_and_idempotent():
    from atom.plugin.vllm.lmcache_connector_patch import _bound_broadcast_staging

    engine, _, _ = _fake_first_rank_engine([64])
    engine.save_only_first_rank = False
    before = engine._broadcast_or_receive_memory_objs
    _bound_broadcast_staging(engine)
    assert engine._broadcast_or_receive_memory_objs is before

    engine, _, _ = _fake_first_rank_engine([64])
    _bound_broadcast_staging(engine)
    patched = engine._broadcast_or_receive_memory_objs
    _bound_broadcast_staging(engine)
    assert engine._broadcast_or_receive_memory_objs is patched


def _pin_connector(pins, loading, grace=1):
    """A worker-side connector stub holding `pins` with `loading` in the step."""
    from atom.plugin.vllm.lmcache_connector_patch import (
        _release_unscheduled_lookup_pins,
    )

    unpinned = []

    engine = SimpleNamespace(
        lookup_pins=dict(pins),
        lookup_unpin=lambda rid: (
            unpinned.append(rid),
            engine.lookup_pins.pop(rid, None),
        ),
    )
    meta = SimpleNamespace(
        requests=[SimpleNamespace(req_id=r, load_spec=object()) for r in loading]
    )
    started = []

    conn = SimpleNamespace(
        _lmcache_engine=SimpleNamespace(lmcache_engine=engine),
        _get_connector_metadata=lambda: meta,
        start_load_kv=lambda ctx, **kw: started.append(1),
    )
    _release_unscheduled_lookup_pins(conn, grace_steps=grace)
    return conn, engine, unpinned, started


def test_pins_that_never_became_loads_are_released():
    """Mirrors the native connector: a matched-but-unloaded lookup must unpin."""
    conn, engine, unpinned, started = _pin_connector(
        pins={"a": {}, "b": {}, "c": {}}, loading=["b"]
    )

    conn.start_load_kv(None)  # first sighting: inside the grace window
    assert unpinned == []
    conn.start_load_kv(None)  # still unscheduled -> release
    assert sorted(unpinned) == ["a", "c"]
    assert "b" in engine.lookup_pins, "a loading request must keep its pin"
    assert len(started) == 2, "the real start_load_kv must still run"


def test_a_pin_that_starts_loading_is_kept():
    conn, _engine, unpinned, _ = _pin_connector(pins={"a": {}}, loading=[])
    conn.start_load_kv(None)
    conn._get_connector_metadata = lambda: SimpleNamespace(
        requests=[SimpleNamespace(req_id="a", load_spec=object())]
    )
    conn.start_load_kv(None)
    conn.start_load_kv(None)
    assert unpinned == [], "the grace counter must reset once loading starts"


def test_pin_release_survives_a_broken_engine():
    """Cleanup is best-effort: it must never block the forward pass."""
    from atom.plugin.vllm.lmcache_connector_patch import (
        _release_unscheduled_lookup_pins,
    )

    started = []
    conn = SimpleNamespace(
        _lmcache_engine=SimpleNamespace(lmcache_engine=None),
        _get_connector_metadata=lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        start_load_kv=lambda ctx, **kw: started.append(1),
    )
    _release_unscheduled_lookup_pins(conn)
    conn.start_load_kv(None)
    assert started == [1]


def test_a_save_only_request_does_not_keep_its_pin():
    """Only a load reads the pinned chunks; a save in flight must not hold them."""
    from atom.plugin.vllm.lmcache_connector_patch import (
        _release_unscheduled_lookup_pins,
    )

    unpinned = []
    engine = SimpleNamespace(
        lookup_pins={"a": {}},
        lookup_unpin=lambda rid: (
            unpinned.append(rid),
            engine.lookup_pins.pop(rid, None),
        ),
    )
    meta = SimpleNamespace(requests=[SimpleNamespace(req_id="a", load_spec=None)])
    conn = SimpleNamespace(
        _lmcache_engine=SimpleNamespace(lmcache_engine=engine),
        _get_connector_metadata=lambda: meta,
        start_load_kv=lambda ctx, **kw: None,
    )
    _release_unscheduled_lookup_pins(conn, grace_steps=1)

    conn.start_load_kv(None)
    conn.start_load_kv(None)
    assert unpinned == ["a"]
