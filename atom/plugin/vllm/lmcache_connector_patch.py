# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Make LMCache usable with ATOM's DSA KV layouts under the vLLM plugin.

GLM-5.2 registers a narrow indexer key cache next to each wide MLA cache and
vLLM folds them into one ``UniformTypeKVCacheSpecs`` group -- 99 tensors in two
page shapes for GLM-5.2-FP8. Three upstream behaviours get in the way:

1. LMCache's default GPU connector sizes its pointer array from
   ``model_config.get_num_layers()`` (78) and dies on the first store with
   ``could not broadcast input array from shape (99,) into shape (78,)``. Its
   V3 connector partitions by page shape instead, but is off by default.

2. V3 builds its layer-group map inside the first transfer, while
   ``LMCacheEngine.store`` sizes the memory object before that from
   ``metadata.get_shapes()`` -- so the first store asks for a group the memory
   object does not have (``IndexError`` at ``gpu_connectors.py:612``).

3. LMCache builds no engine for the scheduler role, but its aborted-request
   cleanup asserts one (``vllm_v1_adapter.py:1779``) from inside the
   scheduler's ``_free_request``, so one client disconnect kills the
   EngineCore. Not DSA-specific.

``KVConnectorFactory.create_connector`` is the hook for all three: one choke
point for both roles, before LMCache's config singleton exists, and it receives
the resolved ``KVCacheConfig``.
"""

import logging
import os
from typing import NamedTuple

logger = logging.getLogger("atom")

_LMCACHE_V3_ENV = "LMCACHE_USE_GPU_CONNECTOR_V3"
# Reference lmcache.v1.config_base._to_bool
_LMCACHE_V3_IS_SET = frozenset({"true", "1"})


def _lmcache_config():
    try:
        from lmcache.v1.config import LMCacheEngineConfig

        return LMCacheEngineConfig.from_env()
    except Exception:
        logger.debug(
            "ATOM vLLM plugin: could not pre-read LMCache config", exc_info=True
        )
        return None


def _is_lmcache_connector(kv_transfer_config) -> bool:
    if kv_transfer_config is None:
        return False

    names = [str(getattr(kv_transfer_config, "kv_connector", "") or "")]
    extra = getattr(kv_transfer_config, "kv_connector_extra_config", None) or {}
    names += [
        str(sub.get("kv_connector", ""))
        for sub in extra.get("connectors", None) or ()
        if isinstance(sub, dict)
    ]
    return any("lmcache" in name.lower() for name in names)


def _layer_specs(kv_cache_config):
    # In DSA, UniformTypeKVCacheSpecs can hold layers with different hidden sizes,
    # so we need to walk over all layers to collect all specs.
    specs = []
    for group in getattr(kv_cache_config, "kv_cache_groups", None) or ():
        spec = getattr(group, "kv_cache_spec", None)
        if spec is None:
            continue
        nested = getattr(spec, "kv_cache_specs", None)
        specs.extend(nested.values() if nested else [spec])
    return specs


class _PageGeometry(NamedTuple):
    # Reference lmcache.v1.kv_layer_groups.LayerGroupIdentity

    kv_size: int
    num_kv_heads: int
    head_size: int
    block_size: "int | None"
    dtype: str


def _page_geometries(kv_cache_config) -> "set[_PageGeometry]":
    try:
        from vllm.v1.kv_cache_interface import MLAAttentionSpec
    except ImportError:
        MLAAttentionSpec = ()

    geometries = set()
    for spec in _layer_specs(kv_cache_config):
        num_kv_heads = getattr(spec, "num_kv_heads", None)
        head_size = getattr(spec, "head_size", None)
        if num_kv_heads is None or head_size is None:
            continue
        mla = isinstance(spec, MLAAttentionSpec) or "mla" in type(spec).__name__.lower()
        geometries.add(
            _PageGeometry(
                kv_size=1 if mla else 2,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                block_size=getattr(spec, "storage_block_size", None)
                or getattr(spec, "block_size", None),
                dtype=str(getattr(spec, "dtype", None)),
            )
        )
    return geometries


def _v3_cannot_express(geometries):
    """Why V3 would mis-address this layout, or None if it can carry it.

    V3 partitions by page shape but not by ``GPUKVFormat`` (probed once from
    ``kv_caches[0]``) nor by the scalar ``block_size`` it hands to the transfer
    kernel, so caches disagreeing on either are read with the wrong layout.
    """
    for field, label in (("kv_size", "kv_size"), ("block_size", "physical block size")):
        values = {getattr(g, field) for g in geometries} - {None}
        if len(values) > 1:
            return f"its caches disagree on {label} ({sorted(values)})"
    return None


def _reject_unsafe_chunking(arch, block_size) -> None:
    # Sparse MLA can preshuffle the indexer cache, but LMCache has settings that
    # may chunk the indexer cache independently, oblivious to the preshuffle.
    # So we need to reject chunk settings that would slice a preshuffled page
    config = _lmcache_config()
    # block_size=1 is not preshuffled
    if config is None or not block_size or block_size <= 1:
        return

    chunk_size = getattr(config, "chunk_size", 0)
    if chunk_size and chunk_size % block_size:
        problem = (
            f"LMCACHE_CHUNK_SIZE={chunk_size} is not a multiple of the "
            f"{block_size}-token KV page"
        )
    elif getattr(config, "save_unfull_chunk", False):
        problem = "LMCACHE_SAVE_UNFULL_CHUNK stores a partial trailing page"
    else:
        return

    msg = (
        f"{problem}, which corrupts {arch}'s preshuffled DSA indexer cache while "
        "leaving the MLA cache intact -- the model keeps running and silently "
        "attends to the wrong tokens."
    )
    logger.error(msg)
    raise ValueError(msg)


def _model_declares_indexer_cache(vllm_config) -> bool:
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_text_config", None) or getattr(
        model_config, "hf_config", None
    )
    return getattr(hf_config, "index_head_dim", None) is not None


def enforce_lmcache_gpu_connector(vllm_config, kv_cache_config=None) -> bool:
    # Select V3 when the KV layout requires it
    if not _is_lmcache_connector(getattr(vllm_config, "kv_transfer_config", None)):
        return False

    cache_config = getattr(vllm_config, "cache_config", None)
    if cache_config is not None and not getattr(
        cache_config, "enable_prefix_caching", False
    ):
        logger.warning(
            "An LMCache KV connector is configured but prefix caching is off. "
            "The CPU/NVMe tier can still be filled, but vLLM will not reuse any "
            "HBM prefix. Pass --enable-prefix-caching."
        )

    model_config = getattr(vllm_config, "model_config", None)
    arch = (getattr(model_config, "architectures", None) or ["<unknown>"])[0]

    geometries = _page_geometries(kv_cache_config)
    block_size = None
    if geometries:
        unsupported = _v3_cannot_express(geometries)
        if unsupported is not None:
            msg = (
                f"{arch} cannot be served with an LMCache KV connector: "
                f"{unsupported}. Drop --kv-transfer-config to serve without KV "
                "offload."
            )
            logger.error(msg)
            raise ValueError(msg)
        needs_v3 = len(geometries) > 1
        reason = f"{len(geometries)} distinct KV page geometries"
        block_size = next(iter(geometries)).block_size
    else:
        needs_v3 = _model_declares_indexer_cache(vllm_config)
        reason = "a DSA indexer KV cache"

    if not needs_v3:
        return False

    _reject_unsafe_chunking(arch, block_size)

    configured = os.environ.get(_LMCACHE_V3_ENV)
    if configured is None:
        os.environ[_LMCACHE_V3_ENV] = "True"
        logger.info(
            "ATOM vLLM: %s has %s, which LMCache's default GPU connector "
            "cannot express. Setting %s=True.",
            arch,
            reason,
            _LMCACHE_V3_ENV,
        )
    elif configured.strip().lower() not in _LMCACHE_V3_IS_SET:
        # An explicit choice is never silently overridden, only refused.
        msg = (
            f"{_LMCACHE_V3_ENV}={configured!r} is incompatible with {arch}, which "
            f"has {reason}. LMCache reads only 'true' and '1' as enabled, so this "
            "selects its default GPU connector, which fails on the first store. "
            f"Set {_LMCACHE_V3_ENV}=True, or drop --kv-transfer-config."
        )
        logger.error(msg)
        raise ValueError(msg)
    return True


def _bound_broadcast_staging(engine, window: int = 32) -> None:
    # In LMCache's save_only_first_rank node, the first rank copies its loaded
    # cache to all other ranks through raw_tensors.to(device). In the upstream
    # LMCache implementation, the receiving ranks gather all chunks in one go.
    # If all data combined exceeds the GPU memory, the receiving ranks abort,
    # and the first rank gets deadlocked waiting on the collectives.
    # To avoid this, instead of accumulating all chunks, batch the staging
    # into smaller number of chunks.
    # lmcache.v1.cache_engine.LMCacheEngine._broadcast_or_receive_memory_objs
    # but with batched chunks
    if not getattr(engine, "save_only_first_rank", False):
        return
    original = getattr(engine, "_broadcast_or_receive_memory_objs", None)
    if original is None or getattr(original, "_atom_bounded_staging", False):
        return

    import torch
    from lmcache import torch_device_type
    from lmcache.v1.memory_management import MemoryObjMetadata, TensorMemoryObj

    original_retrieve = engine.retrieve
    to_gpu_kwargs = None

    # batched_to_gpu needs retrieve's paged-KV kwargs, which the broadcast
    # helper is not given
    def retrieve(tokens, mask=None, **kwargs):
        nonlocal to_gpu_kwargs
        to_gpu_kwargs = kwargs
        try:
            return original_retrieve(tokens, mask, **kwargs)
        finally:
            to_gpu_kwargs = None

    def receive(reordered_chunks, ret_mask):
        first_rank = engine.metadata.first_rank
        local_rank = engine.metadata.worker_id % torch.cuda.device_count()
        device = f"{torch_device_type}:{local_rank}"
        chunk_count = engine.broadcast_object_fn(None, first_rank)
        if chunk_count is None:
            logger.warning("ATOM vLLM: rank %d received None chunk_count", local_rank)
            return

        pending: list = []

        def upload():
            if not pending:
                return
            engine.gpu_connector.batched_to_gpu(
                [c[1] for c in pending],
                [c[2] for c in pending],
                [c[3] for c in pending],
                **to_gpu_kwargs,
            )
            if not engine.async_loading:
                for _key, memory_obj, _start, _end in pending:
                    memory_obj.ref_count_down()
            pending.clear()

        for _ in range(chunk_count):
            combined_metadata = engine.broadcast_object_fn(None, first_rank)
            if combined_metadata is None:
                logger.warning(
                    "ATOM vLLM: rank %d received None chunk metadata", local_rank
                )
                break
            start, end, metadata_dict = combined_metadata
            ret_mask[start:end] = True
            metadata = MemoryObjMetadata.from_dict(metadata_dict)
            raw_tensor = torch.empty(
                torch.Size([metadata.get_size()]), dtype=torch.uint8, device=device
            )
            engine.broadcast_fn(raw_tensor, first_rank)
            pending.append(
                (
                    None,
                    TensorMemoryObj(
                        raw_data=raw_tensor, metadata=metadata, parent_allocator=None
                    ),
                    start,
                    end,
                )
            )
            if len(pending) >= window:
                upload()
        upload()

    def broadcast_or_receive(reordered_chunks, ret_mask):
        # Outside our retrieve wrapper the kwargs are unavailable, so windowing
        # is impossible and upstream has to handle the whole step
        if engine.metadata.is_first_rank() or to_gpu_kwargs is None:
            original(reordered_chunks, ret_mask)
        else:
            receive(reordered_chunks, ret_mask)

    broadcast_or_receive._atom_bounded_staging = True
    engine._broadcast_or_receive_memory_objs = broadcast_or_receive
    engine.retrieve = retrieve
    logger.info(
        "ATOM vLLM: bounded LMCache's first-rank restore staging to %d chunks "
        "per receiving rank",
        window,
    )


def _prebuild_gpu_connector_layer_groups(connector, kv_caches) -> None:
    # Run V3's group discovery at this point, rather than inside the first transfer
    impl = getattr(connector, "_lmcache_engine", None)
    gpu_connector = getattr(
        getattr(impl, "lmcache_engine", None), "gpu_connector", None
    )
    # Only V3 exposes _initialize_kv_cache_pointers, so use this for V3 lookup
    initialize_pointers = getattr(gpu_connector, "_initialize_kv_cache_pointers", None)
    if initialize_pointers is None or getattr(gpu_connector, "init", False):
        return

    gpu_connector.initialize_kvcaches_ptr(kvcaches=list(kv_caches.values()))
    initialize_pointers()
    _bound_broadcast_staging(getattr(impl, "lmcache_engine", None))

    manager = getattr(
        getattr(gpu_connector, "metadata", None), "kv_layer_groups_manager", None
    )
    groups = getattr(manager, "kv_layer_groups", None) or ()
    logger.info(
        "ATOM vLLM: primed LMCache KV layer groups from %d registered caches -> %s",
        len(kv_caches),
        [(g.num_layers, g.shape_desc.hs) for g in groups],
    )


def _mark_lmcache(connector, flag) -> bool:
    if "lmcache" not in type(connector).__name__.lower():
        return False
    if getattr(connector, flag, False):
        return False
    setattr(connector, flag, True)
    return True


def _wrap_register_kv_caches(connector, required: bool = False) -> None:
    # Prebuild the KV tensors once vLLM register_kv_caches
    register_kv_caches = getattr(connector, "register_kv_caches", None)
    if register_kv_caches is None or not _mark_lmcache(
        connector, "_atom_lmcache_group_prebuilt"
    ):
        return

    def register_and_prime(kv_caches):
        register_kv_caches(kv_caches)
        try:
            _prebuild_gpu_connector_layer_groups(connector, kv_caches)
        except Exception:
            if required:
                raise
            logger.warning(
                "ATOM vLLM: could not pre-build LMCache's KV layer groups; "
                "falling back to its lazy discovery.",
                exc_info=True,
            )

    connector.register_kv_caches = register_and_prime


class _NullLMCacheEngine:
    """Stands in for the engine a scheduler-role connector does not have.

    ``storage_manager`` is what LMCache's aborted-request cleanup checks before
    cancelling; ``is_healthy`` only because the stub is briefly visible to
    ``LMCacheManager.is_healthy()``, which backs a Prometheus gauge.
    """

    storage_manager = None

    @staticmethod
    def is_healthy() -> bool:
        return True


def _release_unscheduled_lookup_pins(connector, grace_steps: int = 1) -> None:
    # Mirroring atom.kv_transfer.offload.dense.connector, release lookup pins
    # for requests that are never served so that they can be evicted.
    start_load_kv = getattr(connector, "start_load_kv", None)
    impl = getattr(connector, "_lmcache_engine", None)
    if start_load_kv is None or impl is None:
        return
    if getattr(start_load_kv, "_atom_pin_release", False):
        return

    idle_steps: dict = {}

    def release_stale_pins():
        engine = getattr(impl, "lmcache_engine", None)
        pins = getattr(engine, "lookup_pins", None)
        if not pins:
            idle_steps.clear()
            return
        metadata = connector._get_connector_metadata()
        # A request that is only being saved never reads the pinned chunks
        loading = {
            str(getattr(req, "req_id", ""))
            for req in getattr(metadata, "requests", ())
            if getattr(req, "load_spec", None) is not None
        }
        for lookup_id in list(pins):
            key = str(lookup_id)
            if key in loading:
                idle_steps.pop(key, None)
                continue
            idle = idle_steps.get(key, 0) + 1
            if idle > grace_steps:
                engine.lookup_unpin(key)
                idle_steps.pop(key, None)
                logger.debug(
                    "ATOM vLLM: released the LMCache lookup pin for %s, which "
                    "was matched but never loaded",
                    key,
                )
            else:
                idle_steps[key] = idle
        for key in list(idle_steps):
            if key not in pins:
                idle_steps.pop(key, None)

    def start_load_kv_releasing(forward_context, **kwargs):
        try:
            release_stale_pins()
        except Exception:  # optional third-party cleanup boundary
            logger.debug(
                "ATOM vLLM: releasing stale LMCache lookup pins failed",
                exc_info=True,
            )
        return start_load_kv(forward_context, **kwargs)

    start_load_kv_releasing._atom_pin_release = True
    connector.start_load_kv = start_load_kv_releasing
    logger.info(
        "ATOM vLLM: releasing LMCache lookup pins that do not become loads "
        "within %d step(s), so an unschedulable request cannot pin the CPU tier",
        grace_steps + 1,
    )


def _wrap_request_finished(connector) -> None:
    # Let an aborted request reach LMCache's cleanup without an engine
    impl = getattr(connector, "_lmcache_engine", None)
    request_finished = getattr(connector, "request_finished", None)
    if impl is None or request_finished is None:
        return
    if not _mark_lmcache(connector, "_atom_lmcache_abort_guard"):
        return

    def request_finished_without_engine(request, block_ids):
        manager = getattr(impl, "_manager", None)
        aborted = getattr(getattr(request, "status", None), "name", "") == (
            "FINISHED_ABORTED"
        )
        if (
            not aborted
            or manager is None
            or getattr(manager, "_lmcache_engine", "missing") is not None
        ):
            return request_finished(request, block_ids)

        manager._lmcache_engine = _NullLMCacheEngine()
        try:
            return request_finished(request, block_ids)
        finally:
            manager._lmcache_engine = None

    connector.request_finished = request_finished_without_engine


def _wrap_lmcache_connectors(connector, required: bool) -> None:

    # vLLM's MultiConnector builds its children directly instead of coming
    # back through the factory, so recursing here is the only way a nested
    # LMCache connector is reached.
    _wrap_register_kv_caches(connector, required)
    _wrap_request_finished(connector)
    _release_unscheduled_lookup_pins(connector)
    for child in getattr(connector, "_connectors", None) or ():
        _wrap_lmcache_connectors(child, required)


def apply_vllm_lmcache_connector_patch() -> None:
    import functools

    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    original = KVConnectorFactory.create_connector
    if getattr(original, "_atom_lmcache_connector_patched", False):
        return

    @functools.wraps(original.__func__)
    def create_connector(cls, config, role, kv_cache_config=None, *args, **kwargs):
        required = enforce_lmcache_gpu_connector(config, kv_cache_config)
        extra = () if kv_cache_config is None else (kv_cache_config,)
        connector = original.__func__(cls, config, role, *extra, *args, **kwargs)
        _wrap_lmcache_connectors(connector, required)
        return connector

    create_connector._atom_lmcache_connector_patched = True
    KVConnectorFactory.create_connector = classmethod(create_connector)
    logger.info(
        "ATOM vLLM: patched vLLM KVConnectorFactory to select and prime "
        "LMCache's multi-geometry GPU connector for DSA KV layouts"
    )
