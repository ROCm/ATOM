"""Prefix caching through vLLM-managed checkpoint images, for any ATOM model
whose per-request state lives in slots outside the blocks vLLM hashes.

Such a model (DeepSeek-V4's sliding-window ring and compressor tails, for one)
exposes its paged cache to vLLM as one opaque proxy layer. A block-table hit
brings back the paged history and nothing a resumer needs at the boundary. This
module lets vLLM keep, at the boundaries native ATOM checkpoints at, a byte image
of that slot and hand it back on a hit:

* An image is native ATOM's checkpoint image (``PagedStateCheckpointSpec``), cut
  over ``k = spec.units_per_checkpoint`` PAGE units. vLLM sees it as ``k``
  single-layer ``MambaSpec`` groups in ``align`` mode at the proxy's block size,
  which share the proxy's tensor and block pool. A hit returns the ``k`` image
  block ids with the proxy blocks; vLLM only reports the hit when both are cached.
* The copy between a slot and ``k`` PAGE units is the model's native
  ``execute_paged_state_copies``, supplied by the model family's adapter.
* A block id means the same unit to vLLM and ATOM but not the same bytes, so every
  vLLM-side byte operation on these blocks is switched off: the align pre-copy,
  new-block zeroing and partial-hit CoW block copies.

Everything model-specific comes through a ``PagedStateImageAdapter`` the model's
bridge registers. This module and ``paged_state_image_scheduler`` import no model
code.
"""

from __future__ import annotations

import functools
import logging

import numpy as np
import torch
from torch import nn
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
)

logger = logging.getLogger("atom")


# ---------------------------------------------------------------------------
# Adapter: what a model family supplies
# ---------------------------------------------------------------------------


class PagedStateImageAdapter:
    """One model family's side of the image path.

    ``sizing(vllm_config)`` returns an object with ``spec`` (native
    ``PagedStateCheckpointSpec``), ``page_bytes`` (what vLLM prices a block at),
    ``num_slots``, ``slot_bytes`` and ``usable_blocks(tensor_blocks)`` (blocks the
    pool may hand out once the slot area is withheld). The copier a bridge builds
    exposes ``checkpoint_spec`` and native
    ``execute_paged_state_copies(stores, restores, descriptor_slot)``.
    """

    #: Family name in logs and errors.
    name: str = "paged-state"
    #: Suffix of the image layer names.
    layer_suffix: str = ""
    #: The proxy's and every image group's block size.
    block_size: int = 0
    #: vLLM attention backend of the image layers (a ``PagedStateImageBackend``).
    backend_cls: type | None = None

    @property
    def proxy_layer_name(self) -> str:
        raise NotImplementedError

    def matches(self, vllm_config) -> bool:
        raise NotImplementedError

    def sizing(self, vllm_config):
        raise NotImplementedError

    def image_layer_names(self, k: int) -> list[str]:
        return [f"model.layers.{1 + i}.{self.layer_suffix}" for i in range(k)]

    def is_image_layer(self, name: str) -> bool:
        return name.endswith("." + self.layer_suffix)

    def checkpoint_defaults(self) -> tuple[int, bool]:
        """Native (state_checkpoint_interval_tokens, state_checkpoint_demand)."""
        raise NotImplementedError


_adapters: list[PagedStateImageAdapter] = []


def register_image_adapter(adapter: PagedStateImageAdapter) -> PagedStateImageAdapter:
    for a in _adapters:
        if type(a) is type(adapter):
            return a
    _adapters.append(adapter)
    return adapter


def adapter_for_config(vllm_config) -> PagedStateImageAdapter | None:
    for a in _adapters:
        if a.matches(vllm_config):
            return a
    return None


def adapter_for_kv_cache_config(cfg) -> PagedStateImageAdapter | None:
    groups = getattr(cfg, "kv_cache_groups", ())
    for a in _adapters:
        if any(a.proxy_layer_name in g.layer_names for g in groups):
            return a
    return None


# ---------------------------------------------------------------------------
# Switches
# ---------------------------------------------------------------------------


def images_on(adapter: PagedStateImageAdapter | None, vllm_config) -> bool:
    """Whether this config's prefix hits are restored from checkpoint images."""
    cc = getattr(vllm_config, "cache_config", None)
    return (
        adapter is not None
        and adapter.matches(vllm_config)
        and bool(getattr(cc, "enable_prefix_caching", False))
    )


# ---------------------------------------------------------------------------
# Image layers: k single-layer MambaSpec(align) groups
# ---------------------------------------------------------------------------


def _noop_image_copy(state, block_ids, cur_block_idx, num_accepted_tokens):
    """vLLM's align pre-copy, made to copy nothing.

    ATOM writes an image itself, through its own carve; vLLM's view of the same
    block id is other blocks' bytes (see module docstring).
    """
    from vllm.model_executor.layers.mamba.mamba_utils import MambaCopySpec

    src = state[block_ids[cur_block_idx]]
    return MambaCopySpec(start_addr=src.data_ptr(), num_elements=0)


def image_state_copy_funcs():
    return (_noop_image_copy,)


class _PagedStateImageMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        return None

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return None


class PagedStateImageBackend(AttentionBackend):
    """Image layers' backend; a family subclasses it with its name and block."""

    block_size: int = 0

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return cls.block_size

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        raise NotImplementedError("image layers are MambaSpec; vLLM sizes them")

    @staticmethod
    def get_impl_cls():
        return nn.Identity

    @staticmethod
    def get_builder_cls():
        return _PagedStateImageMetadataBuilder

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)


class PagedStateImageLayer(nn.Module, AttentionLayerBase):
    """One PAGE unit of a checkpoint image, as vLLM sees it."""

    def __init__(self, prefix: str, part: int, page_bytes: int, adapter):
        super().__init__()
        self.prefix = prefix
        self.part = int(part)
        self.page_bytes = int(page_bytes)
        self.adapter = adapter
        self._atom_state_image_layer = True
        self._atom_image_group_id = None
        self.kv_cache = (torch.tensor([]),)
        self.impl = nn.Identity()

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.adapter.backend_cls

    def get_kv_cache_spec(self, vllm_config):
        from vllm.v1.kv_cache_interface import MambaSpec

        return MambaSpec(
            block_size=self.adapter.block_size,
            shapes=((self.page_bytes,),),
            dtypes=(torch.uint8,),
            mamba_cache_mode="align",
        )

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # One state per block: the whole page. Only its address is ever used
        # (by the no-op copy spec); ATOM never reads bytes through this view.
        pages = kv_cache.reshape(kv_cache.shape[0], -1)
        self.kv_cache = (pages,)


def register_image_layers(
    adapter, vllm_config, layer_cls=PagedStateImageLayer
) -> list[str]:
    """Register the k image layers next to the proxy (worker side)."""
    sizing = adapter.sizing(vllm_config)
    sfc = vllm_config.compilation_config.static_forward_context
    names = adapter.image_layer_names(sizing.spec.units_per_checkpoint)
    for i, name in enumerate(names):
        existing = sfc.get(name)
        if isinstance(existing, PagedStateImageLayer):
            continue
        if existing is not None:
            raise ValueError(f"Duplicate layer name: {name}")
        sfc[name] = layer_cls(name, i, sizing.page_bytes, adapter)
    return names


# ---------------------------------------------------------------------------
# KV-cache config: align groups, startup self-check, slot tail
# ---------------------------------------------------------------------------

_KV_CFG_TARGETS = ("vllm.v1.core.kv_cache_utils", "vllm.v1.engine.core")
_KV_CFG_ATTR = "get_kv_cache_configs"


def check_image_groups(cfg, adapter, sizing) -> None:
    from vllm.v1.kv_cache_interface import MambaSpec

    k = sizing.spec.units_per_checkpoint
    proxy_name = adapter.proxy_layer_name
    image_names = set(adapter.image_layer_names(k))
    groups = list(cfg.kv_cache_groups)
    proxy_groups = [g for g in groups if proxy_name in g.layer_names]
    image_groups = [
        g for g in groups if any(adapter.is_image_layer(n) for n in g.layer_names)
    ]
    problems = []
    if len(proxy_groups) != 1:
        problems.append(f"{len(proxy_groups)} proxy groups")
    if len(image_groups) != k:
        problems.append(f"{len(image_groups)} image groups, need k={k}")
    found = set()
    for g in image_groups:
        spec = g.kv_cache_spec
        if len(g.layer_names) != 1:
            problems.append(f"image group with layers {g.layer_names}")
        if not isinstance(spec, MambaSpec) or spec.mamba_cache_mode != "align":
            problems.append(f"image group spec {type(spec).__name__}")
        found.update(g.layer_names)
    if found != image_names:
        problems.append(f"image layers {sorted(found)} != {sorted(image_names)}")
    for g in groups:
        if g.kv_cache_spec.block_size != adapter.block_size:
            problems.append(f"group block_size {g.kv_cache_spec.block_size}")
        if g.kv_cache_spec.page_size_bytes != sizing.page_bytes:
            problems.append(f"group page {g.kv_cache_spec.page_size_bytes}")
    if len(cfg.kv_cache_tensors) != 1:
        problems.append(f"{len(cfg.kv_cache_tensors)} kv tensors, need 1 shared")
    else:
        shared = set(cfg.kv_cache_tensors[0].shared_by)
        if proxy_name not in shared or not image_names <= shared:
            problems.append("the tensor is not shared by the proxy and every image")
    if problems:
        raise RuntimeError(
            f"{adapter.name} prefix caching needs 1 proxy group + k single-layer "
            "MambaSpec(align) image groups sharing one tensor at block size "
            f"{adapter.block_size}; got: " + "; ".join(problems)
        )


def _kv_cache_configs_hook(original, vllm_config, kv_cache_specs, available_memory):
    adapter = adapter_for_config(vllm_config)
    if adapter is None:
        return original(vllm_config, kv_cache_specs, available_memory)
    from atom.plugin.vllm.paged_state_image_scheduler import ImagePlacement

    proxy_name = adapter.proxy_layer_name
    bs = adapter.block_size
    pc = images_on(adapter, vllm_config)
    sizing = adapter.sizing(vllm_config)
    k = sizing.spec.units_per_checkpoint
    if pc:
        if getattr(vllm_config, "speculative_config", None) is not None:
            raise ValueError(
                f"{adapter.name} prefix caching through checkpoint images does not "
                "drive the MTP draft proxy; drop --speculative-config or pass "
                "--no-enable-prefix-caching."
            )
        vllm_config.cache_config.mamba_cache_mode = "align"
    configs = original(vllm_config, kv_cache_specs, available_memory)
    max_model_len = int(vllm_config.model_config.max_model_len)
    per_req = -(-max_model_len // bs)
    placement = (
        ImagePlacement.from_env(adapter.checkpoint_defaults(), bs) if pc else None
    )
    # A request holds at most two images' blocks at once.
    images_per_req = placement.images_per_request() if pc else 0
    for cfg in configs:
        if not any(proxy_name in g.layer_names for g in cfg.kv_cache_groups):
            continue
        if pc:
            check_image_groups(cfg, adapter, sizing)
            logger.info(
                "ATOM %s image placement: state_checkpoint_interval_tokens=%d "
                "state_checkpoint_demand=%s block_size=%d images_per_request=%d; "
                "pool check: %d proxy blocks (max_model_len=%d) + %d x k=%d image blocks = %d",
                adapter.name,
                placement.interval,
                placement.demand,
                bs,
                images_per_req,
                per_req,
                max_model_len,
                images_per_req,
                k,
                per_req + images_per_req * k,
            )
        tensor_blocks = int(cfg.num_blocks)
        usable = sizing.usable_blocks(tensor_blocks)
        need = per_req + images_per_req * k
        if usable < need:
            images = f" + {images_per_req}x{k} image blocks" if pc else ""
            raise ValueError(
                f"{adapter.name} plugin pool too small: "
                f"{tensor_blocks} blocks of {sizing.page_bytes} B, "
                f"{tensor_blocks - usable} withheld for {sizing.num_slots} slots of "
                f"{sizing.slot_bytes} B, {usable} left; one max_model_len="
                f"{max_model_len} request needs {need} "
                f"({per_req} KV blocks{images})"
            )
        cfg.num_blocks = usable
        logger.info(
            "ATOM %s %s: page_bytes=%d page_unit_bytes=%d image_k=%d "
            "image_bytes=%d block_size=%d slots=%d slot_bytes=%d tensor_blocks=%d "
            "reserve_blocks=%d schedulable_blocks=%d kv_groups=%d "
            "mamba_cache_mode=%s long_prefill_token_threshold=%s",
            adapter.name,
            "image prefix" if pc else "proxy (no prefix caching)",
            sizing.page_bytes,
            sizing.spec.page_unit_bytes,
            k if pc else 0,
            sizing.spec.image_bytes,
            bs,
            sizing.num_slots,
            sizing.slot_bytes,
            tensor_blocks,
            tensor_blocks - usable,
            usable,
            len(cfg.kv_cache_groups),
            vllm_config.cache_config.mamba_cache_mode,
            vllm_config.scheduler_config.long_prefill_token_threshold,
        )
    return configs


def _apply_kv_cache_configs_patch() -> bool:
    import importlib

    modules = []
    for name in _KV_CFG_TARGETS:
        try:
            modules.append(importlib.import_module(name))
        except Exception as e:  # noqa: BLE001 - optional/version-dependent module
            logger.debug("ATOM image install: %s unavailable (%s)", name, e)
    modules = [m for m in modules if hasattr(m, _KV_CFG_ATTR)]
    if not modules:
        return False
    current = getattr(modules[0], _KV_CFG_ATTR)
    if getattr(current, "_atom_state_image_patched", False):
        return False

    @functools.wraps(current)
    def patched(vllm_config, kv_cache_specs, available_memory):
        return _kv_cache_configs_hook(
            current, vllm_config, kv_cache_specs, available_memory
        )

    patched._atom_state_image_patched = True
    for module in modules:
        setattr(module, _KV_CFG_ATTR, patched)
    return True


def _config_has_proxy(cfg) -> bool:
    return adapter_for_kv_cache_config(cfg) is not None


def _apply_zeroing_patch() -> bool:
    from vllm.v1.kv_cache_interface import KVCacheConfig

    prop = KVCacheConfig.__dict__.get("needs_kv_cache_zeroing")
    if prop is None or getattr(prop.fget, "_atom_state_image_patched", False):
        return False
    original = prop.fget

    def needs_kv_cache_zeroing(self) -> bool:
        # vLLM zeroes by its own view of a block, which is other blocks' rows in
        # an opaque proxy carve. ATOM writes every block before reading it.
        if _config_has_proxy(self):
            return False
        return original(self)

    needs_kv_cache_zeroing._atom_state_image_patched = True
    KVCacheConfig.needs_kv_cache_zeroing = property(needs_kv_cache_zeroing)
    return True


# ---------------------------------------------------------------------------
# Runner side: lifecycle events, the step's store plan, image group ids
# ---------------------------------------------------------------------------

# Worker-side lifecycle events go straight to every slot allocator in the
# process (target and, if any, draft), before this step's metadata is built.
_slot_allocators: list = []
_event_counts = {"finished": 0, "preempted": 0, "resumed": 0}
# This step's image stores as the scheduler hashed them: req_id -> boundary.
_store_plan: dict[str, int] = {}


# The image worker of each registered family's target model in this process.
_workers: dict[str, ImageWorker] = {}
# Stores registered by builds, and where they ran: right after the step's model
# forward (the only intended site) or at the next build (a step whose forward
# did not run after its build).
_store_counts = {"planned": 0, "after_forward": 0, "at_next_build": 0}


def current_image_store_plan() -> dict[str, int]:
    return _store_plan


def register_image_worker(worker) -> None:
    """The worker whose stores run after every model forward of this process."""
    _workers[worker.adapter.name] = worker


def store_counts() -> dict:
    return dict(_store_counts)


def register_slot_allocator(allocator) -> None:
    """``allocator.release(req_ids)`` gets vLLM's finished / preempted / resumed ids."""
    if not any(a is allocator for a in _slot_allocators):
        _slot_allocators.append(allocator)


def _apply_runner_patches() -> bool:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    original_update = GPUModelRunner._update_states
    if getattr(original_update, "_atom_state_image_patched", False):
        return False

    @functools.wraps(original_update)
    def wrapped_update_states(self, scheduler_output, *args, **kwargs):
        finished = getattr(scheduler_output, "finished_req_ids", None) or ()
        preempted = getattr(scheduler_output, "preempted_req_ids", None) or ()
        cached = getattr(scheduler_output, "scheduled_cached_reqs", None)
        resumed = getattr(cached, "resumed_req_ids", None) or ()
        gone = set(finished) | set(preempted) | set(resumed)
        if gone:
            for allocator in _slot_allocators:
                allocator.release(gone)
        _event_counts["finished"] += len(finished)
        _event_counts["preempted"] += len(preempted)
        _event_counts["resumed"] += len(resumed)
        _store_plan.clear()
        _store_plan.update(getattr(scheduler_output, "atom_image_stores", None) or {})
        copies = getattr(scheduler_output, "kv_cache_block_copies", None)
        if copies and _config_has_proxy(getattr(self, "kv_cache_config", None)):
            raise RuntimeError(
                "vLLM scheduled whole-block copies on an ATOM proxy pool; its view "
                "of a block is not ATOM's carve, so they would corrupt it"
            )
        return original_update(self, scheduler_output, *args, **kwargs)

    wrapped_update_states._atom_state_image_patched = True
    GPUModelRunner._update_states = wrapped_update_states

    original_init_kv = GPUModelRunner.initialize_kv_cache

    @functools.wraps(original_init_kv)
    def wrapped_initialize_kv_cache(self, kv_cache_config, *args, **kwargs):
        result = original_init_kv(self, kv_cache_config, *args, **kwargs)
        sfc = self.compilation_config.static_forward_context
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            for name in group.layer_names:
                layer = sfc.get(name)
                if getattr(layer, "_atom_state_image_layer", False):
                    layer._atom_image_group_id = gid
        return result

    wrapped_initialize_kv_cache._atom_state_image_patched = True
    GPUModelRunner.initialize_kv_cache = wrapped_initialize_kv_cache

    original_forward = GPUModelRunner._model_forward

    @functools.wraps(original_forward)
    def wrapped_model_forward(self, *args, **kwargs):
        # A FULL cudagraph replay never enters the model's Python, so the step's
        # stores run here: after the replay or the forward is enqueued, on the
        # same stream, outside any capture, before the next step's build.
        output = original_forward(self, *args, **kwargs)
        for worker in list(_workers.values()):
            flush_worker(worker)
        return output

    wrapped_model_forward._atom_state_image_patched = True
    GPUModelRunner._model_forward = wrapped_model_forward
    return True


def install() -> tuple[bool, bool, bool, bool]:
    """Install every image hook. Idempotent; a no-op for unregistered models.

    Returns which hooks this call installed: (get_kv_cache_configs, kv zeroing,
    runner events, placement scheduler).
    """
    from atom.plugin.vllm.paged_state_image_scheduler import apply_scheduler_patches

    return (
        _apply_kv_cache_configs_patch(),
        _apply_zeroing_patch(),
        _apply_runner_patches(),
        apply_scheduler_patches(),
    )


# ---------------------------------------------------------------------------
# Per-step image ops (worker)
# ---------------------------------------------------------------------------


class ImageWorker:
    """A target model's image path on one worker: its copier, the image groups'
    ids and the stores a step leaves for after its forward."""

    def __init__(self, adapter, copier, group_ids):
        self.adapter = adapter
        self.copier = copier
        self.group_ids = list(group_ids)
        self.pending_stores = None


def image_unit_ids(
    group_ids, rows, cols, name: str = "paged-state"
) -> list[tuple[int, ...]]:
    """`k` image block ids per (batch row, block column), from the host block table."""
    from atom.plugin.vllm.req_id_passthrough_patch import get_current_input_batch

    batch = get_current_input_batch()
    if batch is None or group_ids is None:
        raise RuntimeError(
            f"{name} image ops need the vLLM input batch and the image group "
            "ids; the runner patches did not run"
        )
    tables = [batch.block_table[g].block_table.np for g in group_ids]
    out = []
    for row, col in zip(rows, cols):
        units = tuple(int(t[row, col]) for t in tables)
        if any(u <= 0 for u in units):
            raise RuntimeError(
                f"{name} image for batch row {row} at block {col} is missing "
                f"(unit ids {units}); refusing to run from an unrestored slot"
            )
        out.append(units)
    return out


def _ops(copier, slots, units, store: bool):
    from atom.model_engine.page_unit_checkpoint import (
        CheckpointRestoreOp,
        CheckpointStoreOp,
    )

    spec = copier.checkpoint_spec
    cls = CheckpointStoreOp if store else CheckpointRestoreOp
    return [
        cls(int(s), tuple(u), spec.image_bytes, spec.layout_id)
        for s, u in zip(slots, units)
    ]


def execute_image_restores(copier, slots, units) -> None:
    if not slots:
        return
    copier.execute_paged_state_copies([], _ops(copier, slots, units, False), 0)


def execute_image_stores(copier, slots, units) -> None:
    if not slots:
        return
    copier.execute_paged_state_copies(_ops(copier, slots, units, True), [], 1)


def flush_image_stores(model) -> None:
    """Store this step's boundary images (the model's own worker)."""
    flush_worker(getattr(model, "_atom_image_worker", None))


def flush_worker(worker, site: str = "after_forward") -> None:
    if worker is None or not worker.pending_stores:
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "image stores reached a CUDA graph capture; a replay would store the "
            "capture batch's slots every step"
        )
    pending = worker.pending_stores
    worker.pending_stores = None
    slots, units = pending
    execute_image_stores(worker.copier, slots, units)
    _store_counts[site] += len(slots)


def plan_image_ops(
    worker,
    md,
    *,
    rows: int,
    slots,
    chunk_start,
    lens,
    fresh_rows,
    block: int,
    is_target: bool,
    name: str,
) -> None:
    """Restores for this step's prefix hits; stores where the scheduler hashed.

    A row bound to a fresh slot with ``chunk_start > 0`` resumes from a cached
    prefix (a cross-request hit, or a preempted request's own blocks): its slot
    is reset and then restored from the image at ``chunk_start`` before the
    forward. A row the scheduler's store plan names (native placement: an
    interval rung, the prompt-end anchor or a demand rung, which prefill was cut
    to end on, or a decode keep) leaves its slot as that boundary's image, stored
    right after the forward into the block vLLM allocated for it (align mode
    keeps the running state block at column ``end // block - 1``) and hashed.
    Results go on ``md`` as ``image_restore_*`` / ``image_store_*``.
    """
    if not rows:
        return
    md.image_restore_rows = []
    md.image_restore_slots = []
    md.image_restore_units = []
    md.image_store_rows = []
    md.image_store_slots = []
    md.image_store_units = []
    resumed = [i for i in fresh_rows if int(chunk_start[i]) > 0]
    if worker is None:
        if resumed and is_target:
            raise RuntimeError(
                f"{name} request(s) resumed at chunk_start "
                f"{[int(chunk_start[i]) for i in resumed]} into a fresh slot, "
                "but no checkpoint image path is bound; the slot would be read "
                "unrestored"
            )
        return
    for i in resumed:
        if int(chunk_start[i]) % block:
            raise RuntimeError(
                f"{name} prefix hit at {int(chunk_start[i])} is not on a "
                f"{block}-token image boundary"
            )
    if resumed:
        md.image_restore_rows = resumed
        md.image_restore_slots = [int(slots[i]) for i in resumed]
        md.image_restore_units = image_unit_ids(
            worker.group_ids,
            resumed,
            [int(chunk_start[i]) // block - 1 for i in resumed],
            name,
        )
    plan = dict(current_image_store_plan())
    # One step's plan is good for that step's build only.
    current_image_store_plan().clear()
    if not plan:
        return
    from atom.plugin.vllm.req_id_passthrough_patch import get_current_req_ids

    req_ids = list(get_current_req_ids() or [])[:rows]
    ends = chunk_start[:rows].astype(np.int64) + lens[:rows].astype(np.int64)
    row_of = {rid: i for i, rid in enumerate(req_ids)}
    stores = []
    for rid, end in plan.items():
        i = row_of.get(rid)
        if i is None or int(ends[i]) != int(end) or int(end) % block:
            raise RuntimeError(
                f"{name} image for {rid} was hashed at {end} but this step "
                f"{'does not carry it' if i is None else f'ends at {int(ends[i])}'}; "
                "the cached image would never be written"
            )
        stores.append(i)
    stores.sort()
    if stores:
        md.image_store_rows = stores
        md.image_store_slots = [int(slots[i]) for i in stores]
        md.image_store_units = image_unit_ids(
            worker.group_ids, stores, [int(ends[i]) // block - 1 for i in stores], name
        )


def run_image_step(worker, md, *, capturing: bool, plan, reset_fn, name: str) -> None:
    """One step's slot work, outside any captured graph, in this order:

    1. stores a step left behind (a build whose forward did not run);
    2. ``plan()`` (``plan_image_ops`` with the bridge's row data);
    3. ``reset_fn(md.reset_slots)``: fresh slots back to the reset state;
    4. restores of this step's prefix hits;
    5. stores registered for right after the forward, where the runner runs
       them (``GPUModelRunner._model_forward``).
    """
    if not capturing and worker is not None and worker.pending_stores:
        # Still stream-ordered after the forward that produced the state and
        # before anything below touches a slot or a block.
        logger.warning("ATOM %s: flushing image stores left by a step", name)
        flush_worker(worker, site="at_next_build")
    if not capturing:
        plan()
    reset_slots = getattr(md, "reset_slots", None)
    if reset_slots:
        reset_fn(reset_slots)
    if worker is None:
        return
    # A prefix hit's fresh slot: reset above, then its boundary image back.
    if getattr(md, "image_restore_slots", None):
        execute_image_restores(
            worker.copier, md.image_restore_slots, md.image_restore_units
        )
    if getattr(md, "image_store_slots", None):
        worker.pending_stores = (md.image_store_slots, md.image_store_units)
        _store_counts["planned"] += len(md.image_store_slots)
