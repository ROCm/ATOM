"""Prefix caching through vLLM-managed state checkpoints, for any ATOM model
whose per-request state lives in slots outside the blocks vLLM hashes.

Such a model (DeepSeek-V4's sliding-window ring and compressor tails, for one)
exposes its paged cache to vLLM as one opaque proxy layer. A block-table hit
brings back the paged history and nothing a resumer needs at the boundary. This
module lets vLLM keep, at the boundaries native ATOM checkpoints at, a state checkpoint
of that slot and hand it back on a hit:

* A state checkpoint is native ATOM's own (``PagedStateCheckpointSpec``), cut
  over ``k = spec.units_per_checkpoint`` PAGE units. vLLM sees it as ``k``
  single-layer ``MambaSpec`` groups in ``align`` mode at the proxy's block size,
  which share the proxy's tensor and block pool. A hit returns the ``k`` state
  checkpoint block ids with the proxy blocks; vLLM only reports the hit when both
  are cached.
* The copy between a slot and ``k`` PAGE units is the model's native
  ``execute_paged_state_copies``, supplied by the model family's adapter.
* A block id means the same unit to vLLM and ATOM but not the same bytes, so every
  vLLM-side byte operation on these blocks is switched off: the align pre-copy,
  new-block zeroing and partial-hit CoW block copies.

Everything model-specific comes through a ``PagedStateCheckpointAdapter`` the model's
bridge registers. This module and ``paged_state_checkpoint_scheduler`` import no model
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


class PagedStateCheckpointAdapter:
    """One model family's side of the state checkpoint path.

    ``sizing(vllm_config)`` returns an object with ``spec`` (native
    ``PagedStateCheckpointSpec``), ``page_bytes`` (what vLLM prices a block at),
    ``num_slots``, ``slot_bytes`` and ``usable_blocks(tensor_blocks)`` (blocks the
    pool may hand out once the slot area is withheld). The copier a bridge builds
    exposes ``checkpoint_spec`` and native
    ``execute_paged_state_copies(stores, restores, descriptor_slot)``.
    """

    #: Family name in logs and errors.
    name: str = "paged-state"
    #: Suffix of the state checkpoint layer names.
    layer_suffix: str = ""
    #: The proxy's and every state checkpoint group's block size.
    block_size: int = 0
    #: vLLM attention backend of the state checkpoint layers (a
    #: ``PagedStateCheckpointBackend``).
    backend_cls: type | None = None
    #: The drafters ``drafter_state_in_checkpoint`` accepts, for the refusal message.
    state_checkpoint_drafters: str = ""

    @property
    def proxy_layer_name(self) -> str:
        raise NotImplementedError

    def matches(self, vllm_config) -> bool:
        raise NotImplementedError

    def sizing(self, vllm_config):
        raise NotImplementedError

    def state_checkpoint_layer_names(self, k: int) -> list[str]:
        return [f"model.layers.{1 + i}.{self.layer_suffix}" for i in range(k)]

    def is_state_checkpoint_layer(self, name: str) -> bool:
        return name.endswith("." + self.layer_suffix)

    def checkpoint_defaults(self) -> tuple[int, bool]:
        """Native (state_checkpoint_interval_tokens, state_checkpoint_demand)."""
        raise NotImplementedError

    def drafter_state_in_checkpoint(self, vllm_config) -> bool:
        """Whether the configured speculative drafter keeps all of its state in
        the target's slot, so a state checkpoint carries it. State checkpoints refuse to
        run with any other drafter: a prefix hit would resume it without its state."""
        return False

    def check_state_checkpoints_supported(self, vllm_config) -> None:
        """Raise ``ValueError`` when state checkpoints cannot run with this config. Runs
        where a worker registers the state checkpoint layers and in EngineCore's
        KV-config hook, each time before the family's sizing."""


_adapters: list[PagedStateCheckpointAdapter] = []


def register_state_checkpoint_adapter(
    adapter: PagedStateCheckpointAdapter,
) -> PagedStateCheckpointAdapter:
    for a in _adapters:
        if type(a) is type(adapter):
            return a
    _adapters.append(adapter)
    return adapter


def adapter_for_config(vllm_config) -> PagedStateCheckpointAdapter | None:
    for a in _adapters:
        if a.matches(vllm_config):
            return a
    return None


def adapter_for_kv_cache_config(cfg) -> PagedStateCheckpointAdapter | None:
    groups = getattr(cfg, "kv_cache_groups", ())
    for a in _adapters:
        if any(a.proxy_layer_name in g.layer_names for g in groups):
            return a
    return None


# ---------------------------------------------------------------------------
# Switches
# ---------------------------------------------------------------------------


def state_checkpoints_on(
    adapter: PagedStateCheckpointAdapter | None, vllm_config
) -> bool:
    """Whether this config's prefix hits are restored from state checkpoints."""
    cc = getattr(vllm_config, "cache_config", None)
    return (
        adapter is not None
        and adapter.matches(vllm_config)
        and bool(getattr(cc, "enable_prefix_caching", False))
    )


# ---------------------------------------------------------------------------
# State checkpoint layers: k single-layer MambaSpec(align) groups
# ---------------------------------------------------------------------------


def _noop_state_checkpoint_copy(state, block_ids, cur_block_idx, num_accepted_tokens):
    """vLLM's align pre-copy, made to copy nothing.

    ATOM writes a state checkpoint itself, through its own carve; vLLM's view of the
    same block id is other blocks' bytes (see module docstring).
    """
    from vllm.model_executor.layers.mamba.mamba_utils import MambaCopySpec

    src = state[block_ids[cur_block_idx]]
    return MambaCopySpec(start_addr=src.data_ptr(), num_elements=0)


def state_checkpoint_copy_funcs():
    return (_noop_state_checkpoint_copy,)


class _PagedStateCheckpointMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support = AttentionCGSupport.UNIFORM_BATCH

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        return None

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return None


class PagedStateCheckpointBackend(AttentionBackend):
    """State checkpoint layers' backend; a family subclasses it with its name and
    block."""

    block_size: int = 0

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return cls.block_size

    @staticmethod
    def get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"
    ):
        raise NotImplementedError(
            "state checkpoint layers are MambaSpec; vLLM sizes them"
        )

    @staticmethod
    def get_impl_cls():
        return nn.Identity

    @staticmethod
    def get_builder_cls():
        return _PagedStateCheckpointMetadataBuilder

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)


class PagedStateCheckpointLayer(nn.Module, AttentionLayerBase):
    """One PAGE unit of a state checkpoint, as vLLM sees it."""

    def __init__(self, prefix: str, part: int, page_bytes: int, adapter):
        super().__init__()
        self.prefix = prefix
        self.part = int(part)
        self.page_bytes = int(page_bytes)
        self.adapter = adapter
        self._atom_state_checkpoint_layer = True
        self._atom_state_checkpoint_group_id = None
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


def register_state_checkpoint_layers(
    adapter, vllm_config, layer_cls=PagedStateCheckpointLayer
) -> list[str]:
    """Register the k state checkpoint layers next to the proxy (worker side)."""
    # The worker builds the model and profiles it before EngineCore's KV-config
    # hook runs; refuse here first.
    adapter.check_state_checkpoints_supported(vllm_config)
    sizing = adapter.sizing(vllm_config)
    sfc = vllm_config.compilation_config.static_forward_context
    names = adapter.state_checkpoint_layer_names(sizing.spec.units_per_checkpoint)
    for i, name in enumerate(names):
        existing = sfc.get(name)
        if isinstance(existing, PagedStateCheckpointLayer):
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


def check_state_checkpoint_groups(cfg, adapter, sizing) -> None:
    from vllm.v1.kv_cache_interface import MambaSpec

    k = sizing.spec.units_per_checkpoint
    proxy_name = adapter.proxy_layer_name
    state_checkpoint_names = set(adapter.state_checkpoint_layer_names(k))
    groups = list(cfg.kv_cache_groups)
    proxy_groups = [g for g in groups if proxy_name in g.layer_names]
    state_checkpoint_groups = [
        g
        for g in groups
        if any(adapter.is_state_checkpoint_layer(n) for n in g.layer_names)
    ]
    problems = []
    if len(proxy_groups) != 1:
        problems.append(f"{len(proxy_groups)} proxy groups")
    if len(state_checkpoint_groups) != k:
        problems.append(
            f"{len(state_checkpoint_groups)} state checkpoint groups, need k={k}"
        )
    found = set()
    for g in state_checkpoint_groups:
        spec = g.kv_cache_spec
        if len(g.layer_names) != 1:
            problems.append(f"state checkpoint group with layers {g.layer_names}")
        if not isinstance(spec, MambaSpec) or spec.mamba_cache_mode != "align":
            problems.append(f"state checkpoint group spec {type(spec).__name__}")
        found.update(g.layer_names)
    if found != state_checkpoint_names:
        problems.append(
            f"state checkpoint layers {sorted(found)} != "
            f"{sorted(state_checkpoint_names)}"
        )
    for g in groups:
        if g.kv_cache_spec.block_size != adapter.block_size:
            problems.append(f"group block_size {g.kv_cache_spec.block_size}")
        if g.kv_cache_spec.page_size_bytes != sizing.page_bytes:
            problems.append(f"group page {g.kv_cache_spec.page_size_bytes}")
    # Every KVCacheTensor is a placement in one backing allocation; the proxy
    # and each state checkpoint must put block b's page at the same bytes.
    placed = {}
    for t in cfg.kv_cache_tensors:
        for i, name in enumerate(t.layers):
            placed[name] = (t.offset + i * t.layer_stride, t.block_stride)
    misplaced = sorted(
        n
        for n in (proxy_name, *state_checkpoint_names)
        if placed.get(n) != (0, sizing.page_bytes)
    )
    if misplaced:
        problems.append(
            f"{misplaced} not at byte 0 with block stride {sizing.page_bytes}"
        )
    if problems:
        raise RuntimeError(
            f"{adapter.name} prefix caching needs 1 proxy group + k single-layer "
            "MambaSpec(align) state checkpoint groups sharing one tensor at block size "
            f"{adapter.block_size}; got: " + "; ".join(problems)
        )


def follow_native_hit_rule(vllm_config) -> bool:
    """Native ATOM's prefix-hit rule under an EAGLE-class drafter (MTP): no
    block dropped beyond the last one, which is always recomputed for logits.

    vLLM drops one more block for EAGLE drafters because a draft's paged KV row
    at a cached block's last position was computed with the token after the
    block, which a reader may not share. A family whose draft has no pages and
    keeps its state in the slot gets that state back from the state checkpoint, as
    native restores it; the one row computed with the writer's next token only steers
    proposals, which the target verifies. Sets vLLM's own
    ``disable_eagle_block_drop`` (EngineCore, before the scheduler is built).
    Returns whether it changed the config.
    """
    spec = getattr(vllm_config, "speculative_config", None)
    use_eagle = getattr(spec, "use_eagle", None)
    if not callable(use_eagle) or not use_eagle():
        return False
    if getattr(spec, "disable_eagle_block_drop", False):
        return False
    spec.disable_eagle_block_drop = True
    logger.info(
        "ATOM state checkpoint prefix caching: the draft's state is in the state "
        "checkpoint, so prefix hits follow native ATOM's rule "
        "(disable_eagle_block_drop=True)"
    )
    return True


def _refit_max_model_len(vllm_config, configs, proxy_name, sizing) -> None:
    """``--max-model-len -1``: vLLM fitted the length to every physical block;
    fit it again to the blocks left once the slot tail is withheld (minus the
    null block), with vLLM's own estimate. The allocation itself is unchanged:
    the slots live in the withheld tail. EngineCore passes a lowered length on
    to the workers and the frontend."""
    from vllm.v1.core.kv_cache_utils import (
        _auto_fit_max_model_len,
        _pool_bytes_per_block,
    )

    mine = [
        c
        for c in configs
        if any(proxy_name in g.layer_names for g in c.kv_cache_groups)
    ]
    if not mine:
        return
    _auto_fit_max_model_len(
        vllm_config,
        [c.kv_cache_groups for c in mine],
        [
            (sizing.usable_blocks(int(c.num_blocks)) - 1)
            * _pool_bytes_per_block(c.kv_cache_groups)
            for c in mine
        ],
    )


def _kv_cache_configs_hook(original, vllm_config, kv_cache_specs, available_memory):
    adapter = adapter_for_config(vllm_config)
    proxy_name = None if adapter is None else adapter.proxy_layer_name
    # Only a model that registered the family's proxy layer is the family's;
    # anything else (e.g. vLLM's own implementation) keeps vLLM's allocator.
    if proxy_name is None or not any(proxy_name in s for s in kv_cache_specs):
        return original(vllm_config, kv_cache_specs, available_memory)
    from atom.plugin.vllm.paged_state_checkpoint_scheduler import (
        STATE_CHECKPOINT_BLOCKS_PER_GROUP,
        StateCheckpointPlacement,
    )

    bs = adapter.block_size
    pc = state_checkpoints_on(adapter, vllm_config)
    if pc:
        adapter.check_state_checkpoints_supported(vllm_config)
        spec = getattr(vllm_config, "speculative_config", None)
        if spec is not None:
            if not adapter.drafter_state_in_checkpoint(vllm_config):
                only = (
                    f" ({adapter.state_checkpoint_drafters})"
                    if adapter.state_checkpoint_drafters
                    else ""
                )
                raise ValueError(
                    f"{adapter.name} prefix caching through state checkpoints "
                    "carries a drafter's state only for a drafter that lives in "
                    f"the target's slot{only}; speculative method "
                    f"{getattr(spec, 'method', None)!r} keeps its own state, which "
                    "a prefix hit would not restore. Pass "
                    "--no-enable-prefix-caching to run it, or drop "
                    "--speculative-config."
                )
            follow_native_hit_rule(vllm_config)
        # Align mode also makes vLLM correct an async spec-decode step's
        # positions before the metadata build, which decode keeps rely on.
        vllm_config.cache_config.mamba_cache_mode = "align"
    sizing = adapter.sizing(vllm_config)
    k = sizing.spec.units_per_checkpoint
    configs = original(vllm_config, kv_cache_specs, available_memory)
    if getattr(vllm_config.model_config, "original_max_model_len", None) == -1:
        _refit_max_model_len(vllm_config, configs, proxy_name, sizing)
    max_model_len = int(vllm_config.model_config.max_model_len)
    per_req = -(-max_model_len // bs)
    placement = (
        StateCheckpointPlacement.from_env(adapter.checkpoint_defaults(), bs)
        if pc
        else None
    )
    # A running request's peak: its KV blocks, and in each state checkpoint group the
    # running block plus the previous one, published or not.
    state_checkpoint_blocks = STATE_CHECKPOINT_BLOCKS_PER_GROUP * k if pc else 0
    for cfg in configs:
        if not any(proxy_name in g.layer_names for g in cfg.kv_cache_groups):
            continue
        if pc:
            check_state_checkpoint_groups(cfg, adapter, sizing)
            logger.info(
                "ATOM %s state checkpoint placement: "
                "state_checkpoint_interval_tokens=%d state_checkpoint_demand=%s "
                "block_size=%d state_checkpoints_per_request=%d; pool check: %d "
                "proxy blocks (max_model_len=%d) + %d x k=%d state checkpoint "
                "blocks + 1 null block = %d",
                adapter.name,
                placement.interval,
                placement.demand,
                bs,
                placement.state_checkpoints_per_request(),
                per_req,
                max_model_len,
                STATE_CHECKPOINT_BLOCKS_PER_GROUP,
                k,
                per_req + state_checkpoint_blocks + 1,
            )
        tensor_blocks = int(cfg.num_blocks)
        usable = sizing.usable_blocks(tensor_blocks)
        # vLLM's block pool keeps one block (the null block) for good.
        need = per_req + state_checkpoint_blocks + 1
        if usable < need:
            state_checkpoints = (
                f" + {STATE_CHECKPOINT_BLOCKS_PER_GROUP}x{k} state checkpoint blocks"
                if pc
                else ""
            )
            raise ValueError(
                f"{adapter.name} plugin pool too small: "
                f"{tensor_blocks} blocks of {sizing.page_bytes} B, "
                f"{tensor_blocks - usable} withheld for {sizing.num_slots} slots of "
                f"{sizing.slot_bytes} B, {usable} left; one max_model_len="
                f"{max_model_len} request needs {need} "
                f"({per_req} KV blocks{state_checkpoints} + 1 null block)"
            )
        cfg.num_blocks = usable
        logger.info(
            "ATOM %s %s: page_bytes=%d page_unit_bytes=%d units_per_checkpoint=%d "
            "image_bytes=%d block_size=%d slots=%d slot_bytes=%d tensor_blocks=%d "
            "reserve_blocks=%d schedulable_blocks=%d kv_groups=%d "
            "mamba_cache_mode=%s long_prefill_token_threshold=%s",
            adapter.name,
            "state checkpoint prefix" if pc else "proxy (no prefix caching)",
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
            logger.debug("ATOM state checkpoint install: %s unavailable (%s)", name, e)
    modules = [m for m in modules if hasattr(m, _KV_CFG_ATTR)]
    if not modules:
        return False
    current = getattr(modules[0], _KV_CFG_ATTR)
    if getattr(current, "_atom_state_checkpoint_patched", False):
        return False

    @functools.wraps(current)
    def patched(vllm_config, kv_cache_specs, available_memory):
        return _kv_cache_configs_hook(
            current, vllm_config, kv_cache_specs, available_memory
        )

    patched._atom_state_checkpoint_patched = True
    for module in modules:
        setattr(module, _KV_CFG_ATTR, patched)
    return True


def _config_has_proxy(cfg) -> bool:
    return adapter_for_kv_cache_config(cfg) is not None


def _apply_zeroing_patch() -> bool:
    from vllm.v1.kv_cache_interface import KVCacheConfig

    prop = KVCacheConfig.__dict__.get("needs_kv_cache_zeroing")
    if prop is None or getattr(prop.fget, "_atom_state_checkpoint_patched", False):
        return False
    original = prop.fget

    def needs_kv_cache_zeroing(self) -> bool:
        # vLLM zeroes by its own view of a block, which is other blocks' rows in
        # an opaque proxy carve. ATOM writes every block before reading it.
        if _config_has_proxy(self):
            return False
        return original(self)

    needs_kv_cache_zeroing._atom_state_checkpoint_patched = True
    KVCacheConfig.needs_kv_cache_zeroing = property(needs_kv_cache_zeroing)
    return True


# ---------------------------------------------------------------------------
# Runner side: lifecycle events, the step's store plan, state checkpoint group ids
# ---------------------------------------------------------------------------

# Worker-side lifecycle events go straight to every slot allocator in the
# process (the target's; an MTP draft only looks its slots up there), before
# this step's metadata is built.
_slot_allocators: list = []
_event_counts = {"finished": 0, "preempted": 0, "resumed": 0}
# This step's state checkpoint stores as the scheduler hashed them: req_id -> boundary.
_store_plan: dict[str, int] = {}
# Speculative decode: this step's decode rows -> the spacing threshold a
# committed end must reach to keep a state checkpoint (decided after the step).
_decode_plan: dict[str, int] = {}


# The state checkpoint worker of each registered family's target model in this process.
_workers: dict[str, StateCheckpointWorker] = {}
# Stores registered by builds: planned; issued right after the step's model
# forward (no drafter) or, with a drafter, at the next build once the forward
# ran; found still pending at the next build with no forward run (refused).
_store_counts = {
    "planned": 0,
    "after_forward": 0,
    "next_build": 0,
    "at_next_build": 0,
    "decode_keep": 0,
}


def current_state_checkpoint_store_plan() -> dict[str, int]:
    return _store_plan


def register_state_checkpoint_worker(worker) -> None:
    """The worker whose stores run after every model forward of this process."""
    _workers[worker.adapter.name] = worker


def store_counts() -> dict:
    return dict(_store_counts)


def register_slot_allocator(allocator) -> None:
    """``allocator.release(req_ids)`` gets vLLM's finished / preempted / resumed ids."""
    if not any(a is allocator for a in _slot_allocators):
        _slot_allocators.append(allocator)


def _note_scheduler_output(runner, scheduler_output) -> None:
    """One step's scheduler events, taken before the runner applies them.

    Slots come back on finished / preempted / resumed ids, this step's
    placement plans replace the last step's, and whole-block copies on a proxy
    pool are refused.
    """
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
    _store_plan.update(
        getattr(scheduler_output, "atom_state_checkpoint_stores", None) or {}
    )
    _decode_plan.clear()
    _decode_plan.update(
        getattr(scheduler_output, "atom_state_checkpoint_decode_candidates", None) or {}
    )
    copies = getattr(scheduler_output, "kv_cache_block_copies", None)
    if copies and _config_has_proxy(getattr(runner, "kv_cache_config", None)):
        raise RuntimeError(
            "vLLM scheduled whole-block copies on an ATOM proxy pool; its view "
            "of a block is not ATOM's carve, so they would corrupt it"
        )


# Where each runner first receives a step's scheduler output: V1 in
# `_update_states`; Model Runner V2 in `finish_requests`, which `execute_model`
# calls first and which is where V2 drops finished and preempted requests.
_STEP_ENTRY_METHODS = ("_update_states", "finish_requests")


def _patch_runner_class(runner_cls) -> bool:
    """Hook one runner class. Returns whether this call installed anything."""
    entry = next((m for m in _STEP_ENTRY_METHODS if hasattr(runner_cls, m)), None)
    if entry is None:
        raise RuntimeError(
            f"{runner_cls.__module__}.{runner_cls.__name__} has none of "
            f"{_STEP_ENTRY_METHODS}; ATOM state slots would never be released"
        )
    original_entry = getattr(runner_cls, entry)
    if getattr(original_entry, "_atom_state_checkpoint_patched", False):
        return False

    @functools.wraps(original_entry)
    def wrapped_entry(self, scheduler_output, *args, **kwargs):
        _note_scheduler_output(self, scheduler_output)
        return original_entry(self, scheduler_output, *args, **kwargs)

    wrapped_entry._atom_state_checkpoint_patched = True
    setattr(runner_cls, entry, wrapped_entry)

    original_init_kv = runner_cls.initialize_kv_cache

    @functools.wraps(original_init_kv)
    def wrapped_initialize_kv_cache(self, kv_cache_config, *args, **kwargs):
        result = original_init_kv(self, kv_cache_config, *args, **kwargs)
        sfc = self.compilation_config.static_forward_context
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            for name in group.layer_names:
                layer = sfc.get(name)
                if getattr(layer, "_atom_state_checkpoint_layer", False):
                    layer._atom_state_checkpoint_group_id = gid
        return result

    wrapped_initialize_kv_cache._atom_state_checkpoint_patched = True
    runner_cls.initialize_kv_cache = wrapped_initialize_kv_cache
    if entry == "_update_states":
        install_store_sites(runner_cls)
    return True


def _apply_runner_patches() -> bool:
    """Hook every GPU model runner class this vLLM ships, V1 and V2 alike.

    The two classes share no base, so hooking one leaves the other's slots
    unreleased. The store sites are installed on V1 only; a deployment that
    would need them under V2 is refused where its checkpoints are bound.
    """
    from atom.plugin.vllm.vllm_runtime import gpu_model_runner_classes

    installed = False
    for runner_cls in gpu_model_runner_classes():
        installed = _patch_runner_class(runner_cls) or installed
    return installed


def install_store_sites(runner_cls) -> None:
    """Where a step's stores run, on vLLM's V1 runner: after the step's last
    forward into the slot, on the same stream, outside any capture, before the
    next step's restores.

    A FULL cudagraph replay never enters the model's Python, so without a
    drafter that is right after ``_model_forward``. With one, the draft's layers
    live in the same slot and write it later in the step (``sample_tokens``
    runs the drafter), so the stores go where native ATOM drains them: into the
    next batch's build (``run_state_checkpoint_step``), once both forwards ran. The
    target forward only marks them as run, so a build whose forward never ran is still
    refused rather than copied.
    """
    original = runner_cls._model_forward

    @functools.wraps(original)
    def wrapped_model_forward(self, *args, **kwargs):
        output = original(self, *args, **kwargs)
        for worker in list(_workers.values()):
            if getattr(self, "speculative_config", None) is None:
                flush_worker(worker)
            elif worker.pending_stores:
                worker.forward_ran = True
        return output

    wrapped_model_forward._atom_state_checkpoint_patched = True
    runner_cls._model_forward = wrapped_model_forward


def install() -> tuple[bool, bool, bool, bool]:
    """Install every state checkpoint hook. Idempotent; a no-op for unregistered models.

    Returns which hooks this call installed: (get_kv_cache_configs, kv zeroing,
    runner events, placement scheduler).
    """
    from atom.plugin.vllm.paged_state_checkpoint_scheduler import (
        apply_scheduler_patches,
    )

    return (
        _apply_kv_cache_configs_patch(),
        _apply_zeroing_patch(),
        _apply_runner_patches(),
        apply_scheduler_patches(),
    )


# ---------------------------------------------------------------------------
# Per-step state checkpoint ops (worker)
# ---------------------------------------------------------------------------


class StateCheckpointWorker:
    """A target model's state checkpoint path on one worker: its copier, the state
    checkpoint groups' ids and the stores a step leaves for after its forward."""

    def __init__(self, adapter, copier, group_ids):
        self.adapter = adapter
        self.copier = copier
        self.group_ids = list(group_ids)
        self.pending_stores = None
        # With a drafter: the target forward after the build that planned
        # ``pending_stores`` ran (they are issued at the next build).
        self.forward_ran = False
        # Speculative decode: req_id -> (slot, {boundary: state checkpoint unit ids})
        # for every boundary the last step could commit to; the next build stores
        # the one it did commit to, if any.
        self.decode_candidates: dict[str, tuple[int, dict]] = {}


def state_checkpoint_unit_ids(
    group_ids, rows, cols, name: str = "paged-state"
) -> list[tuple[int, ...]]:
    """`k` state checkpoint block ids per (batch row, block column), from the host block
    table."""
    from atom.plugin.vllm.req_id_passthrough_patch import get_current_input_batch

    batch = get_current_input_batch()
    if batch is None or group_ids is None:
        raise RuntimeError(
            f"{name} state checkpoint ops need the vLLM input batch and the state "
            "checkpoint group ids; the runner patches did not run"
        )
    tables = [batch.block_table[g].block_table.np for g in group_ids]
    out = []
    for row, col in zip(rows, cols):
        units = tuple(int(t[row, col]) for t in tables)
        if any(u <= 0 for u in units):
            raise RuntimeError(
                f"{name} state checkpoint for batch row {row} at block {col} is "
                f"missing (unit ids {units}); refusing to run from an unrestored slot"
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


def execute_state_checkpoint_restores(copier, slots, units) -> None:
    if not slots:
        return
    copier.execute_paged_state_copies([], _ops(copier, slots, units, False), 0)


def execute_state_checkpoint_stores(copier, slots, units) -> None:
    if not slots:
        return
    copier.execute_paged_state_copies(_ops(copier, slots, units, True), [], 1)


def flush_state_checkpoint_stores(model) -> None:
    """Store this step's boundary state checkpoints (the model's own worker)."""
    flush_worker(getattr(model, "_atom_state_checkpoint_worker", None))


def flush_worker(worker, site: str = "after_forward") -> None:
    if worker is None or not worker.pending_stores:
        return
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "state checkpoint stores reached a CUDA graph capture; a replay would "
            "store the capture batch's slots every step"
        )
    pending = worker.pending_stores
    worker.pending_stores = None
    worker.forward_ran = False
    slots, units = pending
    execute_state_checkpoint_stores(worker.copier, slots, units)
    _store_counts[site] += len(slots)


def plan_state_checkpoint_ops(
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
    is reset and then restored from the state checkpoint at ``chunk_start`` before the
    forward. A row the scheduler's store plan names (native placement: an
    interval rung, the prompt-end anchor or a demand rung, which prefill was cut
    to end on, or a decode keep) leaves its slot as that boundary's state checkpoint,
    stored after the step's forwards (``run_state_checkpoint_step``) into the block vLLM
    allocated for it (align mode
    keeps the running state block at column ``end // block - 1``) and hashed.
    Results go on ``md`` as ``state_checkpoint_restore_*`` /
    ``state_checkpoint_store_*``.
    """
    if not rows:
        return
    md.state_checkpoint_restore_rows = []
    md.state_checkpoint_restore_slots = []
    md.state_checkpoint_restore_units = []
    md.state_checkpoint_store_rows = []
    md.state_checkpoint_store_slots = []
    md.state_checkpoint_store_units = []
    resumed = [i for i in fresh_rows if int(chunk_start[i]) > 0]
    if worker is None:
        if resumed and is_target:
            raise RuntimeError(
                f"{name} request(s) resumed at chunk_start "
                f"{[int(chunk_start[i]) for i in resumed]} into a fresh slot, "
                "but no state checkpoint path is bound; the slot would be read "
                "unrestored"
            )
        return
    from atom.plugin.vllm.req_id_passthrough_patch import get_current_req_ids

    req_ids = list(get_current_req_ids() or [])[:rows]
    if worker.decode_candidates or _decode_plan:
        starts = _exact_starts(rows, name)
        store_decode_keeps(worker, req_ids, starts, fresh_rows)
        record_decode_candidates(worker, req_ids, slots, starts, lens, block, name)
    for i in resumed:
        if int(chunk_start[i]) % block:
            raise RuntimeError(
                f"{name} prefix hit at {int(chunk_start[i])} is not on a "
                f"{block}-token state checkpoint boundary"
            )
    if resumed:
        md.state_checkpoint_restore_rows = resumed
        md.state_checkpoint_restore_slots = [int(slots[i]) for i in resumed]
        md.state_checkpoint_restore_units = state_checkpoint_unit_ids(
            worker.group_ids,
            resumed,
            [int(chunk_start[i]) // block - 1 for i in resumed],
            name,
        )
    plan = dict(current_state_checkpoint_store_plan())
    # One step's plan is good for that step's build only.
    current_state_checkpoint_store_plan().clear()
    if not plan:
        return
    ends = chunk_start[:rows].astype(np.int64) + lens[:rows].astype(np.int64)
    row_of = {rid: i for i, rid in enumerate(req_ids)}
    stores = []
    for rid, end in plan.items():
        i = row_of.get(rid)
        if i is None or int(ends[i]) != int(end) or int(end) % block:
            raise RuntimeError(
                f"{name} state checkpoint for {rid} was hashed at {end} but this step "
                f"{'does not carry it' if i is None else f'ends at {int(ends[i])}'}; "
                "the cached state checkpoint would never be written"
            )
        stores.append(i)
    stores.sort()
    if stores:
        md.state_checkpoint_store_rows = stores
        md.state_checkpoint_store_slots = [int(slots[i]) for i in stores]
        md.state_checkpoint_store_units = state_checkpoint_unit_ids(
            worker.group_ids, stores, [int(ends[i]) // block - 1 for i in stores], name
        )


def _exact_starts(rows: int, name: str):
    """Each row's first position this step. Under async speculative decode the
    builder's ``seq_lens_cpu_upper_bound`` assumes every in-flight draft was
    accepted; in align mode (state checkpoints on) vLLM applies the correction to the
    input batch before the metadata build, so this is exact."""
    from atom.plugin.vllm.req_id_passthrough_patch import get_current_input_batch

    batch = get_current_input_batch()
    if batch is None:
        raise RuntimeError(
            f"{name} decode keeps under speculative decode need the vLLM input "
            "batch; the runner patches did not run"
        )
    return np.asarray(batch.num_computed_tokens_cpu[:rows], dtype=np.int64)


def store_decode_keeps(worker, req_ids, starts, fresh_rows) -> None:
    """Native ``hash_decode_blocks`` keeps at the committed end of a step, out
    of the slot that step left. That end is this step's exact start; the last
    step's forwards (target and draft) are done and this one's have not run.
    A row on a fresh slot (resumed) is not the slot the last step wrote."""
    cands, worker.decode_candidates = worker.decode_candidates, {}
    if not cands:
        return
    fresh = {int(i) for i in fresh_rows}
    row_of = {rid: i for i, rid in enumerate(req_ids)}
    slots, units = [], []
    for rid, (slot, by_end) in cands.items():
        i = row_of.get(rid)
        if i is None or i in fresh:
            continue
        unit_ids = by_end.get(int(starts[i]))
        if unit_ids is not None:
            slots.append(slot)
            units.append(unit_ids)
    if slots:
        if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "decode keep state checkpoint stores reached a CUDA graph capture"
            )
        execute_state_checkpoint_stores(worker.copier, slots, units)
        _store_counts["decode_keep"] += len(slots)


def record_decode_candidates(worker, req_ids, slots, starts, lens, block, name):
    """The state checkpoint blocks of every boundary this step's decode rows could
    commit to, at or past their spacing threshold (vLLM allocates state checkpoint
    blocks up to the scheduled end, so each is there)."""
    plan = dict(_decode_plan)
    _decode_plan.clear()
    out = {}
    for i, rid in enumerate(req_ids):
        threshold = plan.get(rid)
        if threshold is None:
            continue
        start = int(starts[i])
        end = start + int(lens[i])
        ends = [
            b
            for b in range((start // block + 1) * block, end + 1, block)
            if b >= threshold
        ]
        if ends:
            unit_ids = state_checkpoint_unit_ids(
                worker.group_ids, [i] * len(ends), [b // block - 1 for b in ends], name
            )
            out[rid] = (int(slots[i]), dict(zip(ends, unit_ids)))
    worker.decode_candidates = out


def run_state_checkpoint_step(
    worker, md, *, capturing: bool, plan, reset_fn, name: str
) -> None:
    """One step's slot work, outside any captured graph, in this order:

    1. the stores the last build left behind: with a drafter, issued here once
       its forward ran (native drains them into the next batch's build, after
       the target and the draft forward); refused if it never ran;
    2. ``plan()`` (``plan_state_checkpoint_ops`` with the bridge's row data);
    3. ``reset_fn(md.reset_slots)``: fresh slots back to the reset state;
    4. restores of this step's prefix hits;
    5. stores registered for after the step's forwards
       (``install_store_sites``).
    """
    if not capturing and worker is not None and worker.forward_ran:
        flush_worker(worker, "next_build")
    if not capturing and worker is not None and worker.pending_stores:
        # The last build planned stores that its forward never issued, so the
        # slots still hold an earlier position's state while the scheduler has
        # already hashed (and may already have handed to a later step's prefix
        # hit) the state checkpoint blocks for the planned boundary. Neither copying nor
        # unhashing from here can make that right.
        _store_counts["at_next_build"] += len(worker.pending_stores[0])
        logger.error(
            "ATOM %s: state checkpoint stores planned by the previous build were "
            "never issued",
            name,
        )
        raise RuntimeError(
            f"{name} state checkpoint stores planned by the previous step's metadata "
            "build were never issued: that step's model forward never ran after the "
            "build (or ran outside GPUModelRunner._model_forward). Their state "
            "checkpoint blocks are already hashed for the planned boundary and copying "
            "the slot now would cache an earlier position's state."
        )
    if not capturing:
        plan()
    reset_slots = getattr(md, "reset_slots", None)
    if reset_slots:
        reset_fn(reset_slots)
    if worker is None:
        return
    # A prefix hit's fresh slot: reset above, then its boundary state checkpoint back.
    if getattr(md, "state_checkpoint_restore_slots", None):
        execute_state_checkpoint_restores(
            worker.copier,
            md.state_checkpoint_restore_slots,
            md.state_checkpoint_restore_units,
        )
    if getattr(md, "state_checkpoint_store_slots", None):
        worker.pending_stores = (
            md.state_checkpoint_store_slots,
            md.state_checkpoint_store_units,
        )
        worker.forward_ran = False
        _store_counts["planned"] += len(md.state_checkpoint_store_slots)
