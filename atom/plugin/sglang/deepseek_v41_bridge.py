# SPDX-License-Identifier: MIT
"""Run DeepSeek-V4.1 (CSA2) on SGLang's plugin path against an ATOM-owned pool.

SGLang 0.5.22 renames V4.1 to ``DeepseekV4ForCausalLM`` (``model_type`` stays
``deepseek_v41``), forces ``page_size=256`` and builds a DSV4 pool.

SGLang owns scheduling, page allocation and sampling; ATOM owns the model and
the CSA2 pool: ``ATOMDeepSeekV41ProxyKVPool`` is one uint8 arena (PAGEs, then
STATE slots) sized by ``ATOMDeepSeekV41PoolConfigurator``. Each ``ForwardBatch``
is translated into ATOM's scheduled batch and staged by ATOM's own builder
once per step, outside the forward.

The translation relies on: SGLang page id == V4.1 PAGE id (block tables are
``req_to_token[req, ::256] // 256``; page 0 is SGLang's padding page); STATE
slot == ``req_pool_idx`` (unique among live requests, row 0 is padding; a
request starting at position 0 resets its slot, so a prefix hit, which needs
restored state, is refused); the window lives in the STATE ring, so SGLang
gets a plain paged allocator (``needs_paged_swa_allocator = False``).
"""

from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("atom")

DEEPSEEK_V41_ARCH = "DeepseekV41ForCausalLM"
DEEPSEEK_V41_MODEL_TYPES = ("deepseek_v41", "deepseek_v41_text")
# SGLang forces page_size 256 for DeepSeek-V4 off NPU (its model override);
# V4.1's PAGE follows it.
ATOM_DEEPSEEK_V41_SGLANG_BLOCK_SIZE = 256

# This process's metadata builder (`_install_pool_sizing_patch`).
_V41_BUILDER = None


def is_deepseek_v41_config(config: Any) -> bool:
    """Whether a flat HF config is DeepSeek-V4.1. Every caller passes SGLang's
    normalized config (``deepseek_v41``, renamed to a V4 architecture) or
    ATOM's text config (``deepseek_v41_text``), so the model type decides.
    ``resolve_model_arch_spec`` asks this first for every plugin model and
    reads its config with ``getattr`` throughout; this check keeps that."""
    return getattr(config, "model_type", None) in DEEPSEEK_V41_MODEL_TYPES


def sglang_v41_kv_cache_dtype(kv_cache_dtype: str) -> str:
    """The V4.1 pool format for the operator's ``--kv-cache-dtype`` (the raw
    ServerArgs value). SGLang's DSV4 hook admits only ``bfloat16`` and
    ``fp8_e4m3`` and resolves ``auto`` to ``fp8_e4m3`` in its own bags, so
    ``auto`` is refused rather than served as a BF16 pool SGLang believes
    is FP8."""
    if kv_cache_dtype == "bfloat16":
        return "bf16"
    resolved = " (SGLang resolves auto to fp8_e4m3 for DeepSeek-V4.1)"
    raise ValueError(
        "DeepSeek-V4.1 on the SGLang plugin serves a BF16 KV pool only; launch "
        f"with --kv-cache-dtype bfloat16, got {kv_cache_dtype!r}"
        + (resolved if kv_cache_dtype == "auto" else "")
    )


# Native's FP4PTPC_RECIPE in SGLang's spelling.
FP4PTPC_SGLANG_RECIPE = (
    "DeepSeek-V4.1 FP4PTPC needs its routed experts converted to per-channel "
    "FP8 (recipes/atom_sglang/DeepSeek-V4.1-Flash.md): "
    '--model-loader-extra-config \'{"online_quant_config": '
    '{"layer_quant_config": {"*.ffn.experts*": "ptpc_fp8"}}}\''
)


def refuse_fp4ptpc_without_online_quant(atom_config: Any) -> None:
    """Refuse FP4PTPC without its experts converted to FP8, as native does."""
    if atom_config.hf_config.fp4ptpc and atom_config.online_quant_config is None:
        raise ValueError(FP4PTPC_SGLANG_RECIPE)


# --- Launch contract ---


def refuse_unsupported_sglang_features() -> None:
    """Fail before weights load on SGLang features this bridge cannot serve."""
    from sglang.srt import runtime_context as sglang_config

    memory = sglang_config.get_memory()
    exec_ = sglang_config.get_exec()
    disagg = sglang_config.get_disagg()
    parallel = sglang_config.get_parallel()
    schedule = sglang_config.get_schedule()
    chunk = schedule.chunked_prefill_size
    graphs = exec_.graph.cuda_graph_config
    unsupported = (
        # Unchunked, SGLang admits a first prompt whole, past the builder's
        # buffers: the forward would refuse it and take the server down.
        (
            "disabled chunked prefill (--chunked-prefill-size must be > 0)",
            chunk is None or int(chunk) <= 0,
        ),
        (
            "radix cache (launch with --disable-radix-cache)",
            not memory.disable_radix_cache,
        ),
        ("hierarchical cache", memory.enable_hierarchical_cache),
        # Not validated with the ATOM pool.
        ("--enable-unified-memory", memory.enable_unified_memory),
        # SGLang refuses it itself on hardware other than CUDA and gfx950.
        (
            "encoder SWA bounded replay",
            exec_.features.enable_encoder_swa_bounded_replay,
        ),
        # Every prefill row runs every layer here.
        (
            "decoder SWA bounded replay",
            exec_.features.enable_decoder_swa_bounded_replay,
        ),
        # The pool is sized by SGLang's hybrid-SWA full token count.
        ("--disable-hybrid-swa-memory", schedule.disable_hybrid_swa_memory),
        # The ATOM arena is allocated outside SGLang's memory-saver region.
        ("--enable-memory-saver", exec_.features.enable_memory_saver),
        (
            "speculative decoding",
            sglang_config.get_spec().speculative_algorithm is not None,
        ),
        ("PD disaggregation", str(disagg.disaggregation_mode) != "null"),
        ("DP attention", parallel.enable_dp_attention),
        ("decode context parallelism (--dcp-size > 1)", parallel.dcp_size > 1),
        (
            "CUDA graphs (launch with --disable-cuda-graph)",
            graphs.decode.backend != "disabled" or graphs.prefill.backend != "disabled",
        ),
        # Image rows need ATOM's vision path, which the wrapper does not feed.
        (
            "multimodal requests (launch with --language-model-only)",
            not disagg.language_model_only,
        ),
    )
    refused = [name for name, enabled in unsupported if enabled]
    if refused:
        raise ValueError(
            "DeepSeek-V4.1 on the SGLang plugin does not support " + ", ".join(refused)
        )


def align_atom_config_with_sglang(atom_config: Any) -> None:
    """Size the builder by SGLang's resolved schedule (``Config`` saw the raw
    ServerArgs): its fixed buffers must hold every step SGLang can form.

    No step exceeds these: SGLang runs at most ``max_running_requests``
    requests, a decode step has one token per request, and its
    ``PrefillAdder`` charges mixed-in decode rows against the chunk budget, so
    a step never has more tokens than the largest of ``max_running_requests``,
    ``chunked_prefill_size`` and ``max_prefill_tokens`` (``Config`` also keeps
    a 16384-token floor)."""
    from sglang.srt.runtime_context import get_schedule

    schedule = get_schedule()
    atom_config.max_num_batched_tokens = max(
        int(atom_config.max_num_batched_tokens),
        int(schedule.chunked_prefill_size),
        int(schedule.max_prefill_tokens),
        int(schedule.max_running_requests),
    )
    atom_config.max_num_seqs = int(schedule.max_running_requests)
    # The builder sizes its score workspace from this fraction.
    atom_config.gpu_memory_utilization = float(schedule.mem_fraction_static)


# --- Metadata builder ---


# H2D publication: native's ModelRunner owns a registry over its
# ``forward_vars``; here the shim runner and the cache's host sources do.

# `PagedAttentionCache`'s host sources, published by `prepare_model_inputs`.
_V41_CACHE_SOURCES = (
    "_cursor_staging",
    "tentative_staging",
    "_reset_slots",
    "_reset_histories",
)


def _v41_publication_owner(variables, device):
    """A publication owner over the ``CpuGpuBuffer``s in ``variables``."""
    from atom.utils import CpuGpuBuffer
    from atom.utils.h2d import PublicationOwner

    event = torch.cuda.Event()
    owner = PublicationOwner(device, event)
    members = {}
    for name, buffer in variables.items():
        if isinstance(buffer, CpuGpuBuffer):
            binding = owner.bind(buffer, name, unit=buffer.publication_unit)
            if buffer.publication_group:
                members.setdefault(buffer.publication_group, []).append(binding)
    groups = {name: owner.group(name, bindings) for name, bindings in members.items()}
    event.record()
    return owner, groups


@contextmanager
def _v41_publication_epoch(owner):
    owner.begin()
    try:
        yield
    except BaseException:
        owner.fail()
        raise
    else:
        owner.finish()


def _init_v41_publication(runner, builder):
    """The shim runner's registry: `_v41_publication_owner` over its
    ``forward_vars`` plus one group per phase the builder declares."""
    owner, groups = _v41_publication_owner(runner.forward_vars, builder.device)
    for name, names in builder.h2d_group_members().items():
        groups[name] = owner.group(
            name, [runner.forward_vars[item]._publication for item in names]
        )
    return owner, groups


def _make_builder_cls():
    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )

    class SGLangDeepseekV41MetadataBuilder(DeepseekV41MetadataBuilder):
        """ATOM's V4.1 builder, staged once per step in one publication epoch."""

        def prepare_model_inputs(self, input_ids, metadata):
            cache = metadata.cache
            if cache._sglang_h2d_owner is None:
                sources = {name: getattr(cache, name) for name in _V41_CACHE_SOURCES}
                cache._sglang_h2d_owner, _ = _v41_publication_owner(
                    sources, self.device
                )
            with _v41_publication_epoch(cache._sglang_h2d_owner):
                super().prepare_model_inputs(input_ids, metadata)

        def _prepare(self, batch, running_bs, running_tokens, **kwargs):
            runner = self.model_runner
            if runner.h2d_owner is None:
                runner.h2d_owner, runner.h2d_groups = _init_v41_publication(
                    runner, self
                )
            with _v41_publication_epoch(runner.h2d_owner):
                # No runner token inputs carry the query metadata on this path.
                self.publish_query_metadata(
                    np.asarray(batch.num_scheduled_tokens, np.int32),
                    running_bs,
                    running_tokens,
                )
                return super()._prepare(batch, running_bs, running_tokens, **kwargs)

    return SGLangDeepseekV41MetadataBuilder


def init_deepseek_v41_builder(atom_config: Any):
    """Create this process's builder (idempotent) before SGLang's memory
    profile, so the KV budget counts its score workspace."""
    global _V41_BUILDER
    if _V41_BUILDER is not None:
        return _V41_BUILDER

    from atom.utils import CpuGpuBuffer

    device = torch.device("cuda", torch.cuda.current_device())
    max_num_batched_tokens = int(atom_config.max_num_batched_tokens)
    runner = SimpleNamespace(
        block_size=ATOM_DEEPSEEK_V41_SGLANG_BLOCK_SIZE,
        device=device,
        config=atom_config,
        max_bs=int(atom_config.max_num_seqs),
        max_num_batched_tokens=max_num_batched_tokens,
        h2d_owner=None,
        h2d_groups=None,
        forward_vars={
            # `prepare_batch_step` takes its request-start dtype from here.
            "positions": CpuGpuBuffer(
                max_num_batched_tokens,
                dtype=torch.int64,
                device=device,
                pin_memory=True,
            )
        },
    )
    _V41_BUILDER = _make_builder_cls()(runner)
    logger.info(
        "ATOM DeepSeek-V4.1 SGLang builder: block_size=%d max_bs=%d "
        "max_num_batched_tokens=%d kv=%s index=%s paged_bytes=%d state_bytes=%d",
        runner.block_size,
        runner.max_bs,
        max_num_batched_tokens,
        atom_config.kv_cache_dtype,
        atom_config.index_cache_dtype,
        int(_V41_BUILDER.geometry.paged_bytes),
        int(_V41_BUILDER.geometry.state_bytes),
    )
    return _V41_BUILDER


def get_deepseek_v41_builder():
    if _V41_BUILDER is None:
        raise RuntimeError(
            "DeepSeek-V4.1 metadata builder is not initialized; it is created "
            "before SGLang profiles memory for the KV pool"
        )
    return _V41_BUILDER


# --- KV pool ---


def _arena_bytes(geometry, num_pages: int, num_slots: int) -> int:
    return int(geometry.paged_extents(num_pages)[1]) + num_slots * int(
        geometry.state_bytes
    )


def _make_backed_cache_cls():
    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache

    class SGLangBackedPagedAttentionCache(PagedAttentionCache):
        """``PagedAttentionCache`` over the proxy pool's arena."""

        def __init__(self, *args, backing: torch.Tensor, **kwargs):
            # `super().__init__` calls `_allocate_backing`.
            self._external_backing = backing
            # The publication owner over its host sources, made at the first
            # staged step (`SGLangDeepseekV41MetadataBuilder`).
            self._sglang_h2d_owner = None
            super().__init__(*args, **kwargs)

        def _allocate_backing(self, size, device):
            # A BF16 pool: the whole cache is one flat arena.
            backing = self._external_backing
            if backing.numel() < size:
                raise ValueError(
                    f"A CSA2 backing of {backing.numel()} bytes is short of the "
                    f"{size} this geometry needs"
                )
            view = backing[:size]
            # Core's own allocations are zeroed; the cursor reset and every
            # cold read assume it.
            view.zero_()
            return view

    return SGLangBackedPagedAttentionCache


def _make_proxy_pool_cls():
    from sglang.srt.mem_cache.base_swa_memory_pool import BaseSWAKVPool

    class ATOMDeepSeekV41ProxyKVPool(BaseSWAKVPool):
        """SGLang's DSV4 pool (installed under its name) over ATOM's CSA2
        arena. SGLang builds it only in ``_build_dsv4_kv_pool``, which passes
        every argument below."""

        is_atom_v41_proxy_pool = True
        # SGLang reads these off any DSV4 pool (no SWA allocator, no rings).
        _unified_kv = False
        needs_paged_swa_allocator = False
        request_window = None
        full_kv_pool = None
        swa_kv_pool = None
        c4_kv_pool = None
        c128_kv_pool = None
        custom_mem_pool = None
        enable_custom_mem_pool = False
        layer_transfer_counter = None

        def __init__(
            self,
            max_num_reqs: int,
            swa_size: int,
            page_size: int,
            swa_page_size: int,
            dtype: torch.dtype,
            layer_num: int,
            device: str,
            start_layer: int,
            end_layer: int,
            num_req_slots: int,
            full_size: int,
            **_unused_kwargs: Any,
        ) -> None:
            if _V41_BUILDER is None:
                # No sizing hook ran: the builder is outside the profiled budget.
                raise RuntimeError(
                    "DeepSeek-V4.1 proxy pool built before the metadata builder; "
                    "SGLang did not profile memory through "
                    "KVCacheConfigurator._profile_available_bytes"
                )
            builder = _V41_BUILDER
            geometry = builder.geometry
            self.max_num_reqs = int(max_num_reqs)
            self.page_size = int(page_size)
            self.swa_page_size = int(swa_page_size)
            self.dtype = dtype
            self.store_dtype = torch.uint8
            self.device = device
            self.start_layer = int(start_layer)
            self.end_layer = int(end_layer)
            self.layer_num = int(layer_num)
            # `size` is the token capacity SGLang's allocator schedules against:
            # its hybrid-SWA full count (--disable-hybrid-swa-memory, which
            # leaves it None, is refused at launch).
            self.size = int(full_size)
            self.full_size = self.size
            self.swa_size = int(swa_size)
            self.size_swa = self.swa_size
            self.full_to_swa_index_mapping = None

            # SGLang's padding page 0 is still addressable: count it.
            self.num_pages = self.size // self.page_size + 1
            # One STATE slot per req_to_token row (row 0 is padding).
            self.num_req_slots = int(num_req_slots)
            self.num_slots = self.num_req_slots
            total_bytes = _arena_bytes(geometry, self.num_pages, self.num_slots)
            self.raw_arena = torch.empty(total_bytes, dtype=torch.uint8, device=device)
            # Reported by /get_internal_state and the scheduler metrics, in GiB.
            self.mem_usage = total_bytes / (1 << 30)

            builder.num_blocks = self.num_pages
            builder.cache = _make_backed_cache_cls()(
                geometry,
                self.num_pages,
                self.num_slots,
                builder.device,
                max_tokens=builder.max_num_batched_tokens,
                workspace=builder.score_workspace,
                backing=self.raw_arena,
            )
            logger.info(
                "ATOM DeepSeek-V4.1 proxy pool: %d PAGEs x %d B + %d STATE slots "
                "x %d B = %.2f GiB (SGLang tokens %d)",
                self.num_pages,
                int(geometry.paged_bytes),
                self.num_slots,
                int(geometry.state_bytes),
                self.mem_usage,
                self.size,
            )

        def get_kv_size_bytes(self):
            return self.raw_arena.nbytes

        def register_mapping(self, full_to_swa_index_mapping: torch.Tensor) -> None:
            self.full_to_swa_index_mapping = full_to_swa_index_mapping

        def translate_loc_from_full_to_swa(
            self, kv_indices: torch.Tensor
        ) -> torch.Tensor:
            raise NotImplementedError(
                "ATOM V4.1 proxy pool has no SWA token pool; the window lives "
                "in ATOM's STATE ring"
            )

        def get_state_buf_infos(self):
            return ([], [], [])

        def get_contiguous_buf_infos(self):
            return ([self.raw_arena.data_ptr()], [self.raw_arena.nbytes], [1])

        def get_kv_buffer(self, layer_id: int):
            raise NotImplementedError("ATOM V4.1 proxy pool is owned by ATOM")

        def set_kv_buffer(self, *args, **kwargs) -> None:
            raise NotImplementedError("ATOM V4.1 proxy pool is written by ATOM kernels")

        def get_key_buffer(self, layer_id: int):
            raise NotImplementedError("ATOM V4.1 proxy pool is owned by ATOM")

        def get_value_buffer(self, layer_id: int):
            raise NotImplementedError("ATOM V4.1 proxy pool is owned by ATOM")

        def move_kv_cache(self, tgt_loc: torch.Tensor, src_loc: torch.Tensor) -> None:
            raise NotImplementedError(
                "DeepSeek-V4.1 on the SGLang plugin does not support KV "
                "relocation yet; launch with --disable-radix-cache"
            )

    return ATOMDeepSeekV41ProxyKVPool


def _is_atom_v41_target(kvc) -> bool:
    """Whether SGLang's KV configurator is building this plugin's V4.1 pool."""
    # Installed from `prepare_model`'s V4.1 branch only, so ``kvc.model`` is
    # the ATOM wrapper. No draft worker exists: speculative decoding is refused.
    return is_deepseek_v41_config(kvc.model_config.hf_config)


def _make_pool_configurator_cls():
    from sglang.srt.model_executor.pool_configurator import (
        MemoryPoolConfig,
        MemoryPoolConfigurator,
    )

    class ATOMDeepSeekV41PoolConfigurator(MemoryPoolConfigurator):
        """Size SGLang's V4.1 token pool by what ATOM's arena costs instead of
        SGLang's DSV4 layout: ``geometry.paged_bytes`` per PAGE plus the
        padding PAGE, one STATE slot per request row, and the
        ``req_to_token`` rows nothing else budgets."""

        def __init__(self, kvc) -> None:
            builder = get_deepseek_v41_builder()
            geometry = builder.geometry
            self.geometry = geometry
            self.page_size = int(kvc.page_size)
            self.context_len = int(kvc.model_config.context_len)
            from sglang.srt.runtime_context import get_schedule

            # Resolved before the model is built; DP attention is refused.
            self.reqs = int(get_schedule().max_running_requests)

        def _pool_bytes(self, pages: int, slots: int) -> int:
            # `pages` usable PAGEs plus the padding PAGE, the request rows'
            # STATE slots and the int32 req_to_token rows beside them.
            return (
                _arena_bytes(self.geometry, pages + 1, slots)
                + slots * self.context_len * 4
            )

        def _config(self, tokens: int) -> MemoryPoolConfig:
            tokens = tokens // self.page_size * self.page_size
            return MemoryPoolConfig(
                max_total_num_tokens=tokens,
                full_max_total_num_tokens=tokens,
                swa_max_total_num_tokens=0,
            )

        def calculate_pool_sizes(
            self, available_bytes: int, page_size: int
        ) -> MemoryPoolConfig:
            assert page_size == self.page_size
            # One req_to_token row per request plus SGLang's padding row 0.
            slots = self.reqs + 1
            fixed = self._pool_bytes(0, slots)
            pages = max(0, (available_bytes - fixed) // int(self.geometry.paged_bytes))
            # The extents are aligned per owner; step back until they fit.
            while pages > 0 and self._pool_bytes(pages, slots) > available_bytes:
                pages -= 1
            logger.info(
                "ATOM DeepSeek-V4.1 pool sizing: available=%.2f GiB, "
                "fixed=%.2f GiB (%d request slots), %d PAGEs x %d B -> %d tokens",
                available_bytes / (1 << 30),
                fixed / (1 << 30),
                slots,
                pages,
                int(self.geometry.paged_bytes),
                pages * self.page_size,
            )
            return self._config(pages * self.page_size)

        def calculate_pool_sizes_from_max_tokens(
            self, max_total_num_tokens: int, page_size: int
        ) -> MemoryPoolConfig:
            assert page_size == self.page_size
            return self._config(max_total_num_tokens)

    return ATOMDeepSeekV41PoolConfigurator


def _install_pool_sizing_patch() -> None:
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    from sglang.srt.model_executor import pool_configurator

    if getattr(pool_configurator, "_atom_v41_sizing_patched", False):
        return
    configurator_cls = _make_pool_configurator_cls()
    original_factory = pool_configurator.create_memory_pool_configurator

    def create_memory_pool_configurator(kvc):
        if _is_atom_v41_target(kvc):
            return configurator_cls(kvc)
        return original_factory(kvc)

    # kv_cache_configurator imports the factory inside its methods.
    pool_configurator.create_memory_pool_configurator = create_memory_pool_configurator

    original_profile = KVCacheConfigurator._profile_available_bytes

    def profile_available_bytes(self, pre_model_load_memory):
        # SGLang post-processes the weights after `load_weights` returns, so
        # this is the first point where the builder sees the memory they
        # leave; its buffers then come out of the profiled budget.
        if _is_atom_v41_target(self):
            init_deepseek_v41_builder(self.model.atom_config)
        return original_profile(self, pre_model_load_memory)

    KVCacheConfigurator._profile_available_bytes = profile_available_bytes
    pool_configurator._atom_v41_sizing_patched = True


def install_deepseek_v41_proxy_pool_patch() -> None:
    """Make SGLang size and build the V4.1 proxy pool instead of its DSV4 pool."""
    import sglang.srt.mem_cache.deepseek_v4_memory_pool as dsv4_pool

    _install_pool_sizing_patch()
    original_pool_cls = dsv4_pool.DeepSeekV4TokenToKVPool
    if getattr(original_pool_cls, "is_atom_v41_proxy_pool", False):
        return
    proxy_cls = _make_proxy_pool_cls()
    dsv4_pool.DeepSeekV4TokenToKVPool = proxy_cls
    dsv4_pool.ATOMDeepSeekV41ProxyKVPool = proxy_cls
    # Modules that imported the class by name (the configurator) too.
    for module in list(sys.modules.values()):
        if getattr(module, "DeepSeekV4TokenToKVPool", None) is original_pool_cls:
            module.DeepSeekV4TokenToKVPool = proxy_cls


# --- Per-forward translation ---


def _cpu_ints(cpu_value, gpu_value, n: int) -> np.ndarray:
    # SGLang's host mirror when it has one (never req_pool_indices on DECODE).
    value = cpu_value if cpu_value is not None else gpu_value
    if torch.is_tensor(value):
        value = value[:n].detach().cpu().numpy()
    return np.asarray(value[:n], dtype=np.int64)


def _req_to_token() -> torch.Tensor:
    # ForwardBatch carries no pools; SGLang's forward context does.
    from sglang.srt.model_executor.forward_context import get_req_to_token_pool

    return get_req_to_token_pool().req_to_token


def _block_rows(req_to_token, req_pool_indices, ends, block_size: int):
    # DECODE / EXTEND rows only (IDLE is refused), each ending at >= 1 token.
    widths = [-(-int(end) // block_size) for end in ends]
    widest = max(widths)
    table = (
        req_to_token[req_pool_indices, : widest * block_size : block_size] // block_size
    ).to(torch.int32)
    table = table.cpu().numpy()
    rows = [table[i, : widths[i]].copy() for i in range(len(widths))]
    _refuse_padding_page(rows)
    return rows


def _refuse_padding_page(rows) -> None:
    # SGLang never hands out its padding page 0: such a row was never written.
    if any(np.any(row <= 0) for row in rows):
        raise ValueError(
            "DeepSeek-V4.1 block table names SGLang's padding page 0 in a live "
            "request's row"
        )


def _refuse_shared_or_padding_slots(slots, count: int) -> None:
    if np.unique(slots).size != count or np.any(slots <= 0):
        raise ValueError(
            f"DeepSeek-V4.1 needs one live req_pool_idx per request; got {slots}"
        )


def build_v41_scheduled_batch(forward_batch, block_size: int):
    """ATOM's scheduled batch for one SGLang ``ForwardBatch``: ``DECODE`` or
    ``EXTEND`` (the eager runner turns ``MIXED`` into it); others refused."""
    mode = forward_batch.forward_mode
    if not (mode.is_decode() or mode.is_context_parallel_extend()):
        raise NotImplementedError(
            f"DeepSeek-V4.1 SGLang bridge does not translate {mode} yet"
        )
    bs = int(forward_batch.batch_size)
    ends = _cpu_ints(forward_batch.seq_lens_cpu, forward_batch.seq_lens, bs)
    if mode.is_decode():
        # positions = seq_lens - 1: seq_lens already counts the decode token.
        lengths = np.ones(bs, dtype=np.int64)
        prefix = ends - 1
    else:
        lengths = _cpu_ints(
            forward_batch.extend_seq_lens_cpu, forward_batch.extend_seq_lens, bs
        )
        prefix = _cpu_ints(
            forward_batch.extend_prefix_lens_cpu, forward_batch.extend_prefix_lens, bs
        )
    if np.any(prefix + lengths != ends):
        raise ValueError(
            "DeepSeek-V4.1 forward batch lengths disagree: "
            f"prefix={prefix.tolist()} lengths={lengths.tolist()} "
            f"ends={ends.tolist()}"
        )
    # SGLang mirrors req_pool_indices on the host for EXTEND only.
    slots = _cpu_ints(
        forward_batch.req_pool_indices_cpu, forward_batch.req_pool_indices, bs
    ).astype(np.int32)
    _refuse_shared_or_padding_slots(slots, bs)
    rows = _block_rows(
        _req_to_token(), forward_batch.req_pool_indices[:bs], ends, block_size
    )
    return SimpleNamespace(
        is_dummy_run=False,
        req_ids=list(forward_batch.rids[:bs]),
        num_scheduled_tokens=lengths.astype(np.int32),
        context_lens=ends,
        # SGLang's ForwardBatch does not say whether a chunk finishes its
        # prompt, so every request counts as finishing (None).
        is_final_chunk=None,
        replaying={},
        state_slots_committed=[int(s) for s in slots],
        block_tables=rows,
        total_seqs_num=bs,
        total_tokens_num=int(lengths.sum()),
    )


def build_deepseek_v41_forward_metadata(forward_batch):
    """Stage this forward's CSA2 step once, before the model (``_prepare``,
    ``prepare_model_inputs``, Engram), marked ``staged_outside_forward`` so
    the model's ``v41_begin_forward`` does not stage it again."""
    from atom.model_ops.engram.device.staging import EngramStagedRows

    mode = forward_batch.forward_mode
    builder = get_deepseek_v41_builder()
    batch = build_v41_scheduled_batch(forward_batch, builder.block_size)
    input_ids = forward_batch.input_ids
    running_tokens = int(input_ids.shape[0])
    running_bs = batch.total_seqs_num
    if mode.is_decode():
        metadata, _ = builder.prepare_decode(batch, running_bs, running_tokens, 1)
    else:
        metadata, _ = builder.prepare_prefill(batch, running_bs, running_tokens)
    builder.prepare_model_inputs(input_ids, metadata)
    # Engram's staging too (its cursor advance rides on it).
    rows = metadata.engram_embeddings
    if isinstance(rows, EngramStagedRows):
        rows.stage()
        rows.join()
    metadata.staged_outside_forward = True
    return metadata


def log_deepseek_v41_runtime(model: Any) -> None:
    """Startup proof: what actually got constructed for this rank (the
    wrapper's ``install_adapters``, once per wrapper)."""
    attn_classes = sorted({type(layer.attn).__name__ for layer in model.layers})
    logger.info(
        "ATOM DeepSeek-V4.1 SGLang runtime: model=%s replay=%s attention=%s",
        type(model).__name__,
        model.replay,
        attn_classes,
    )
