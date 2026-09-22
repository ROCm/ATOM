# SPDX-License-Identifier: MIT
"""Run DeepSeek-V4.1 (CSA2) on vLLM's plugin path against a proxy KV cache.

vLLM owns block allocation and hands a backend a block table; V4.1's paged
runtime owns a pool whose geometry no vLLM KV-cache spec can describe -- 40
layers of window ring, two compression ratios, an FP8 index plane, an Engram
cursor, and a per-request STATE region whose size scales with concurrency
rather than with history. The bridge reconciles the two the way the V4 bridge
does: declare one fake attention layer whose ``FullAttentionSpec`` is a byte
arena (uint8, one head, a synthetic ``head_size``), let vLLM size and allocate
it, then carve V4.1's own planes out of that storage.

Two things make V4.1 simpler than V4 here, and one makes it harder.

Simpler: V4.1's PAGE size and vLLM's block size are both 256, so a vLLM block
id *is* a V4.1 page id -- no remapping, and the block table rows go straight
into ``RequestSpan.block_ids``. And V4.1 binds no per-layer cache: the cache
and the step travel on the metadata object every layer reads out of ATOM's
forward context, so there is nothing to rebind per module.

Harder: the pool holds two currencies in one contiguous region. PAGE bytes
scale with history (``geometry.paged_bytes`` per page); STATE bytes scale with
in-flight requests (``geometry.state_bytes`` per slot), and
``V41PoolGeometry.window()`` derives the ring's start from the *absolute* end
of the paged extent, so the two cannot be separate allocations. vLLM only
knows how to size a per-block quantity, so the STATE region is bought as a
tail of extra blocks that the KV-cache manager is then told not to hand out --
see ``deepseek_v41_state_reserve_patch``, which reduces ``num_blocks`` by
``v41_proxy_state_reserve_blocks()`` while leaving the tensor's byte size
alone. ``bind_deepseek_v41_proxy_cache`` re-checks that arithmetic against the
tensor it actually got, so a missing patch fails loudly instead of silently
overlapping PAGE and STATE.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from atom.plugin.vllm.state_slot_allocator import StateSlotAllocator

logger = logging.getLogger(__name__)

# The one fake layer vLLM sees. Named under `model.layers.0.` so vLLM's own
# layer-name parsing (which expects a `layers.<idx>` segment) keeps working.
ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME = "model.layers.0.atom_deepseek_v41_proxy"
# V4.1's PAGE is 256 tokens. vLLM's block size is forced to the same number
# (`get_supported_kernel_block_sizes`), which is what makes a vLLM block id and
# a V4.1 page id the same id.
ATOM_DEEPSEEK_V41_BLOCK_SIZE = 256
# EntryMajorArena retypes carved planes (uint8 -> bf16 / fp8 / fp32), so the
# region it is handed has to start on a boundary every one of those dtypes can
# address.
ATOM_DEEPSEEK_V41_PROXY_ALIGNMENT = 256

_V41_ARCHITECTURES = ("DeepseekV41ForCausalLM",)
_V41_MODEL_TYPE = "deepseek_v41"

# Model path -> normalized text config. `atom.config.get_hf_config` reads and
# normalizes the checkpoint's config.json; doing that once per process keeps
# the sizing helpers below cheap enough to call from a patch.
_V41_TEXT_CONFIG_CACHE: dict[str, object] = {}


def is_deepseek_v41_vllm_config(vllm_config) -> bool:
    """Whether this vLLM config is serving a DeepSeek-V4.1 checkpoint.

    Read off the architectures list first (what the registry dispatched on) and
    fall back to ``model_type``, because the AutoConfig shim registered in
    ``atom.plugin.vllm.register`` is what supplies the architectures and a
    config built by some other route may only carry the type.
    """
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        return False
    architectures = getattr(hf_config, "architectures", None) or ()
    if any(str(arch) in _V41_ARCHITECTURES for arch in architectures):
        return True
    return str(getattr(hf_config, "model_type", "")) == _V41_MODEL_TYPE


def v41_text_config(vllm_config):
    """The normalized V4.1 text config, from the checkpoint.

    Not ``vllm_config.model_config.hf_config``: that is the AutoConfig shim
    vLLM built, which carries the fields vLLM reads and not the normalized
    field set ``build_attention_topology`` / ``V41PoolGeometry`` need.
    ``atom.config.get_hf_config`` is the single authority for that
    normalization, and the runtime model is constructed from the same object.
    """
    from atom.config import get_hf_config

    model = str(vllm_config.model_config.model)
    config = _V41_TEXT_CONFIG_CACHE.get(model)
    if config is None:
        config = get_hf_config(model)
        _V41_TEXT_CONFIG_CACHE[model] = config
    return config


def v41_kv_cache_dtype(vllm_config) -> str:
    """Map vLLM's ``--kv-cache-dtype`` onto ATOM's V4.1 pool dtype.

    V4.1's pool is either unpacked bf16 or the packed fp4 layout; there is no
    fp8 variant, so the fp8 spellings are refused rather than silently
    downgraded -- the choice changes ``paged_bytes`` and therefore the proxy
    layer's ``head_size``, and a mismatch between the sizing vLLM allocated
    with and the geometry the kernels read would corrupt the pool.
    """
    cache_config = getattr(vllm_config, "cache_config", None)
    cache_dtype = getattr(cache_config, "cache_dtype", None) if cache_config else None
    dtype = str(cache_dtype or "auto")
    if dtype == "nvfp4":
        return "fp4"
    if dtype in ("auto", "bfloat16", "float16"):
        return "bf16"
    raise ValueError(
        f"DeepSeek-V4.1 on the vLLM plugin supports --kv-cache-dtype auto/"
        f"bfloat16/nvfp4 (nvfp4 selects ATOM's packed fp4 pool); got {dtype!r}"
    )


def v41_proxy_geometry(vllm_config):
    """The ``V41PoolGeometry`` the proxy pool is sized and carved by.

    One authority for the whole bridge: the proxy layer's ``head_size``, the
    state reserve, and the ``PagedAttentionCache`` bound at cache-bind time all
    come from this object, so the bytes vLLM allocated and the bytes the
    kernels address can never disagree. Cached on the vLLM config because it is
    read once per forward-context build on the fallback path.
    """
    cached = getattr(vllm_config, "_atom_v41_geometry", None)
    if cached is not None:
        return cached
    # Imported here rather than at module scope: this module is imported by the
    # KV-cache reserve patch, which runs before ATOM's attention stack (and its
    # aiter dependency) is needed.
    from atom.model_ops.attentions.deepseek_v41.backend import (
        build_v41_pool_geometry,
    )

    geometry = build_v41_pool_geometry(
        v41_text_config(vllm_config),
        ATOM_DEEPSEEK_V41_BLOCK_SIZE,
        packed=v41_kv_cache_dtype(vllm_config) == "fp4",
        # Speculative decoding (MTP / DSpark) is refused on the plugin path;
        # see `enforce_deepseek_v41_constraints` in the platform.
        speculative_tokens=0,
    )
    try:
        vllm_config._atom_v41_geometry = geometry
    except AttributeError:  # pragma: no cover - frozen config
        pass
    return geometry


def v41_proxy_head_size(vllm_config) -> int:
    """Synthetic ``head_size`` that makes one proxy block hold one V4.1 PAGE.

    ``FullAttentionSpec.page_size_bytes`` is
    ``2 * block_size * num_kv_heads * head_size * itemsize``; with uint8 and one
    head that is ``512 * head_size``. V4.1's ``paged_bytes`` is a multiple of
    512 for every supported configuration, so this is exact rather than rounded
    up -- the ceiling is kept only so an unforeseen geometry over-allocates
    instead of under-allocating.
    """
    paged_bytes = int(v41_proxy_geometry(vllm_config).paged_bytes)
    return -(-paged_bytes // (2 * ATOM_DEEPSEEK_V41_BLOCK_SIZE))


def v41_proxy_page_size_bytes(vllm_config) -> int:
    """Bytes one proxy block occupies -- vLLM's per-block accounting unit."""
    return 2 * ATOM_DEEPSEEK_V41_BLOCK_SIZE * v41_proxy_head_size(vllm_config)


def v41_proxy_state_reserve_blocks(vllm_config) -> int:
    """Proxy blocks to withhold from vLLM so the STATE region has room.

    V4.1's pool is ``paged_extents(pages)[1] + num_slots * state_bytes`` bytes
    of one contiguous allocation, and only the first term is a per-block
    quantity vLLM can express. The second is bought as this many extra blocks
    at the tail of the same tensor, which the reserve patch then subtracts from
    the block count the KV-cache manager is allowed to hand out. Returns 0 for
    anything that is not V4.1, so the patch is a no-op for every other model.
    """
    if not is_deepseek_v41_vllm_config(vllm_config):
        return 0
    geometry = v41_proxy_geometry(vllm_config)
    num_slots = max(1, int(vllm_config.scheduler_config.max_num_seqs))
    state_bytes = int(geometry.state_bytes) * num_slots
    page_size_bytes = v41_proxy_page_size_bytes(vllm_config)
    return -(-state_bytes // page_size_bytes)


class AtomDeepseekV41ProxyMetadataBuilder(AttentionMetadataBuilder):
    """Snapshot the step's host-side batch description; build nothing on device.

    ATOM's own ``DeepseekV41MetadataBuilder`` does the real work, but it has to
    run *inside* the forward: ``prepare_model_inputs`` writes the Engram cursor
    and advances per-request state, which has to happen once, in order, around
    the model call. So this builder only collects what
    ``CommonAttentionMetadata`` cannot supply without a device sync -- request
    ids, per-request query lengths, computed-token counts and block-table rows
    -- and attaches them for :func:`atom_deepseek_v41_forward_context` to
    consume.

    ``_cudagraph_support`` is ``NEVER``: V4.1's step carries host-side Python
    state (Engram hashing, tentative staging, per-slot resets) that cannot be
    replayed from a captured graph, and the platform forces
    ``cudagraph_mode=NONE`` to match.
    """

    _cudagraph_support = AttentionCGSupport.NEVER

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.vllm_config = vllm_config
        self.device = device

    def build(
        self, common_prefix_len: int, common_attn_metadata, fast_build: bool = False
    ):
        if common_prefix_len:
            raise ValueError(
                "ATOM DeepSeek-V4.1 proxy does not support cascade attention"
            )
        if common_attn_metadata is None:
            return common_attn_metadata
        common_attn_metadata.atom_v41_snapshot = snapshot_v41_batch(
            common_attn_metadata
        )
        return common_attn_metadata

    def build_for_cudagraph_capture(self, common_attn_metadata):
        raise NotImplementedError(
            "ATOM DeepSeek-V4.1 runs eager on the vLLM plugin path; "
            "cudagraph_mode must be NONE"
        )


class AtomDeepseekV41ProxyBackend(AttentionBackend):
    forward_includes_kv_cache_update = True

    @staticmethod
    def get_name() -> str:
        return "ATOM_DEEPSEEK_V41_PROXY"

    @staticmethod
    def get_supported_kernel_block_sizes():
        return [ATOM_DEEPSEEK_V41_BLOCK_SIZE]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        return ATOM_DEEPSEEK_V41_BLOCK_SIZE

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # Block-major: permuting the leading (2, num_blocks) pair puts one
        # block's bytes contiguous, which is what lets the flat view below be a
        # page-indexed arena rather than two interleaved halves.
        return (
            (1, 0, 2, 3, 4) if not include_num_layers_dimension else (1, 0, 2, 3, 4, 5)
        )

    @staticmethod
    def get_impl_cls():
        return nn.Identity

    @staticmethod
    def get_builder_cls():
        return AtomDeepseekV41ProxyMetadataBuilder

    @classmethod
    def full_cls_name(cls) -> tuple[str, str]:
        return (cls.__module__, cls.__qualname__)


class AtomDeepseekV41ProxyAttention(nn.Module, AttentionLayerBase):
    """The one layer vLLM sees, so it sizes and allocates V4.1's pool."""

    def __init__(self, prefix: str = ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME):
        super().__init__()
        self.prefix = prefix
        self._atom_v41_proxy_layer = True
        self.kv_cache = torch.tensor([])
        self.impl = nn.Identity()

    def get_attn_backend(self) -> type[AttentionBackend]:
        return AtomDeepseekV41ProxyBackend

    def get_kv_cache_spec(self, vllm_config) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=ATOM_DEEPSEEK_V41_BLOCK_SIZE,
            num_kv_heads=1,
            head_size=v41_proxy_head_size(vllm_config),
            dtype=torch.uint8,
        )


def register_deepseek_v41_proxy_layer(
    vllm_config,
    layer_name: str = ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME,
) -> AtomDeepseekV41ProxyAttention:
    """Put the proxy layer in vLLM's static forward context. Idempotent."""
    sfc = vllm_config.compilation_config.static_forward_context
    existing = sfc.get(layer_name)
    if isinstance(existing, AtomDeepseekV41ProxyAttention):
        return existing
    if existing is not None:
        raise ValueError(f"Duplicate layer name: {layer_name}")
    proxy = AtomDeepseekV41ProxyAttention(prefix=layer_name)
    sfc[layer_name] = proxy
    return proxy


def snapshot_v41_batch(common_attn_metadata):
    """Host-side description of the step, with no device sync.

    ``CommonAttentionMetadata`` carries the per-request tensors on the device;
    its CPU accessors are deprecated precisely because they force a D2H. Every
    field V4.1 needs already exists on the host in vLLM's ``InputBatch``, which
    the req-id pass-through patch exposes, so read it from there.

    Falls back to the device tensors when the patch is not installed (unit
    tests, or a vLLM whose runner method names moved). The fallback also loses
    the ``req_id`` slot key and substitutes each request's first block id --
    stable for a request's lifetime, which is all the allocator needs, at the
    cost of the copy the patch exists to avoid.
    """
    num_reqs = int(common_attn_metadata.num_reqs)
    qsl = common_attn_metadata.query_start_loc_cpu[: num_reqs + 1].numpy()
    query_lens = (qsl[1:] - qsl[:-1]).astype(np.int32)

    input_batch = None
    try:
        from atom.plugin.vllm.req_id_passthrough_patch import get_current_input_batch

        input_batch = get_current_input_batch()
    # The accessor is an ATOM patch over another engine's internals; absent is
    # the caller's ordinary "fall back to the device tensors" case.
    except Exception:  # noqa: BLE001
        input_batch = None

    block_table_np = None
    req_ids = None
    if input_batch is not None:
        try:
            req_ids = list(input_batch.req_ids)[:num_reqs]
            num_computed = np.asarray(
                input_batch.num_computed_tokens_cpu[:num_reqs], dtype=np.int64
            )
            block_table_np = input_batch.block_table[0].block_table.np
        except Exception:  # noqa: BLE001
            req_ids = None
            block_table_np = None
    if block_table_np is None:
        block_table_np = common_attn_metadata.block_table_tensor.cpu().numpy()
        seq_lens_np = common_attn_metadata.seq_lens[:num_reqs].cpu().numpy()
        num_computed = seq_lens_np.astype(np.int64) - query_lens

    # `context_lens` in ATOM's batch protocol is the request's end position
    # after this step -- vLLM's `seq_lens`.
    ends = (num_computed + query_lens).astype(np.int64)
    block_rows = []
    for i in range(num_reqs):
        needed = -(-int(ends[i]) // ATOM_DEEPSEEK_V41_BLOCK_SIZE)
        row = block_table_np[i, :needed]
        block_rows.append(tuple(int(block) for block in row))
    if req_ids is None:
        # No pass-through patch: key on the request's first block, which vLLM
        # keeps for the request's lifetime.
        req_ids = [row[0] if row else -1 for row in block_rows]
    return SimpleNamespace(
        num_reqs=num_reqs,
        req_ids=req_ids,
        query_lens=query_lens,
        num_computed=num_computed,
        ends=ends,
        block_rows=block_rows,
        total_tokens=int(query_lens.sum()),
    )


def make_deepseek_v41_metadata_builder(atom_config, vllm_config, device):
    """Build ATOM's own V4.1 metadata builder against a shim model runner.

    ``DeepseekV41MetadataBuilder`` reads a handful of attributes off the ATOM
    ``ModelRunner`` it normally lives on, and allocates its fixed-address
    staging buffers into that runner's ``forward_vars``. None of that needs a
    runner, so hand it the attributes and an empty ``forward_vars`` dict.

    ``positions`` is seeded here rather than left to the builder because
    ``CommonAttentionBuilder.__init__`` does not create it and
    ``prepare_batch_step`` takes the dtype of its request-start array from it;
    int64 matches what native ATOM allocates.
    """
    from atom.model_ops.attentions.deepseek_v41.backend import (
        DeepseekV41MetadataBuilder,
    )
    from atom.utils import CpuGpuBuffer

    max_num_batched_tokens = int(vllm_config.scheduler_config.max_num_batched_tokens)
    runner = SimpleNamespace(
        block_size=ATOM_DEEPSEEK_V41_BLOCK_SIZE,
        device=device,
        config=atom_config,
        max_bs=max(1, int(vllm_config.scheduler_config.max_num_seqs)),
        max_num_batched_tokens=max_num_batched_tokens,
        forward_vars={
            "positions": CpuGpuBuffer(
                max_num_batched_tokens,
                dtype=torch.int64,
                device=device,
                pin_memory=torch.device(device).type != "cpu",
            )
        },
    )
    return DeepseekV41MetadataBuilder(runner)


def bind_deepseek_v41_proxy_cache(
    model,
    builder,
    vllm_config,
    layer_name: str = ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME,
) -> bool:
    """Carve V4.1's pool out of vLLM's proxy allocation. Idempotent.

    Returns False while the proxy tensor is still empty (profiling, before vLLM
    has decided a block count), which is the caller's signal to run the forward
    on a private scratch cache instead.
    """
    sfc = vllm_config.compilation_config.static_forward_context
    proxy = sfc.get(layer_name)
    if not isinstance(proxy, AtomDeepseekV41ProxyAttention):
        return False
    if not isinstance(proxy.kv_cache, torch.Tensor) or proxy.kv_cache.numel() == 0:
        return False
    ptr = proxy.kv_cache.untyped_storage().data_ptr()
    if getattr(model, "_atom_v41_proxy_cache_ptr", None) == ptr:
        return True

    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache

    physical = proxy.kv_cache.permute(1, 0, 2, 3, 4)
    if not physical.is_contiguous():
        raise ValueError("DeepSeek-V4.1 proxy cache must be block-major contiguous")
    raw = physical.reshape(-1)
    if raw.storage_offset() % ATOM_DEEPSEEK_V41_PROXY_ALIGNMENT:
        raise RuntimeError(
            f"DeepSeek-V4.1 proxy KV storage offset {raw.storage_offset()} is not "
            f"{ATOM_DEEPSEEK_V41_PROXY_ALIGNMENT}B-aligned; EntryMajorArena cannot "
            "retype carved planes safely"
        )

    geometry = v41_proxy_geometry(vllm_config)
    num_slots = max(1, int(vllm_config.scheduler_config.max_num_seqs))
    pages = int(vllm_config.cache_config.num_gpu_blocks or 0)
    if pages <= 0:
        return False
    # `pages` is what the KV-cache manager will hand out; the tensor also holds
    # the withheld tail the STATE region lives in. If the reserve patch never
    # ran, the two overlap -- catch that here rather than at the first request.
    required = int(geometry.paged_extents(pages)[1]) + num_slots * int(
        geometry.state_bytes
    )
    if raw.numel() < required:
        raise RuntimeError(
            "DeepSeek-V4.1 proxy pool is too small: "
            f"{raw.numel()} bytes for {pages} PAGEs + {num_slots} STATE slots "
            f"({required} bytes needed). The STATE tail reserve "
            f"({v41_proxy_state_reserve_blocks(vllm_config)} blocks) was not "
            "applied -- apply_vllm_v41_state_reserve_patch() must run before "
            "vLLM determines the KV-cache config."
        )

    builder.num_blocks = pages
    builder.cache = PagedAttentionCache(
        geometry,
        pages,
        num_slots,
        builder.device,
        max_tokens=int(vllm_config.scheduler_config.max_num_batched_tokens),
        backing=raw,
    )
    if not hasattr(model, "_atom_v41_slot_allocator"):
        model._atom_v41_slot_allocator = StateSlotAllocator(num_slots)
    model._atom_v41_proxy_cache_ptr = ptr
    logger.info(
        "ATOM DeepSeek-V4.1: bound proxy pool -- %d PAGEs x %d B + %d STATE slots "
        "x %d B out of %d B (kv_cache_dtype=%s)",
        pages,
        int(geometry.paged_bytes),
        num_slots,
        int(geometry.state_bytes),
        raw.numel(),
        v41_kv_cache_dtype(vllm_config),
    )
    return True


def get_deepseek_v41_proxy_metadata_from_vllm_context(
    layer_name: str = ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME,
):
    from vllm.forward_context import get_forward_context, is_forward_context_available

    if not is_forward_context_available():
        return None
    meta = get_forward_context().attn_metadata
    if isinstance(meta, dict):
        return meta.get(layer_name)
    if isinstance(meta, list) and meta and isinstance(meta[0], dict):
        return meta[0].get(layer_name)
    return None


def _v41_scheduled_batch(snapshot, slot_allocator):
    """ATOM's scheduled-batch protocol, from the host snapshot.

    ``state_slots_committed`` covers every row, including the zero-token rows
    ``_prepare`` skips, because it is indexed by the same ``i`` as the zipped
    per-request arrays.
    """
    slots, _reset = slot_allocator.assign(snapshot.req_ids, snapshot.num_computed)
    return SimpleNamespace(
        is_dummy_run=False,
        req_ids=snapshot.req_ids,
        num_scheduled_tokens=snapshot.query_lens,
        context_lens=snapshot.ends,
        state_slots_committed=slots,
        block_tables=snapshot.block_rows,
        total_seqs_num=len(slots),
        total_tokens_num=int(snapshot.total_tokens),
    )


def _v41_dummy_batch(running_tokens, *, max_req_tokens, max_reqs):
    """Synthetic requests for the profiling / warmup forward.

    ``_prepare`` routes a dummy batch onto a private scratch
    ``PagedAttentionCache`` and fabricates its own PAGE ids and slot, so this
    never touches the serving pool -- which may not exist yet.

    The tokens are spread over as many requests as it takes to keep every one
    of them inside ``max_req_tokens``, the way ATOM's own ``warmup_model``
    splits its token budget. One request holding the whole forward would own
    ``running_tokens / block_size`` PAGEs, and vLLM profiles at
    ``max_num_batched_tokens`` -- twice ``max_model_len`` at the shipped
    defaults -- while every ``block_tables`` row is only as wide as
    ``max_model_len`` worth of PAGEs. Packing beyond that width is not a
    capacity to grow into: it is a request longer than the model can serve.

    A forward too wide to cover even at ``max_reqs`` full-length requests
    keeps the rows it can fill and leaves the rest of the width as padding,
    which the step already stages (position 0, batch id -1) and every V4.1
    kernel already skips.
    """
    num_reqs = max(1, min(max_reqs, -(-running_tokens // max_req_tokens)))
    base, spare = divmod(running_tokens, num_reqs)
    lengths = tuple(
        min(max_req_tokens, base + (1 if i < spare else 0)) for i in range(num_reqs)
    )
    return SimpleNamespace(
        is_dummy_run=True,
        req_ids=tuple(range(num_reqs)),
        num_scheduled_tokens=lengths,
        context_lens=lengths,
        state_slots_committed=(),
        block_tables=(),
        total_seqs_num=num_reqs,
        total_tokens_num=sum(lengths),
    )


@contextmanager
def atom_deepseek_v41_forward_context(
    *,
    atom_config,
    builder,
    input_ids,
    positions,
    slot_allocator=None,
    common_attn_metadata=None,
    force_dummy: bool = False,
    proxy_layer_name: str = ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME,
):
    """Drive one V4.1 step and publish it on ATOM's forward context.

    Yields the positions tensor the step staged, which is what the model must
    be called with: it spans the forward's full width (padding included) and
    carries each request's absolute positions, where vLLM's own tensor is only
    defined on the rows it scheduled.

    ``running_tokens`` is taken from ``input_ids``, not from the snapshot's
    token count, so ``step.width`` matches the row count the model will really
    run. ``v41_begin_forward`` asserts exactly that equality, and DP/TP token
    padding makes the two differ. The pad rows get position 0 and batch id -1,
    which every V4.1 kernel already skips.
    """
    from atom.utils.forward_context import (
        Context,
        reset_forward_context,
        set_forward_context,
    )

    if common_attn_metadata is None:
        common_attn_metadata = get_deepseek_v41_proxy_metadata_from_vllm_context(
            proxy_layer_name
        )
    snapshot = getattr(common_attn_metadata, "atom_v41_snapshot", None)
    running_tokens = int(input_ids.shape[0]) if input_ids is not None else 0

    dummy = (
        force_dummy
        or snapshot is None
        or slot_allocator is None
        or builder.cache is None
        or snapshot.num_reqs == 0
    )
    if dummy:
        running_tokens = max(running_tokens, 1)
        batch = _v41_dummy_batch(
            running_tokens,
            max_req_tokens=builder.block_table_cols
            * builder.block_ratio
            * builder.block_size,
            max_reqs=builder.max_bs,
        )
        running_bs = batch.total_seqs_num
    else:
        batch = _v41_scheduled_batch(snapshot, slot_allocator)
        running_bs = int(snapshot.num_reqs)
        running_tokens = max(running_tokens, batch.total_tokens_num)

    metadata, step_positions = builder._prepare(batch, running_bs, running_tokens)
    # Engram embeddings, the per-request state reset and the cursor advance --
    # everything that must happen once per step, before any layer runs.
    builder.prepare_model_inputs(input_ids, metadata)

    is_prefill = metadata.state.value.startswith("prefill")
    context = Context(
        positions=step_positions,
        is_prefill=is_prefill,
        is_dummy_run=dummy,
        scheduled_bs=running_bs,
        scheduled_tokens=running_tokens,
        running_bs=running_bs,
        running_tokens=running_tokens,
        input_ids=input_ids,
    )
    set_forward_context(
        attn_metadata=metadata,
        atom_config=atom_config,
        context=context,
        num_tokens=running_tokens,
        # V4.1 runs eager on the plugin path (cudagraph_mode is forced NONE):
        # its step carries host-side Engram and state work that no captured
        # graph can replay.
        in_hipgraph=False,
    )
    try:
        yield step_positions
    finally:
        reset_forward_context()
