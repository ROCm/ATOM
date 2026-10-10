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


def v41_speculative_tokens(vllm_config) -> int:
    """Draft tokens a DSpark verify step carries, or 0 when not speculating.

    Read off vLLM's own speculative config, the way the V4 bridge reads it:
    the draft is proposed and its acceptance decided by vLLM, and what ATOM
    owns is the CSA2 state a verify step leaves behind.
    """
    spec = getattr(vllm_config, "speculative_config", None)
    if spec is None:
        return 0
    n = getattr(spec, "num_speculative_tokens", None)
    return int(n) if n else 0


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
        # The draft width a verify step retains. It moves the geometry -- an
        # extra nextn layer, a window widened by it, and ring slack for the
        # writes a rejected prefix leaves behind -- so the pool the scheduler
        # sizes and the cache the runtime builds have to derive it from the
        # same place, which is this function.
        speculative_tokens=v41_speculative_tokens(vllm_config),
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
    num_slots = v41_num_state_slots(vllm_config)
    state_bytes = int(geometry.state_bytes) * num_slots
    page_size_bytes = v41_proxy_page_size_bytes(vllm_config)
    return -(-state_bytes // page_size_bytes)


def v41_scheduler_state_slots(vllm_config) -> int:
    """STATE slots the allocator may hand to requests: one per scheduled seq."""
    return max(1, int(vllm_config.scheduler_config.max_num_seqs))


def v41_num_state_slots(vllm_config) -> int:
    """Slots the STATE region holds: the scheduler's, and capture's beside them.

    Capture has to run on the serving cache. The cache object is one of the
    captured arguments of the attention break, so a forward that ran on the
    private scratch cache leaves that cache in the graph for the life of the
    entry -- measured as `step.width` frozen at the warmup's while the buffers
    the kernels write are the serving width.

    But the rows vLLM stages for warmup and capture are not requests. They
    never finish, so anything that lends them slots out of the scheduler's
    share is lending them permanently, and a pool sized at exactly
    `max_num_seqs` then has no room for a full batch: `_acquire` only protects
    the keys active in *this* step, so it evicted a request that was live but
    not scheduled that step, and the decode that followed found a reset cursor
    where its own state should be ("needs state at 912, found 1").

    Giving capture its own half removes the competition rather than refereeing
    it. The widest synthetic batch is one row per scheduled sequence, so the
    half is the same size; at 64 slots of 5.03 MiB that is 322 MiB against a
    184 GiB pool. Every site that sizes or addresses the region reads this, so
    the reserve cannot be present in the allocation and absent from the
    addressing.
    """
    return 2 * v41_scheduler_state_slots(vllm_config)


def v41_capture_state_slots(num_reqs: int, vllm_config) -> np.ndarray:
    """Slots for a synthetic batch's rows: the capture half, taken in order.

    Distinct by construction, and disjoint from anything the allocator can
    hand out, so a synthetic batch can neither collide with itself (which
    `begin_step` refuses) nor with a request in flight.
    """
    base = v41_scheduler_state_slots(vllm_config)
    if num_reqs > base:
        raise ValueError(
            f"DeepSeek-V4.1 synthetic batch has {num_reqs} rows, more than the "
            f"{base} STATE slots reserved for capture"
        )
    return np.arange(base, base + num_reqs, dtype=np.int32)


def _v41_staged_running_tokens() -> int:
    """The padded width this forward will run, as of staging time.

    Not from the forward context: vLLM sets that around the model call, and
    this runs before it, so `batch_descriptor` there is the previous step's or
    absent. Staging a 12-row step for a graph captured at 256 produced

        shape '[12, -1, 512]' is invalid for input of size 131072

    on the first replay -- the live step narrower than the buffers the
    captured kernels write.

    `InputBatch.num_tokens_after_padding` is what vLLM itself sizes the
    forward from, and the pass-through patch puts that batch within reach
    here, which is the same source rather than a second guess at it.
    """
    try:
        from atom.plugin.vllm.req_id_passthrough_patch import get_current_input_batch

        batch = get_current_input_batch()
    except Exception:  # noqa: BLE001
        return 0
    padded = getattr(batch, "num_tokens_after_padding", None)
    if padded is not None:
        return int(padded)
    return int(getattr(batch, "num_tokens", 0) or 0)


def _v41_capture_batch(snapshot, vllm_config):
    """The synthetic batch vLLM is about to capture, declared for what it is.

    Two things it is not, and both have to be said or the step is refused:

    It has no history. vLLM's dummy metadata carries a nonzero computed-token
    count, so the rows arrive looking like decodes mid-sequence, and
    `prepare_state` then checks their cursors against a position no slot has
    ever held -- "Request 1 needs state at 2, found 0", during capture, before
    the server is up. Declared fresh, every row takes the reset path that the
    judge exempts by position, which is also what is true of them.

    Its rows are not requests. They take slots from the half the allocator
    never hands out, so a capture cannot evict a request in flight nor collide
    with one -- and they stay on the serving cache, because the scratch cache's
    addresses would be recorded into the graph for the life of the entry.
    """
    query_lens = np.asarray(snapshot.query_lens, dtype=np.int32)
    num_reqs = int(snapshot.num_reqs)
    return SimpleNamespace(
        is_dummy_run=False,
        req_ids=snapshot.req_ids,
        num_scheduled_tokens=query_lens,
        context_lens=query_lens.astype(np.int64),
        state_slots_committed=v41_capture_state_slots(num_reqs, vllm_config),
        block_tables=snapshot.block_rows,
        total_seqs_num=num_reqs,
        total_tokens_num=int(query_lens.sum()),
    )


def _v41_stage_outside_forward(builder, model, vllm_config, snapshot, *, capturing):
    """One CSA2 step, staged where no graph can capture it.

    `capturing` is vLLM telling us the batch is synthetic -- it called
    `build_for_cudagraph_capture` rather than `build`. That is a better signal
    than anything derivable from the batch: four earlier attempts to infer it
    (duplicate request ids, a startup-only window, a capture-active flag, a
    reserved slot range) were each wrong in a way that only showed up under
    load. A synthetic batch takes `arange` slots so it cannot collide with a
    request in flight, and it is still staged against the serving cache --
    routing it to the private scratch cache would bake that cache's addresses
    into the graph for the life of the entry.

    Returns what the forward needs and nothing it can recompute, or None when
    there is no batch to stage.
    """
    if snapshot is None or snapshot.num_reqs == 0:
        return None
    slot_allocator = getattr(model, "_atom_v41_slot_allocator", None)
    from atom.plugin.vllm.dummy_run import in_dummy_run as _in_dummy_run

    if slot_allocator is None and not (
        capturing or _in_dummy_run() or _v41_request_ids_are_synthetic(snapshot)
    ):
        return None
    num_reqs = int(snapshot.num_reqs)
    # `capturing` is vLLM calling `build_for_cudagraph_capture`, which it only
    # does for a FULL graph. A PIECEWISE capture comes through `build` like any
    # other step, so the batch has to be recognised rather than announced --
    # by the repeated placeholder request id, which real traffic cannot
    # produce because vLLM never schedules one request twice in a step.
    from atom.plugin.vllm.dummy_run import in_dummy_run

    # Three signals, union, because no one of them covers every synthetic
    # batch and each was measured to miss a different kind:
    #
    #   capturing          vLLM calling `build_for_cudagraph_capture`, exact,
    #                      but only for a FULL graph.
    #   in_dummy_run()     the window `_dummy_run` spans -- which the warmup
    #                      forward before a capture does not go through, and
    #                      that one arrived with `synthetic=False`, 64 rows
    #                      and one slot between them.
    #   repeated req_id    what that warmup forward does have. Real traffic
    #                      cannot produce it: vLLM never schedules a request
    #                      twice in one step.
    synthetic = capturing or in_dummy_run() or _v41_request_ids_are_synthetic(snapshot)
    if synthetic:
        batch = _v41_capture_batch(snapshot, vllm_config)
    else:
        batch = _v41_scheduled_batch(snapshot, slot_allocator, None)
    running_bs = num_reqs
    running_tokens = max(_v41_staged_running_tokens(), int(batch.total_tokens_num))
    # Fetched before `_prepare`, not after. `_prepare` begins the step: it
    # resets the slots this batch recycles and leaves the step on the cache.
    # Bailing out after that and letting the forward stage the step again cost
    # one cursor advance every time it happened, which is why a decode batch
    # drifted further behind its own position the longer a run went -- four
    # steps behind at request 146, ten by request 135 of the next run.
    input_ids = _v41_step_input_ids(running_tokens)
    if input_ids is None:
        return None
    try:
        metadata, step_positions = builder._prepare(
            batch,
            running_bs,
            running_tokens,
            **_v41_prepare_kwargs(builder, batch, synthetic),
        )
    except ValueError as exc:
        if "STATE slot" not in str(exc):
            raise
        # Not `... or ()`: these are numpy arrays on the scheduled path and an
        # array's truth value raises -- which is how this diagnostic died on
        # the one failure it exists to explain, for the second time today.
        committed = getattr(batch, "state_slots_committed", None)
        slots = [] if committed is None else [int(x) for x in committed]
        tables = getattr(batch, "block_tables", None)
        raise ValueError(
            f"{exc} -- synthetic={synthetic} capturing={capturing} "
            f"num_reqs={num_reqs} slots={slots[:8]} distinct={len(set(slots))} "
            f"cache_slots={getattr(builder.cache, 'num_slots', None)} "
            f"spans={0 if tables is None else len(tables)}"
        ) from exc
    try:
        builder.prepare_model_inputs(input_ids, metadata)
    except ValueError as exc:
        if "needs state at" not in str(exc):
            raise
        _dump_v41_state_rows(snapshot, batch, builder, exc)
        raise
    # The Engram staging too, not just `prepare_model_inputs`. The cursor
    # advance rides on it (`EngramStaging.start` -> `_advance_cursor`), and the
    # forward's own call is skipped for a step staged here -- so leaving it
    # behind meant the cursor was never written, and the second decode of a
    # request found position 0 where it had reached 2.
    rows = getattr(metadata, "engram_embeddings", None)
    stage = getattr(rows, "stage", None)
    if stage is not None:
        stage()
        # Closed here, not left for the forward to close. The staging may fork
        # a prefetch stream, and this runs before the forward -- so the event
        # it records is outside whatever capture the forward is under, and a
        # captured layer waiting on it is refused:
        #
        #     hipErrorStreamCaptureIsolation: dependency created on uncaptured
        #     work in another stream
        #
        # Joining inside this window costs the overlap with the layers that
        # consume the rows, which this step never had: the whole of it happens
        # before the first layer runs.
        join = getattr(rows, "join", None)
        if join is not None:
            join()
    metadata.staged_outside_forward = True
    if getattr(metadata.step, "tentative", False):
        # Held until the sampler says how much of the draft block survived.
        # Set after staging, not at `_prepare`: a step that raised on the way
        # here never reached the cache, and leaving a stale handle behind
        # would commit the wrong step's prefix.
        global _V41_PENDING_SPECULATIVE_STEP
        _V41_PENDING_SPECULATIVE_STEP = (builder, metadata)
    return SimpleNamespace(
        metadata=metadata,
        positions=step_positions,
        running_bs=running_bs,
        running_tokens=running_tokens,
        capturing=synthetic,
    )


def _v41_step_input_ids(running_tokens):
    """This step's token ids, from the batch vLLM is about to run.

    Engram hashes them, so they have to be the step's own. The builder runs
    inside `ModelState.prepare_attn`, which holds the `InputBatch` -- the
    pass-through patch exposes it, and without that patch there is nothing
    here to hash and the caller falls back.
    """
    try:
        from atom.plugin.vllm.req_id_passthrough_patch import get_current_input_batch

        batch = get_current_input_batch()
    except Exception:  # noqa: BLE001
        return None
    ids = getattr(batch, "input_ids", None)
    return None if ids is None else ids[:running_tokens]


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

    ``_cudagraph_support`` is ``UNIFORM_SINGLE_TOKEN_DECODE``, and that is a
    claim about where the host-side state is staged, not about the kernels.
    Engram hashing, the cursor advance and the per-slot resets do not survive
    a replay, so they run here -- once per step, before the forward, outside
    anything a graph captures. What the graph then records reads the
    persistent `forward_vars` buffers this refreshes in place.

    Single-token decode and not wider: a mixed batch still goes through
    PIECEWISE, and speculation would need the verify step's ragged widths
    checked before claiming `UNIFORM_BATCH`.
    """

    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

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
        return self._build_and_attach(common_attn_metadata, capturing=False)

    def build_for_cudagraph_capture(self, common_attn_metadata):
        # vLLM calls this instead of `build` for the synthetic batch it is
        # about to capture a FULL graph from -- which is how the capture says
        # so, rather than this guessing from the batch's shape. Same staging,
        # into the same persistent buffers; only the STATE slots differ.
        return self._build_and_attach(common_attn_metadata, capturing=True)

    def _build_and_attach(self, common_attn_metadata, *, capturing):
        """Stage this step outside the captured region, and hand it over.

        vLLM calls a builder once per step, before `set_forward_context` and
        before the (possibly graph-wrapped) model forward. Staging here is what
        makes a captured graph correct: the cursor advance, the per-slot reset
        and the Engram rows all happen on host-decided values, and a captured
        forward replays without re-running any of it. The kernels that do get
        captured read the persistent `forward_vars` buffers this refreshes in
        place, so their addresses are stable and their contents are this step's.

        Returns the same `CommonAttentionMetadata`, now carrying
        `atom_v41_prepared`, which the forward consumes instead of staging
        again. Left bare when the proxy pool is not bound yet (profiling, the
        first warmup forward): the forward sees nothing prepared and falls back
        to its private scratch cache, exactly as before.
        """
        if common_attn_metadata is None:
            return common_attn_metadata
        common_attn_metadata.atom_v41_snapshot = snapshot_v41_batch(
            common_attn_metadata
        )
        sfc = self.vllm_config.compilation_config.static_forward_context
        proxy = sfc.get(ATOM_DEEPSEEK_V41_PROXY_LAYER_NAME)
        model = getattr(proxy, "_atom_v41_model", None)
        builder = getattr(proxy, "_atom_v41_builder", None)
        if model is None or builder is None:
            return common_attn_metadata
        if not bind_deepseek_v41_proxy_cache(model, builder, self.vllm_config):
            return common_attn_metadata
        prepared = _v41_stage_outside_forward(
            builder,
            model,
            self.vllm_config,
            common_attn_metadata.atom_v41_snapshot,
            capturing=capturing,
        )
        if prepared is not None:
            common_attn_metadata.atom_v41_prepared = prepared
        return common_attn_metadata


class AtomDeepseekV41ProxyBackend(AttentionBackend):
    forward_includes_kv_cache_update = True

    @staticmethod
    def get_name() -> str:
        return "ATOM_DEEPSEEK_V41_PROXY"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        # `kv_cache_spec` is passed positionally by vLLM 0.31
        # (`attention.py` and `composite.py` both call it with one argument)
        # and not at all before that, so it is accepted and defaulted rather
        # than required. V4.1 answers the same way either way: its PAGE is
        # 256 tokens whatever spec is asking, which is the whole reason the
        # proxy layer exists.
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
        # Marked for `_mark_v4_proxy_cache_mode`, which flips the flag below
        # around the memory profile. Without the attribute the layer is never
        # marked and the flag never leaves its default, so the bind cannot
        # tell a profiling pool from the serving one -- see the guard in
        # `bind_deepseek_v41_proxy_cache`.
        self._atom_v4_proxy_layer = True
        self._atom_v4_profiling_kv_cache = False

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


def _v41_scheduled_request_count(common_attn_metadata) -> int:
    """How many rows of this batch are requests, padding excluded.

    vLLM sizes the metadata it hands a builder differently per graph mode:
    `DefaultModelState.prepare_attn` uses `num_reqs_after_padding` for a FULL
    graph and `num_reqs` for everything else. Taking the number off the
    metadata therefore means requests under PIECEWISE and requests-plus-
    padding under FULL.

    The padding rows carry no scheduled tokens, so `_prepare` drops them when
    it builds the step's spans -- and `advance_cursor` then writes
    `scheduled_bs` cursors into the *first* `scheduled_bs` slots, which are no
    longer the slots those spans belong to once a padding row sits among them.
    A live request's cursor was left at the step before its own, and the
    refusal arrived hundreds of tokens later: 1159 wanted, 1158 found, under
    FULL only, while PIECEWISE ran the whole dataset clean.

    The input batch knows the unpadded count. Without the pass-through patch
    there is no padding to subtract either, because that mode is the one vLLM
    pads for, so the metadata's own number stands.
    """
    padded = int(common_attn_metadata.num_reqs)
    try:
        from atom.plugin.vllm.req_id_passthrough_patch import get_current_input_batch

        batch = get_current_input_batch()
    except Exception:  # noqa: BLE001
        return padded
    scheduled = getattr(batch, "num_reqs", None)
    return padded if scheduled is None else min(padded, int(scheduled))


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
    num_reqs = _v41_scheduled_request_count(common_attn_metadata)
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

    # One source for the arithmetic, always. `query_lens` comes from
    # `common_attn_metadata`, so `num_computed` is taken from there too rather
    # than from the persistent `input_batch`: the two are ordered
    # independently, and a KV connector parks requests
    # (WAITING_FOR_REMOTE_KVS) and re-admits them, which condenses the
    # persistent batch and can leave the two orderings a row apart. Pairing
    # one request's `num_computed` with another's `query_len` yields a span
    # whose position is some other request's, and the V4.1 state cursor --
    # which is checked exactly -- then refuses the forward.
    seq_lens_cpu = getattr(common_attn_metadata, "seq_lens_cpu", None)
    if seq_lens_cpu is None:
        seq_lens_cpu = common_attn_metadata.seq_lens[:num_reqs].cpu()
    seq_lens_np = np.asarray(seq_lens_cpu[:num_reqs], dtype=np.int64)
    num_computed = seq_lens_np - query_lens

    block_table_np = None
    req_ids = None
    if input_batch is not None:
        try:
            req_ids = list(input_batch.req_ids)[:num_reqs]
            block_table_np = input_batch.block_table[0].block_table.np
        except Exception:  # noqa: BLE001
            # The accessor is an ATOM patch over another engine's internals;
            # absent is the caller's ordinary "fall back" case.
            req_ids = None
            block_table_np = None
        else:
            # Outside the fallback, deliberately. A row divergence is not a
            # missing accessor: falling back on it would key state slots on
            # first-block ids -- `-1` for every block-less row, so two such
            # requests share one slot -- which is the wrong output this check
            # exists to name. It has to reach the caller.
            _check_row_alignment(input_batch, num_computed, num_reqs)
    if block_table_np is None:
        block_table_np = common_attn_metadata.block_table_tensor.cpu().numpy()

    # `context_lens` in ATOM's batch protocol is the request's end position
    # after this step -- vLLM's `seq_lens`.
    ends = (num_computed + query_lens).astype(np.int64)
    block_rows = []
    for i in range(num_reqs):
        needed = -(-int(ends[i]) // ATOM_DEEPSEEK_V41_BLOCK_SIZE)
        row = block_table_np[i, :needed]
        block_rows.append(tuple(int(block) for block in row))
    ids_from_patch = req_ids is not None
    if req_ids is None:
        # No pass-through patch: key on the request's first block, which vLLM
        # keeps for the request's lifetime.
        req_ids = [row[0] if row else -1 for row in block_rows]
    return SimpleNamespace(
        num_reqs=num_reqs,
        req_ids=req_ids,
        ids_from_patch=ids_from_patch,
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
    if getattr(proxy, "_atom_v4_profiling_kv_cache", False):
        # The memory profile runs with a placeholder pool -- 64 blocks, where
        # the serving one has 246218 -- and its cudagraph capture runs against
        # that same pool. Binding there carves the STATE tail out of a pool
        # 3800x too small and reports it as the pool being too small, which is
        # true of the placeholder and says nothing about the one that serves.
        #
        # This flag was once read while nothing set it: the patch that flips it
        # wrapped `vllm.v1.worker.gpu_model_runner.GPUModelRunner` while the
        # worker had instantiated the unrelated V2 runner, so it stayed False
        # through profiling. The repair for that belongs where the flag is set.
        # A guard here keyed on "is vLLM capturing" instead would be keyed on
        # the wrong question twice over: capture against the *serving* pool is
        # exactly when the bind must happen, or the captured graphs replay a
        # scratch cache.
        return False
    ptr = proxy.kv_cache.untyped_storage().data_ptr()
    if getattr(model, "_atom_v41_proxy_cache_ptr", None) == ptr:
        return True

    from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache

    # vLLM hands this back in one of two ranks, and the difference is which
    # side applied `get_kv_cache_stride_order`. Through 0.28 the tensor keeps
    # the declared `(2, num_blocks, ...)` and the permute below is what makes
    # one block's bytes contiguous. 0.31 returns it already block-major and
    # without the leading K/V pair: `(num_blocks, num_kv_heads, block_size,
    # head_size)`, stride[0] == block_size * head_size. Rank is the only thing
    # that distinguishes them, so it is what this reads -- and the byte check
    # further down is left to judge the size, rather than scaling anything
    # here on the assumption that the missing 2 means what it looks like.
    if proxy.kv_cache.dim() == 5:
        physical = proxy.kv_cache.permute(1, 0, 2, 3, 4)
    elif proxy.kv_cache.dim() == 4:
        physical = proxy.kv_cache
    else:
        raise RuntimeError(
            "DeepSeek-V4.1 proxy KV cache has an unexpected rank "
            f"{proxy.kv_cache.dim()} (shape {tuple(proxy.kv_cache.shape)}); "
            "this bridge knows the 5-dim pre-0.31 layout and the 4-dim "
            "block-major one 0.31 returns"
        )
    if not physical.is_contiguous():
        raise ValueError("DeepSeek-V4.1 proxy cache must be block-major contiguous")
    # The arena is the whole allocation, not the view over it. Through 0.28
    # those are the same tensor. 0.31 builds the view from the block count the
    # KV-cache manager will hand out (`make_kv_cache_view`), which is the count
    # AFTER `deepseek_v41_state_reserve_patch` withheld the STATE tail -- so
    # the tail is present in the allocation and absent from the view, and
    # sizing from `view.numel()` reports a pool short by exactly the reserve
    # while the bytes are sitting right there. Read the storage instead, which
    # is the same number on both versions.
    storage = proxy.kv_cache.untyped_storage()
    raw = torch.empty(0, dtype=torch.uint8, device=proxy.kv_cache.device)
    raw.set_(storage, 0, (storage.nbytes(),))
    if raw.storage_offset() % ATOM_DEEPSEEK_V41_PROXY_ALIGNMENT:
        raise RuntimeError(
            f"DeepSeek-V4.1 proxy KV storage offset {raw.storage_offset()} is not "
            f"{ATOM_DEEPSEEK_V41_PROXY_ALIGNMENT}B-aligned; EntryMajorArena cannot "
            "retype carved planes safely"
        )

    geometry = v41_proxy_geometry(vllm_config)
    # The pool was sized from `geometry` by the reserve patch and is addressed
    # by the builder's own, derived from ATOM's resolved config. They are built
    # from the same text config, so they agree unless a setting the proxy
    # geometry does not read (the index plane's format, say) moved one of them.
    # Compare them rather than trust it: a divergence is a byte layout the
    # kernels and the allocation disagree about, which no later check catches.
    if builder.geometry != geometry:
        raise RuntimeError(
            "DeepSeek-V4.1 proxy pool geometry disagrees with the runtime's: "
            f"{geometry} vs {builder.geometry}"
        )
    num_slots = v41_num_state_slots(vllm_config)
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
            # Facts only. An earlier version of this message named the
            # reserve patch as the cause; the patch had run, the shortfall
            # came from elsewhere, and a reader (me) spent a cycle on the
            # wrong suspect. The reasoning belongs here, not in what an
            # operator reads.
            f"{raw.numel()} bytes for {pages} PAGEs + {num_slots} STATE slots; "
            f"{required} needed, short by {required - raw.numel()}. "
            f"STATE tail: {v41_proxy_state_reserve_blocks(vllm_config)} blocks."
        )

    builder.num_blocks = pages
    builder.cache = PagedAttentionCache(
        geometry,
        pages,
        num_slots,
        builder.device,
        max_tokens=int(vllm_config.scheduler_config.max_num_batched_tokens),
        # The builder's own scorer scratch, as `allocate_per_req_cache` passes
        # on the native path: it is sized before the memory profile and shared
        # with the throwaway profiling cache.
        workspace=builder.score_workspace,
        backing=raw,
    )
    if not hasattr(model, "_atom_v41_slot_allocator"):
        model._atom_v41_slot_allocator = StateSlotAllocator(
            v41_scheduler_state_slots(vllm_config)
        )
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


def _dump_v41_state_rows(snapshot, batch, builder, exc) -> None:
    """Log every column a stale-cursor refusal is a disagreement between.

    Deliberately blind to its own failures: this runs on a path that is
    already raising, and a diagnostic that masks the error it is explaining is
    worse than no diagnostic.
    """
    try:
        # Not `... or []`: `state_slots_committed` is a numpy array on the
        # scheduled path, and an array's truth value raises. This diagnostic
        # died there, on the one failure it exists to explain, and reported
        # itself only as "could not dump state rows".
        committed = getattr(batch, "state_slots_committed", None)
        slots = [] if committed is None else list(committed)
        cache = getattr(builder, "cache", None)
        cursors = None
        if cache is not None:
            cursors = cache.cursor[:, 0].tolist()
        logger.error("DeepSeek-V4.1 stale-cursor refusal: %s", exc)
        logger.error(
            "  %-4s %-44s %10s %9s %6s %10s",
            "row",
            "req_id",
            "computed",
            "query",
            "slot",
            "cursor",
        )
        for i, req_id in enumerate(snapshot.req_ids):
            slot = int(slots[i]) if i < len(slots) else -1
            cursor = (
                int(cursors[slot])
                if cursors is not None and 0 <= slot < len(cursors)
                else -1
            )
            logger.error(
                "  %-4d %-44s %10d %9d %6d %10d",
                i,
                str(req_id)[-44:],
                int(snapshot.num_computed[i]),
                int(snapshot.query_lens[i]),
                slot,
                cursor,
            )
        # Two requests on one slot is a different bug from a cursor that ran
        # ahead, and the two are indistinguishable from the refusal alone.
        used = [int(s) for s in slots[: len(snapshot.req_ids)]]
        if len(set(used)) != len(used):
            logger.error("  SLOT COLLISION: %s", used)
    except Exception:
        logger.exception("DeepSeek-V4.1: could not dump state rows")


def _check_row_alignment(input_batch, num_computed, num_reqs: int) -> None:
    """Refuse a batch whose two orderings disagree, naming the row.

    `req_ids` and the block table are read from the persistent `input_batch`
    while the spans are built from `common_attn_metadata`. They address the
    same requests only while their row orders agree. If they ever do not, the
    state slot is keyed by one request and the position by another -- which is
    wrong output, not a crash, for every model that does not check its cursor.
    So it is checked here, where the row can still be named, rather than left
    to surface as a cursor that is off by the gap between two requests.
    """
    batch_computed = getattr(input_batch, "num_computed_tokens_cpu", None)
    if batch_computed is None:
        return
    mine = np.asarray(batch_computed[:num_reqs], dtype=np.int64)
    if mine.shape != num_computed.shape:
        raise ValueError(
            "DeepSeek-V4.1: input_batch has "
            f"{mine.shape[0]} rows but the step describes {num_computed.shape[0]}"
        )
    bad = np.flatnonzero(mine != num_computed)
    if bad.size:
        row = int(bad[0])
        raise ValueError(
            "DeepSeek-V4.1: the persistent batch and this step's attention "
            f"metadata disagree at row {row}: input_batch says "
            f"{int(mine[row])} computed tokens, the step says "
            f"{int(num_computed[row])}. Their row orders have diverged, so "
            "req_ids and block tables cannot be trusted for this step."
        )


def _v41_scheduled_batch(snapshot, slot_allocator, slots=None):
    """ATOM's scheduled-batch protocol, from the host snapshot.

    ``state_slots_committed`` covers every row, including the zero-token rows
    ``_prepare`` skips, because it is indexed by the same ``i`` as the zipped
    per-request arrays.
    """
    if slots is None:
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


try:
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture
except ImportError:  # vLLM too old for breakable capture; the break is a no-op

    def eager_break_during_capture(fn):
        return fn


def _v41_capture_active() -> bool:
    """Whether a breakable cudagraph capture is recording this forward."""
    try:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture
    except ImportError:
        return False
    return bool(BreakableCUDAGraphCapture.is_active())


def _v41_live_running_tokens(input_ids) -> int:
    """The padded width this forward runs, as of this step.

    `input_ids` cannot answer it on the replay path. It is one of the break's
    captured arguments, so it is the slice vLLM passed when the graph was
    recorded, and its length is that forward's width. vLLM pads a batch up to
    the descriptor it dispatched, so a two-request decode replayed on the
    width-32 graph must still stage 32 rows; staging 2 left the step narrower
    than the buffers the captured kernels write, and `rope_quant_window` said
    so as `shape '[2, 512]' is invalid for input of size 16384`.

    The descriptor on vLLM's forward context carries that width and is set
    before each call, outside the captured region, so it is live on replay.
    `input_ids` remains the fallback for callers that run with no vLLM forward
    context at all, where it is the only width there is.
    """
    fallback = int(input_ids.shape[0]) if input_ids is not None else 0
    try:
        from vllm.forward_context import (
            get_forward_context,
            is_forward_context_available,
        )

        if not is_forward_context_available():
            return fallback
        descriptor = get_forward_context().batch_descriptor
    except (ImportError, AssertionError, AttributeError):
        return fallback
    num_tokens = getattr(descriptor, "num_tokens", None)
    result = fallback if num_tokens is None else max(fallback, int(num_tokens))

    return result


def _v41_request_ids_are_synthetic(snapshot) -> bool:
    """Whether this snapshot describes vLLM's own batch rather than traffic.

    vLLM's warmup and capture batches repeat one placeholder request id across
    every row. Keyed on that id, the slot allocator hands every row the same
    STATE slot and `begin_step` refuses the batch, since STATE is per in-flight
    request and two rows cannot share one.

    A repeat is the whole test: vLLM never schedules one request twice in a
    step, so real traffic cannot produce one. These rows get their own keys
    (`_v41_warmup_keys`) and stay on the serving pool, which is the only one
    the captured graph may record.

    The one other way rows can collide is `snapshot_v41_batch`'s fallback key,
    the request's first block id, which is -1 for every block-less row. That
    fallback is only reached when the req_id pass-through patch is absent, and
    the row-divergence check next to it already governs that case.
    """
    # Both keying modes make a repeat impossible for real traffic, which is
    # why this does not ask which one produced the ids. With the pass-through
    # patch they are vLLM's own request ids, unique in a step by construction.
    # Without it `snapshot_v41_batch` keys on the request's first block, and
    # two live requests cannot share one with prefix caching off. What does
    # repeat is a block-less row -- and a scheduled request always owns at
    # least one block, so those rows are not requests.
    ids = list(getattr(snapshot, "req_ids", None) or ())
    return bool(ids) and len(set(ids)) != len(ids)


def _v41_live_step_inputs(
    builder, slot_allocator, input_ids, proxy_layer_name, vllm_config
):
    """This step's batch, read from state the replay path keeps current.

    Read here rather than passed in. `eager_break_during_capture` binds a
    break's arguments at capture time (weakly, so the cudagraph pool can
    reclaim them), and `_replay` never calls the model again -- it walks the
    recorded segments. A break handed a per-step Python object therefore sees
    the object from the forward that was recorded, on every later step: the
    batch this was staged for would be the warmup batch for the life of the
    graph.

    vLLM rebuilds its forward context and the attention metadata on it before
    each call, outside the captured region, so reading the snapshot through
    that context is live on replay. The builder and the slot allocator are
    stable objects whose contents move underneath them, which is what a
    captured segment needs.
    """
    common_attn_metadata = get_deepseek_v41_proxy_metadata_from_vllm_context(
        proxy_layer_name
    )
    snapshot = getattr(common_attn_metadata, "atom_v41_snapshot", None)
    running_tokens = _v41_live_running_tokens(input_ids)
    if (
        snapshot is None
        or slot_allocator is None
        or builder.cache is None
        or snapshot.num_reqs == 0
    ):

        running_tokens = max(running_tokens, 1)
        batch = _v41_dummy_batch(
            running_tokens,
            max_req_tokens=builder.block_table_cols
            * builder.block_ratio
            * builder.block_size,
            max_reqs=builder.max_bs,
        )
        return snapshot, batch, batch.total_seqs_num, running_tokens, True
    slots = (
        v41_capture_state_slots(int(snapshot.num_reqs), vllm_config)
        if _v41_request_ids_are_synthetic(snapshot)
        else None
    )
    batch = _v41_scheduled_batch(snapshot, slot_allocator, slots)
    return (
        snapshot,
        batch,
        int(snapshot.num_reqs),
        max(running_tokens, batch.total_tokens_num),
        False,
    )


def _v41_publish_metadata(metadata) -> None:
    """Put this step's metadata where the other breaks read it.

    The forward boundaries (`v41_begin_forward` / `v41_end_forward`) take the
    step off ATOM's forward context rather than as an argument, which is what
    lets them be breaks at all -- a break's arguments are bound at capture.
    But `set_forward_context` runs in the contextmanager around the model, and
    that is ordinary Python: `_replay` does not call the model, so on a replay
    the context still describes the forward the graph was recorded from.

    Updating the live context object from inside this break closes that gap.
    At capture there is no context yet and this does nothing; the
    contextmanager publishes the same object a moment later.
    """
    from atom.utils.forward_context import get_forward_context

    try:
        context = get_forward_context()
    except AssertionError:
        return
    context.attn_metadata = metadata


@eager_break_during_capture
def v41_stage_step(
    builder, slot_allocator, input_ids, proxy_layer_name, force_dummy, vllm_config
):
    """One CSA2 step's host-side work, as a single break point.

    Both halves have to be in the *same* eager break, and the break has to be
    here rather than on ``prepare_model_inputs`` alone. ``_prepare`` stages the
    step's index/indptr/slot tensors with kernels of its own; captured, they
    would replay the batch that happened to be scheduled when the graph was
    recorded, and the state work that follows would then be reading a stale
    description of the batch. ``prepare_model_inputs`` carries its own break
    decorator for the native engine; nested inside this one it is inert
    (``add_eager`` leaves ``_capturing`` False while it runs the callable), so
    this does not double-break.

    The decorator's in-place contract is met by construction rather than by
    care: ``_prepare`` returns a slice of the persistent ``positions.gpu``
    buffer, and everything else it produces is host-side and consumed only at
    capture time. A replay runs this function and no other Python here, so the
    captured segments keep reading the addresses they recorded while the
    contents are refreshed underneath them.
    """
    snapshot, batch, running_bs, running_tokens, synthetic = _v41_live_step_inputs(
        builder,
        None if force_dummy else slot_allocator,
        input_ids,
        proxy_layer_name,
        vllm_config,
    )
    metadata, step_positions = builder._prepare(batch, running_bs, running_tokens)
    # Engram embeddings, the per-request state reset and the cursor advance --
    # everything that must happen once per step, before any layer runs.
    try:
        builder.prepare_model_inputs(input_ids, metadata)
    except ValueError as exc:
        if "needs state at" not in str(exc):
            raise
        # One shot, at the only place where every column exists together:
        # ATOM's refusal names a request and two positions but not the batch
        # that produced them, and the connector is not in the picture at all.
        # Printing the whole table turns "off by one" into a row to look at.
        _dump_v41_state_rows(snapshot, batch, builder, exc)
        raise
    _v41_publish_metadata(metadata)
    return metadata, step_positions, running_bs, running_tokens, synthetic


# The step whose Engram cursor is still waiting on the sampler, as
# `(builder, metadata)`. A tentative step does not advance its own cursor --
# only the accepted prefix does, and which prefix that is nobody knows until
# vLLM has rejected what it rejects. `v41_commit_speculative_state` closes it.
_V41_PENDING_SPECULATIVE_STEP: tuple | None = None


def _v41_prepare_kwargs(builder, batch, synthetic):
    """Tentative-step arguments for `_prepare`, empty without speculation.

    A speculative step runs the draft's whole block and keeps the Engram
    cursor off to one side; the sampler picks the accepted prefix afterwards.
    Synthetic batches (capture, dummy run) are excluded: nothing samples for
    them, so a tentative step there would be left uncommitted forever.
    """
    speculative_tokens = builder.geometry.speculative_tokens
    if synthetic or not speculative_tokens:
        return {}
    lengths = [int(n) for n in batch.num_scheduled_tokens]
    ends = [int(n) for n in batch.context_lens]
    rows = [(end - length, length) for end, length in zip(ends, lengths) if length]
    if not rows:
        return {}
    # A tentative step is a verification step: every row is a draft block on
    # top of an existing prefix. A prefill row is neither -- it starts at
    # position 0 and is longer than a block -- and the cache refuses the mix.
    # Such a step verifies nothing, so there is nothing to hold back from its
    # cursor and the ordinary advance is right.
    # A row carries a draft block only if it continues an existing prefix and
    # is wider than the single token a plain decode runs. A one-token decode
    # row verifies nothing, so its cursor can advance the ordinary way.
    def _is_draft(row):
        position, length = row
        return position > 0 and 1 < length <= speculative_tokens + 1

    def _fits_tentative(row):
        position, length = row
        return position > 0 and length <= speculative_tokens + 1

    if not any(_is_draft(row) for row in rows):
        return {}
    if not all(_fits_tentative(row) for row in rows):
        # Draft rows beside prefill rows: the draft rows would advance their
        # cursor over tokens verification may yet reject. Loud, because the
        # damage is a silent per-request drift.
        raise NotImplementedError(
            "DeepSeek-V4.1 on the vLLM plugin cannot verify a draft block in "
            "the same step as a prefill "
            f"(rows position/length: {rows[:16]}, spec={speculative_tokens})"
        )
    return {
        "tentative": True,
        "max_q_len": max(length for _position, length in rows),
    }


def v41_commit_speculative_state(anchors) -> bool:
    """Write the accepted prefix's cursor for the step the sampler just ended.

    `anchors`: each request's flat row of its last accepted token, in the
    scheduled order `_prepare` built. Returns whether a step was waiting.
    """
    global _V41_PENDING_SPECULATIVE_STEP

    pending = _V41_PENDING_SPECULATIVE_STEP
    if pending is None:
        return False
    _V41_PENDING_SPECULATIVE_STEP = None
    builder, metadata = pending
    builder.commit_speculative_state(metadata, anchors)
    return True


def v41_speculative_step_is_pending() -> bool:
    return _V41_PENDING_SPECULATIVE_STEP is not None


def _v41_prepared_for_this_step(proxy_layer_name):
    """What the builder staged for this step, or None if it did not run."""
    common = get_deepseek_v41_proxy_metadata_from_vllm_context(proxy_layer_name)
    return getattr(common, "atom_v41_prepared", None)


@contextmanager
def atom_deepseek_v41_forward_context(
    *,
    atom_config,
    builder,
    input_ids,
    positions,
    slot_allocator=None,
    vllm_config=None,
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

    # Which batch this step runs is decided inside `v41_stage_step`, not here.
    # Everything this function does outside that call runs at capture time and
    # never again: `_replay` walks the recorded segments and does not call the
    # model. Deciding here and passing the result in would pin every later
    # replay to the batch that was scheduled when the graph was recorded.
    # Staged by the metadata builder, outside anything a graph can capture.
    # The forward consumes it; it does not re-stage, because the cursor
    # advance and the per-slot reset are once-per-step and would be applied
    # twice. `prepared` is absent only before the proxy pool is bound, where
    # the fallback below runs the step on a private scratch cache.
    prepared = _v41_prepared_for_this_step(proxy_layer_name)
    if prepared is not None:
        metadata = prepared.metadata
        step_positions = prepared.positions
        running_bs = prepared.running_bs
        running_tokens = prepared.running_tokens
        dummy = False
    else:
        metadata, step_positions, running_bs, running_tokens, dummy = v41_stage_step(
            builder,
            slot_allocator,
            input_ids,
            proxy_layer_name,
            force_dummy,
            vllm_config,
        )

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
        # False on this path, and not because nothing is being captured.
        #
        # `in_hipgraph` reads as "a graph is recording" but its one consumer
        # for V4.1 is `side_stream`, whose contract is narrower: it forks to a
        # side stream *only inside ATOM's own capture loop*, the window where
        # ATOM owns the thread and the capture. Natively the two coincide, so
        # the flag can stand for both. Under the plugin they come apart --
        # vLLM runs the capture, on its own thread, and a fork opened by this
        # flag then cannot be ended where it began:
        #
        #     capture_end() -> HIP error: attempt to terminate a thread-local
        #     capture sequence from another thread
        #
        # Measured, with `ATOM_DSV41_SIDE_STREAMS` at its default 0: the
        # compressor and indexer streams are None and never forked, and the
        # one that did fork is the MoE's `alt_stream`, which takes the same
        # gate.
        #
        # So this reports the thing the consumer actually asks about. If a
        # second consumer ever wants "a graph is recording", it should ask
        # `_v41_capture_active()` directly rather than widen this.
        in_hipgraph=False,
    )
    try:
        yield step_positions
    finally:
        reset_forward_context()
