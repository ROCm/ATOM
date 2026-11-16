# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import copy
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch

from atom.config import KVCacheTensor, get_current_atom_config
from atom.model_ops.attention_gdn import GatedDeltaNet, fused_gdn_gating
from atom.model_ops.attentions.gdn_attn import (
    GDNAttentionMetadata,
    compute_causal_conv1d_metadata,
)
from atom.model_ops.fla_ops import fused_recurrent_gated_delta_rule
from atom.model_ops.fla_ops.chunk import pop_last_intermediate_states
from atom.model_ops.fla_ops.replayssm import replayssm_gated_delta_rule
from atom.model_ops.mamba_ops.causal_conv1d import causal_conv1d_update
from atom.plugin.sglang.attention_backend.backend_resolver import (
    reconstruct_linear_metadata,
    resolve_attn_backend,
    resolve_mamba_req_pool,
)
from atom.plugin.sglang.attention_backend.gdn_radix_track import (
    RadixTrackPlan,
    fused_track_metadata_supported,
    fused_track_supported,
    write_radix_track_snapshots,
)
from atom.plugin.sglang.attention_backend.gdn_replayssm import (
    note_full_state_write,
    prepare_layer,
    replayssm_enabled,
)
from atom.plugin.sglang.patches.qwen4_exp_gdn_pad import (
    apply_gdn_pad_sentinels,
)
from atom.utils import envs
from atom.utils.forward_context import (
    AttentionMetaData,
    Context,
    DPMetadata,
    _forward_kv_cache_context,
    get_forward_context,
    reset_forward_context,
    set_forward_context,
    set_kv_cache_data,
)

logger = logging.getLogger(__name__)


def _align_flydsl_decode_slots(index: torch.Tensor, batch: int) -> torch.Tensor:
    """Contiguous int32 prefix of length ``batch`` for FlyDSL decode.

    Must stay a view into Hybrid's static cuda-graph buffer. A ``.clone()`` /
    fresh ``.contiguous()`` allocation here bakes a throwaway address into the
    decode graph (see ``_linear_static_ple_slots``).
    """
    if index.shape[0] != batch:
        index = index[:batch]
    if index.dtype != torch.int32:
        raise RuntimeError(
            f"GDN decode slot index must be int32 static buffer, got {index.dtype}"
        )
    if not index.is_contiguous():
        raise RuntimeError(
            "GDN decode slot index must be a contiguous view of the Hybrid "
            "cuda-graph mamba_cache_indices buffer"
        )
    return index


def _flydsl_linear_impls() -> list[Any]:
    """Qwen4 GDN layers that opted into AITER FlyDSL (`allow_aiter_flydsl`)."""
    try:
        ctx = get_current_atom_config().compilation_config.static_forward_context
    except Exception:  # noqa: BLE001
        return []
    cached = getattr(_flydsl_linear_impls, "_cache", None)
    if cached is not None and cached[0] is ctx:
        return cached[1]
    impls: list[Any] = []
    for module in ctx.values():
        impl = getattr(module, "impl", None)
        if impl is not None and getattr(impl, "allow_aiter_flydsl", False):
            impls.append(impl)
    _flydsl_linear_impls._cache = (ctx, impls)
    return impls


def _flydsl_prefill_supported(device: torch.device) -> bool:
    """FlyDSL prefill GEMM is valid on gfx942. gfx950 hits hipBLAS internally."""
    if torch.version.hip is None or device.type != "cuda":
        return True
    arch = torch.cuda.get_device_properties(device).gcnArchName.split(":")[0]
    return arch == "gfx942"


def _ensure_flydsl_policy(state: torch.Tensor):
    """Bind Native `select_policy` once, same as `_build_gdn_cache_tensor`."""
    impls = _flydsl_linear_impls()
    if not impls:
        return None
    policy = impls[0].gdn_flydsl_policy
    if policy is not None:
        return policy
    from atom.model_ops.fla_ops.gdn_flydsl import GDNFlyDSLPolicy, select_policy

    # ReplaySSM addresses the same contiguous checkpoint Triton prefill
    # writes. FlyDSL's VK view is a different layout; Native keeps both
    # stages off once the ring is on.
    policy = select_policy(
        allowed=True,
        replayssm=replayssm_enabled(),
        lossy_decode=bool(envs.ATOM_ENABLE_GDN_DECODE_LOSSY_FAST),
        state=state,
        activation_dtype=impls[0].dt_bias.dtype,
    )
    if policy.prefill and not _flydsl_prefill_supported(state.device):
        policy = GDNFlyDSLPolicy(prefill=False, decode=policy.decode)
    for impl in impls:
        impl.gdn_flydsl_policy = policy
    return policy


# (lengths, cu_seqlens data_ptr, tensor version) -> schedule.
# AITER binds the schedule to that exact offset tensor. Reuse only when the
# same buffer is unchanged (CUDA-graph replay). A new eager batch rebuilds
# from the host length list and does not read query_start_loc back to the host.
_PREFILL_SCHEDULES: dict[tuple, object] = {}


def _host_prefill_lengths(forward_batch: Any, num_seqs: int) -> tuple[int, ...] | None:
    """Scheduler-side query lengths. Avoids a GPU ``tolist`` of ``query_start_loc``."""
    cpu = getattr(forward_batch, "extend_seq_lens_cpu", None)
    if cpu is None:
        seqs = getattr(forward_batch, "extend_seq_lens", None)
        if isinstance(seqs, (list, tuple)):
            cpu = seqs
        elif isinstance(seqs, torch.Tensor) and seqs.device.type == "cpu":
            cpu = seqs.tolist()
    if cpu is None:
        return None
    vals = [int(x) for x in cpu]
    if len(vals) < num_seqs:
        return None
    return tuple(vals[:num_seqs])


def _flydsl_prefill_metadata(
    query_start_loc: torch.Tensor, forward_batch: Any
) -> object | None:
    """One FlyDSL K1-K5 schedule, shared by every GDN layer and reused by shape.

    Native builds this from the host length list in ``prepare_prefill``, outside
    the decode graph. The same length tuple keeps the schedule tensors so a
    captured prefill does not rebuild them or sync ``query_start_loc`` to host.
    """
    impls = _flydsl_linear_impls()
    if not impls:
        return None
    policy = impls[0].gdn_flydsl_policy
    # Metadata is sometimes built before the KV bind on this step. Native gates
    # on `_flydsl_prefill_enabled`; if policy is still unset, still try — the
    # helper returns None when the backend is Triton or AITER ops are missing.
    if policy is not None and not policy.prefill:
        return None
    num_seqs = int(query_start_loc.shape[0]) - 1
    if num_seqs <= 0:
        return None
    lengths = _host_prefill_lengths(forward_batch, num_seqs)
    capturing = query_start_loc.is_cuda and torch.cuda.is_current_stream_capturing()
    if lengths is None:
        if capturing:
            return None
        lengths = tuple(
            int(n) for n in (query_start_loc[1:] - query_start_loc[:-1]).tolist()
        )
    # Identity, not just the length tuple: FlyDSL rejects a schedule whose
    # cu_seqlens object is a different tensor or was written after it was built.
    key = (lengths, int(query_start_loc.data_ptr()), int(query_start_loc._version))
    cached = _PREFILL_SCHEDULES.get(key)
    if cached is not None:
        return cached
    if capturing:
        # AITER refuses to build a schedule on the capturing stream. The eager
        # warmup of this buffer fills the cache before capture.
        return None
    from atom.model_ops.fla_ops.gdn_flydsl import build_prefill_metadata

    metadata = build_prefill_metadata(lengths, query_start_loc)
    if metadata is not None:
        _PREFILL_SCHEDULES[key] = metadata
    return metadata


def _track_mamba_radix_slots(forward_batch: Any, layer_id: int) -> None:
    """Decode-side mamba track after Native GDN compute.

    Community bakes ``_track_mamba_state_decode`` inside the decode CUDA graph.
    Native GDN + FlyDSL has repeatedly faulted with that binding under
    radix/extra_buffer, so decode-graph capture **skips** track here and
    ``track_mamba_radix_after_decode_graph`` (post-replay epilogue) performs
    the same snapshot with live Hybrid metadata. Eager decode tracks inline.
    Never host-sync on ``mask.any()``.

    Decode only: a decode step's live state IS the state at its track point.
    Extend is tracked by ``SGLangGatedDeltaNet._track_mamba_states``, which
    reads the chunk-boundary state an unaligned track point needs; a row copy
    of the finished state here would overwrite that snapshot.

    ``check_freed_slots=True`` skips ``src/dst == -1`` (graph pad / tombstones);
    the track kernel otherwise faults on ``ptr + (-1)*stride``.
    """
    if getattr(forward_batch, "mamba_track_mask", None) is None:
        return
    if not forward_batch.forward_mode.is_decode_or_idle():
        return
    # Do not bake track into the full decode graph; epilogue owns decode track.
    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        return
    attn_backend = SGLangGDNForwardContext._resolve_attn_backend(forward_batch)
    if attn_backend is None:
        return
    linear = SGLangGDNForwardContext._linear_attn_backend(attn_backend)
    if linear is None or getattr(linear, "forward_metadata", None) is None:
        return
    fm = linear.forward_metadata
    pool = getattr(linear, "req_to_token_pool", None)
    if pool is None:
        return
    try:
        layer_cache = pool.mamba2_layer_cache(layer_id)
    except Exception:  # noqa: BLE001
        return
    cache_indices = getattr(fm, "mamba_cache_indices", None)
    track_indices = getattr(fm, "mamba_track_indices", None)
    if cache_indices is None or track_indices is None:
        return

    bs = int(forward_batch.batch_size)
    # Match PR #32555 / Hybrid replay: only the refreshed [:bs] prefix is live.
    if cache_indices.shape[0] != bs:
        cache_indices = cache_indices[:bs]
    if track_indices.shape[0] != bs:
        track_indices = track_indices[:bs]
    track_mask = forward_batch.mamba_track_mask
    if track_mask is not None and track_mask.shape[0] != bs:
        track_mask = track_mask[:bs]

    # Prefer community coalesced all-layers track at the last mamba layer
    # (same as ``GDNAttnBackend.forward_decode(..., layer.layer_id)``). Always
    # pass check_freed_slots=True: Flash full-graph pads use PAD_SLOT_ID=-1 and
    # the track kernel faults on ``ptr + (-1)*stride`` when that flag is off.
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        track_mamba_states_if_needed,
    )

    track_pools = getattr(linear, "_track_pools", None)
    pools = track_pools() if callable(track_pools) else None
    if pools is not None:
        conv_pool, ssm_pool, last_pool_idx = pools
        if pool.mamba_map.get(layer_id) == last_pool_idx:
            from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
                track_mamba_states_all_layers,
            )

            track_mamba_states_all_layers(
                conv_pool,
                ssm_pool,
                cache_indices,
                track_mask,
                track_indices,
                bs,
                check_freed_slots=True,
            )
        return

    track_mamba_states_if_needed(
        layer_cache.conv[0],
        layer_cache.temporal,
        cache_indices,
        track_mask,
        track_indices,
        bs,
        check_freed_slots=True,
    )


def track_mamba_radix_after_decode_graph(
    forward_batch: Any, attn_backend: Any = None
) -> None:
    """Post-replay mamba track (Native epilogue).

    Community bakes track inside the graph; ``_track_mamba_radix_slots`` skips
    capture, so for a replayed decode graph this epilogue is the only snapshot:
    it runs the coalesced track with the *live* Hybrid metadata after replay.
    Cheap relative to the graphed decode body.

    ``attn_backend`` must be passed from ``DecodeCudaGraphRunner`` — after
    ``execute`` returns, SGLang's forward context is gone so
    ``resolve_attn_backend(forward_batch)`` is usually ``None``.
    """
    if forward_batch is None:
        return
    if getattr(forward_batch, "mamba_track_mask", None) is None:
        return
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None or not callable(getattr(mode, "is_decode_or_idle", None)):
        return
    if not mode.is_decode_or_idle():
        return
    if attn_backend is None:
        attn_backend = SGLangGDNForwardContext._resolve_attn_backend(forward_batch)
    if attn_backend is None:
        return
    # Ensure resolve paths inside ``_track_mamba_radix_slots`` see the backend.
    try:
        if getattr(forward_batch, "attn_backend", None) is None:
            forward_batch.attn_backend = attn_backend
    except Exception:  # noqa: BLE001, S110
        pass
    SGLangGDNForwardContext._patch_forward_batch_pools(forward_batch, attn_backend)
    linear = SGLangGDNForwardContext._linear_attn_backend(attn_backend)
    pool = getattr(linear, "req_to_token_pool", None) if linear is not None else None
    mamba_map = getattr(pool, "mamba_map", None) if pool is not None else None
    if not mamba_map:
        return
    track_pools = getattr(linear, "_track_pools", None)
    pools = track_pools() if callable(track_pools) else None
    if pools is not None:
        # Coalesced all-layers track fires only at the last mamba layer.
        _track_mamba_radix_slots(forward_batch, int(max(mamba_map.keys())))
        return
    for layer_id in mamba_map:
        _track_mamba_radix_slots(forward_batch, int(layer_id))


# type(pool), type(pool.mamba_pool), type(state) -> whether the pool's layer
# accessor is SGLang's own `MambaPool.State.at_layer_idx` slice.
_STATE_SLICE_TYPES: dict[tuple[type, type, type], bool] = {}


def _state_slice_accessor(pool: Any, mamba_pool: Any, state: Any) -> bool:
    key = (type(pool), type(mamba_pool), type(state))
    known = _STATE_SLICE_TYPES.get(key)
    if known is None:
        try:
            from sglang.srt.mem_cache.memory_pool import (
                HybridReqToTokenPool,
                MambaPool,
            )

            known = (
                getattr(key[0], "mamba2_layer_cache", None)
                is HybridReqToTokenPool.mamba2_layer_cache
                and getattr(key[0], "mamba2_layer_index", None)
                is HybridReqToTokenPool.mamba2_layer_index
                and getattr(key[1], "mamba2_layer_cache", None)
                is MambaPool.mamba2_layer_cache
                and getattr(key[2], "at_layer_idx", None)
                is MambaPool.State.at_layer_idx
            )
        except Exception:  # noqa: BLE001 - any other pool asks every time
            known = False
        _STATE_SLICE_TYPES[key] = known
    return known


class _PoolLayerViews:
    """`mamba2_layer_cache` results of one SGLang mamba pool, per layer.

    `MambaPool.State.at_layer_idx` builds a new dataclass of views on every
    call. Those views depend only on the pool's state tensors, which SGLang
    allocates once, so a layer's views are reused while the pool still holds
    the same state object and layer map.
    """

    __slots__ = ("kv_tensors", "layers", "mamba_map", "mamba_pool", "pool", "state")

    def __init__(self, pool: Any, mamba_pool: Any, state: Any) -> None:
        self.pool = pool
        self.mamba_pool = mamba_pool
        self.state = state
        self.mamba_map = pool.mamba_map
        self.layers: dict[int, Any] = {}
        # Last `_build_kv_cache_tensors` result: (use_kv_view, first layer's
        # temporal view, {name: KVCacheTensor}).
        self.kv_tensors: tuple[bool, Any, dict[str, KVCacheTensor]] | None = None


_POOL_LAYER_VIEWS: dict[int, _PoolLayerViews] = {}


def _pool_layer_views(pool: Any) -> _PoolLayerViews | None:
    """The view cache of `pool`, or None when its views must be rebuilt per call:
    a HiCache transfer counter (its per-layer wait lives in
    `mamba2_layer_cache`), or a layer accessor other than SGLang's slice."""
    mamba_pool = getattr(pool, "mamba_pool", None)
    state = getattr(mamba_pool, "mamba_cache", None)
    if (
        state is None
        or getattr(pool, "layer_transfer_counter", None) is not None
        or not _state_slice_accessor(pool, mamba_pool, state)
    ):
        return None
    entry = _POOL_LAYER_VIEWS.get(id(pool))
    if (
        entry is None
        or entry.pool is not pool
        or entry.mamba_pool is not mamba_pool
        or entry.state is not state
        or entry.mamba_map is not pool.mamba_map
    ):
        entry = _PoolLayerViews(pool, mamba_pool, state)
        _POOL_LAYER_VIEWS[id(pool)] = entry
    return entry


def _mamba_layer_cache(pool: Any, layer_id: int) -> Any:
    """`pool.mamba2_layer_cache(layer_id)`, reused while the pool state is the same."""
    entry = _pool_layer_views(pool)
    if entry is None:
        return pool.mamba2_layer_cache(layer_id)
    view = entry.layers.get(layer_id)
    if view is None:
        view = entry.layers[layer_id] = pool.mamba2_layer_cache(layer_id)
    return view


class _GDNStep:
    """Forward-level state of `SGLangGatedDeltaNet`, built once per bound forward.

    Each GDN layer used to re-derive it after its compute: the forward mode,
    the SGLang linear backend, its track metadata and destination slots. The
    radix-track launch plan is built from the same metadata here as well.
    `SGLangGDNForwardContext.bind` publishes it for the forward batch it was
    built from; a layer that finds no step for its batch takes the per-layer
    path (`_track_mamba_states`).
    """

    # What a layer does after its GDN compute (the `track` field).
    NONE = 0  # nothing to track
    LAYER = 1  # the per-layer `_track_mamba_states` (decode, unusual metadata)
    EXTEND = 2  # extend snapshots through `plan` (None: no tracked row)

    __slots__ = (
        "conv_idx",
        "conv_slots",
        "forward_batch",
        "linear_backend",
        "metadata",
        "plan",
        "pool",
        "track",
        "verify",
    )

    def __init__(self, forward_batch: Any, linear_backend: Any) -> None:
        self.forward_batch = forward_batch
        self.linear_backend = linear_backend
        self.metadata = self.pool = self.conv_idx = self.conv_slots = None
        self.plan: RadixTrackPlan | None = None
        mode = forward_batch.forward_mode
        self.verify = bool(mode.is_target_verify())
        self.track = self.NONE
        if self.verify or getattr(forward_batch, "mamba_track_mask", None) is None:
            return
        if mode.is_decode_or_idle():
            self.track = self.LAYER
            return
        if not mode.is_extend():
            return
        # Anything unusual stays on the per-layer path, which handles it as before.
        self.track = self.LAYER
        metadata = getattr(linear_backend, "forward_metadata", None)
        pool = getattr(linear_backend, "req_to_token_pool", None)
        has_mask = getattr(metadata, "has_mamba_track_mask", None)
        if metadata is None or pool is None or has_mask is None:
            return
        self.metadata, self.pool = metadata, pool
        if not has_mask:
            self.track = self.EXTEND
            return
        if not fused_track_metadata_supported(metadata):
            return
        conv_idx = getattr(metadata, "track_conv_indices", None)
        conv_slots = None
        if conv_idx is not None and conv_idx.numel() > 0:
            # SGLang's GDN backend caches the destination slots; without them
            # each layer derives them from the track mask as before.
            conv_slots = getattr(metadata, "conv_states_mask_indices", None)
            if conv_slots is None:
                return
        else:
            conv_idx = None
        self.conv_idx, self.conv_slots = conv_idx, conv_slots
        self.plan = RadixTrackPlan(
            conv_idx,
            conv_slots,
            metadata.track_ssm_h_src,
            metadata.track_ssm_h_dst,
            metadata.track_ssm_final_src,
            metadata.track_ssm_final_dst,
        )
        self.track = self.EXTEND


_GDN_STEP: ContextVar[_GDNStep | None] = ContextVar(
    "atom_sglang_gdn_step", default=None
)


class SGLangGatedDeltaNet(GatedDeltaNet):
    """Run batched ATOM GDN while filling SGLang's verify snapshots."""

    def forward(
        self,
        mixed_qkv: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        core_attn_out: torch.Tensor,
        layer_name: str,
    ) -> torch.Tensor:
        from atom.plugin.sglang.runtime import get_current_forward_batch

        forward_batch = get_current_forward_batch()
        step = _GDN_STEP.get()
        if step is not None and step.forward_batch is not forward_batch:
            step = None
        if forward_batch is None or not (
            step.verify
            if step is not None
            else forward_batch.forward_mode.is_target_verify()
        ):
            if mixed_qkv.stride(-1) != 1:
                mixed_qkv = mixed_qkv.contiguous()
            dtype = mixed_qkv.dtype
            if a.dtype != dtype:
                a = a.to(dtype=dtype)
            if b.dtype != dtype:
                b = b.to(dtype=dtype)
            if self.dt_bias.dtype != dtype:
                self.dt_bias = torch.nn.Parameter(
                    self.dt_bias.detach().to(dtype=dtype), requires_grad=False
                )
            out = super().forward(mixed_qkv, b, a, core_attn_out, layer_name)
            if step is None or step.track == _GDNStep.LAYER:
                self._track_mamba_states(forward_batch, mixed_qkv)
            elif step.track == _GDNStep.EXTEND:
                self._track_extend(step, mixed_qkv)
            return out

        attn_backend = SGLangGDNForwardContext._resolve_attn_backend(forward_batch)
        if attn_backend is None:
            raise RuntimeError(
                "ATOM Qwen3.5 TARGET_VERIFY requires an active SGLang "
                "attention backend."
            )
        linear_backend = SGLangGDNForwardContext._linear_attn_backend(attn_backend)
        if getattr(linear_backend, "forward_metadata", None) is None:
            raise RuntimeError(
                "ATOM Qwen3.5 TARGET_VERIFY requires initialized SGLang "
                "GDN metadata."
            )
        req_to_token_pool = getattr(linear_backend, "req_to_token_pool", None)
        if req_to_token_pool is None:
            raise RuntimeError(
                "ATOM Qwen3.5 TARGET_VERIFY requires the SGLang mamba pool."
            )
        layer_cache = req_to_token_pool.mamba2_layer_cache(self.layer_num)
        if not hasattr(layer_cache, "intermediate_ssm") or not hasattr(
            layer_cache, "intermediate_conv_window"
        ):
            raise RuntimeError(
                "ATOM Qwen3.5 TARGET_VERIFY requires speculative GDN "
                "intermediate-state buffers."
            )

        draft_token_num = int(forward_batch.spec_info.draft_token_num)
        bs = int(forward_batch.batch_size)
        # Validate the token count before the views below. `mixed_qkv`, `a` and
        # `b` are reshaped with a trailing -1, which silently produces a wrong
        # trailing dimension (rather than raising) whenever the row count is a
        # different multiple of bs * draft_token_num.
        expected_tokens = bs * draft_token_num
        for name, tensor in (
            ("mixed_qkv", mixed_qkv),
            ("a", a),
            ("b", b),
            ("core_attn_out", core_attn_out),
        ):
            if tensor.shape[0] != expected_tokens:
                raise RuntimeError(
                    "ATOM GDN TARGET_VERIFY expected "
                    f"{expected_tokens} tokens (batch_size {bs} x "
                    f"draft_token_num {draft_token_num}) but {name} has "
                    f"{tensor.shape[0]}."
                )
        cache_indices = linear_backend.forward_metadata.mamba_cache_indices[:bs]
        conv_states = layer_cache.conv[0]
        # Same view Native passes to GatedDeltaNet: FlyDSL decode keeps a
        # logical KV view over physical VK storage. Verify reads that view.
        ssm_states = layer_cache.temporal
        policy = getattr(self, "gdn_flydsl_policy", None)
        if policy is not None and policy.decode:
            ssm_states = ssm_states.transpose(-1, -2)
        mixed_blocks = mixed_qkv.view(bs, draft_token_num, -1)
        a_blocks = a.view(bs, draft_token_num, -1)
        b_blocks = b.view(bs, draft_token_num, -1)
        output_blocks = core_attn_out.view(
            bs, draft_token_num, *core_attn_out.shape[1:]
        )
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(-1)
        )

        verify = (
            self._verify_replayssm if replayssm_enabled() else self._verify_batched_ssm
        )
        verify(
            layer_cache=layer_cache,
            conv_states=conv_states,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            mixed_blocks=mixed_blocks,
            a_blocks=a_blocks,
            b_blocks=b_blocks,
            output_blocks=output_blocks,
            conv_weights=conv_weights,
            bs=bs,
            draft_token_num=draft_token_num,
        )

        return core_attn_out

    def _track_mamba_states(self, forward_batch: Any, mixed_qkv: torch.Tensor) -> None:
        """Fill SGLang's mamba radix cache (extra_buffer) track slots.

        SGLang's HybridLinearAttnBackend owns the prefix cache and computes the
        track metadata, but the snapshot calls live inside its own GDN kernels,
        which the plugin replaces. They are made here, after the ATOM forward,
        from SGLang's track metadata. On device the conv window, the unaligned
        `h` rows and the aligned slot copies go out in one launch
        (`write_radix_track_snapshots`, the same copies SGLang's eager code
        makes); anything that kernel does not cover falls back to SGLang's
        helper. Every source is final by now: the conv window is read from the
        pre-conv input, the SSM state from the prefill kernel's chunk states
        `h` (Triton or FlyDSL) or the runtime slot.

        Decode goes through `_track_mamba_radix_slots`, which also covers the
        replayed decode graph via the post-replay epilogue.

        This is the per-layer path; a bound forward takes `_track_extend`.
        """
        if (
            forward_batch is None
            or getattr(forward_batch, "mamba_track_mask", None) is None
        ):
            return
        mode = forward_batch.forward_mode
        if mode.is_decode_or_idle():
            _track_mamba_radix_slots(forward_batch, int(self.layer_num))
            return
        if not mode.is_extend():
            return
        attn_backend = SGLangGDNForwardContext._resolve_attn_backend(forward_batch)
        if attn_backend is None:
            return
        linear_backend = SGLangGDNForwardContext._linear_attn_backend(attn_backend)
        metadata = getattr(linear_backend, "forward_metadata", None)
        if metadata is None:
            return
        layer_cache = _mamba_layer_cache(
            linear_backend.req_to_token_pool, self.layer_num
        )
        conv_states = layer_cache.conv[0]
        # `h` is in the logical [K, V] layout; FlyDSL decode stores VK.
        ssm_states = layer_cache.temporal
        policy = getattr(self, "gdn_flydsl_policy", None)
        if policy is not None and policy.decode:
            ssm_states = ssm_states.transpose(-1, -2)

        h = pop_last_intermediate_states()
        if not metadata.has_mamba_track_mask:
            return
        track_conv_indices = metadata.track_conv_indices
        dst = None
        if track_conv_indices is not None and track_conv_indices.numel() > 0:
            # Only SGLang's GDN backend caches the destination slots; indexing
            # the pool with a None would broadcast over every slot.
            dst = getattr(metadata, "conv_states_mask_indices", None)
            if dst is None:
                rows = forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
                dst = metadata.mamba_track_indices[rows]
        else:
            track_conv_indices = None
        if metadata.track_ssm_h_src.numel() > 0 and h is None:
            raise RuntimeError(
                "SGLang mamba radix cache needs the GDN chunk kernel's "
                "intermediate states, but this prefill kept none."
            )
        if mixed_qkv.is_cuda and fused_track_supported(metadata, h):
            write_radix_track_snapshots(
                mixed_qkv,
                track_conv_indices,
                dst,
                conv_states,
                h,
                metadata.track_ssm_h_src,
                metadata.track_ssm_h_dst,
                metadata.track_ssm_final_src,
                metadata.track_ssm_final_dst,
                ssm_states,
            )
            return
        if track_conv_indices is not None:
            # [rows, state_len, dim] -> the pool's [slot, dim, state_len].
            conv_states[dst] = mixed_qkv[track_conv_indices].transpose(1, 2)
        linear_backend._track_mamba_state_extend(forward_batch, h, ssm_states, metadata)

    def _track_extend(self, step: _GDNStep, mixed_qkv: torch.Tensor) -> None:
        """`_track_mamba_states` for an extend whose forward-level half (backend,
        metadata, slots, launch plan) `step` already holds."""
        layer_cache = _mamba_layer_cache(step.pool, self.layer_num)
        h = pop_last_intermediate_states()
        plan = step.plan
        if plan is None:
            return
        if plan.n_h and h is None:
            raise RuntimeError(
                "SGLang mamba radix cache needs the GDN chunk kernel's "
                "intermediate states, but this prefill kept none."
            )
        conv_states = layer_cache.conv[0]
        # `h` is in the logical [K, V] layout; FlyDSL decode stores VK.
        ssm_states = layer_cache.temporal
        policy = getattr(self, "gdn_flydsl_policy", None)
        if policy is not None and policy.decode:
            ssm_states = ssm_states.transpose(-1, -2)
        if mixed_qkv.is_cuda and (not plan.n_h or h.squeeze(0).dim() == 4):
            plan.write(mixed_qkv, conv_states, h, ssm_states)
            return
        if step.conv_idx is not None:
            conv_states[step.conv_slots] = mixed_qkv[step.conv_idx].transpose(1, 2)
        step.linear_backend._track_mamba_state_extend(
            step.forward_batch, h, ssm_states, step.metadata
        )

    def _verify_replayssm(
        self,
        *,
        layer_cache: Any,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        mixed_blocks: torch.Tensor,
        a_blocks: torch.Tensor,
        b_blocks: torch.Tensor,
        output_blocks: torch.Tensor,
        conv_weights: torch.Tensor,
        bs: int,
        draft_token_num: int,
    ) -> None:
        """Verify the draft window through Native ReplaySSM.

        The live checkpoint is the contiguous pool Triton prefill writes.
        The kernel appends records and does not publish a per-step SSM
        snapshot; ``update_mamba_state_after_mtp_verify`` advances the cursor
        once the accept length is known, scatters the conv window, and
        rebuilds the SSM state of any radix-tracked step into its track slot.
        """
        del ssm_states  # logical view; the ring addresses the raw pool
        # FlyDSL's transpose view is not the ReplaySSM layout. The policy
        # above leaves the pool untransposed; refuse a strided checkpoint
        # rather than silently replaying the wrong axes.
        ckpt = layer_cache.temporal
        if not ckpt.is_contiguous() or ckpt.stride(-1) != 1:
            raise RuntimeError(
                "ReplaySSM verify requires a contiguous GDN checkpoint "
                f"(stride={tuple(ckpt.stride())})."
            )
        query, key, value, g, beta = self._spec_conv_qkv(
            layer_cache=layer_cache,
            conv_states=conv_states,
            cache_indices=cache_indices,
            mixed_blocks=mixed_blocks,
            a_blocks=a_blocks,
            b_blocks=b_blocks,
            conv_weights=conv_weights,
            bs=bs,
            draft_token_num=draft_token_num,
        )
        buf_k, buf_u, buf_g, write_pos = prepare_layer(
            self.layer_num,
            ckpt,
            num_v_heads=self.num_v_heads // self.tp_size,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            max_query_len=draft_token_num,
            activation=query,
        )
        slot_idx = cache_indices
        if slot_idx.dtype != torch.int32 or not slot_idx.is_contiguous():
            if ckpt.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "ReplaySSM slot indices must be a stable contiguous "
                    "int32 view during CUDA graph capture."
                )
            slot_idx = slot_idx.to(dtype=torch.int32).contiguous()
        block_output = replayssm_gated_delta_rule(
            q=query,
            k=key,
            v=value,
            g=g,
            beta=beta,
            ckpt=ckpt,
            buf_k=buf_k,
            buf_u=buf_u,
            buf_g=buf_g,
            write_pos=write_pos,
            slot_idx=slot_idx,
            cu_seqlens=self._spec_cu_seqlens(bs, draft_token_num, mixed_blocks.device),
            max_query_len=draft_token_num,
            use_qk_l2norm_in_kernel=True,
            is_kda=False,
            route=envs.ATOM_REPLAYSSM_ROUTE,
        )
        output_blocks.copy_(
            block_output.view(bs, draft_token_num, *output_blocks.shape[2:])
        )

    def _spec_ssm_slot_table(
        self, bs: int, draft_token_num: int, device: torch.device
    ) -> torch.Tensor:
        """`[bs, draft]` table into the flat `intermediate_ssm` slot pool.

        Cached per shape and filled in place: CUDA graph replay requires the
        tensor address to stay put across iterations.
        """
        cache = getattr(self, "_spec_slot_table_cache", None)
        if cache is None:
            cache = {}
            self._spec_slot_table_cache = cache
        key = (bs, draft_token_num, device)
        table = cache.get(key)
        if table is None:
            table = torch.arange(bs, device=device, dtype=torch.int32).unsqueeze(
                1
            ) * draft_token_num + torch.arange(
                draft_token_num, device=device, dtype=torch.int32
            ).unsqueeze(
                0
            )
            cache[key] = table
        return table

    def _verify_batched_ssm(
        self,
        *,
        layer_cache: Any,
        conv_states: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        mixed_blocks: torch.Tensor,
        a_blocks: torch.Tensor,
        b_blocks: torch.Tensor,
        output_blocks: torch.Tensor,
        conv_weights: torch.Tensor,
        bs: int,
        draft_token_num: int,
    ) -> None:
        """Target verify with the SSM recurrent folded into one kernel launch.

        The projected q/k/v of the whole draft block are handed to a single
        `fused_recurrent_gated_delta_rule` call that addresses
        `intermediate_ssm` as a flat `[slot * step]` pool via a 2D index table,
        so every per-step state lands where SGLang's
        `fused_mamba_state_scatter_with_mask` expects it. The live SSM state is
        never written, so it needs no snapshot/restore.

        The conv update is folded into one wide-window call over SGLang's
        deduplicated sliding-window `intermediate_conv_window`.

        Equivalence with the stepwise loop is covered bit-for-bit by
        tests/plugin/test_gdn_target_verify_batched_equiv.py and
        tests/plugin/test_sglang_gdn_verify_batched_ssm.py.
        """
        query_all, key_all, value_all, g, beta = self._spec_conv_qkv(
            layer_cache=layer_cache,
            conv_states=conv_states,
            cache_indices=cache_indices,
            mixed_blocks=mixed_blocks,
            a_blocks=a_blocks,
            b_blocks=b_blocks,
            conv_weights=conv_weights,
            bs=bs,
            draft_token_num=draft_token_num,
        )

        # intermediate_ssm[:, step] is indexed by batch position (see SGLang's
        # fused_mamba_state_scatter_with_mask: src[:, i, step_indices[i]]), so
        # slot = i * draft_token_num + step over the flattened per-layer view.
        pool = layer_cache.intermediate_ssm
        if not pool.is_contiguous():
            raise RuntimeError(
                "ATOM GDN batched TARGET_VERIFY requires a contiguous "
                "intermediate_ssm buffer."
            )
        slot_table = self._spec_ssm_slot_table(bs, draft_token_num, mixed_blocks.device)
        flat_phys = pool.view(pool.shape[0] * pool.shape[1], *pool.shape[2:])
        # FlyDSL keeps live state as a logical KV view over physical VK.
        # ``intermediate_ssm`` is that same physical layout. A transpose view
        # lets Triton write logical KV straight into the bytes SGLang scatters.
        logical_kv = ssm_states.stride(-2) == 1 and ssm_states.stride(-1) != 1
        if logical_kv:
            src = ssm_states[cache_indices]
            pool[:bs, 0].copy_(src.transpose(-1, -2))
            recurrent_state = flat_phys.transpose(-1, -2)
        else:
            # The kernel loads h0 once, before its internal step loop, so seeding
            # step 0's slot with the live state and letting step 0 overwrite that
            # same slot is safe.
            pool[:bs, 0] = ssm_states[cache_indices]
            recurrent_state = flat_phys

        cu_seqlens = self._spec_cu_seqlens(bs, draft_token_num, mixed_blocks.device)
        block_output, _ = fused_recurrent_gated_delta_rule(
            q=query_all,
            k=key_all,
            v=value_all,
            g=g,
            beta=beta,
            initial_state=recurrent_state,
            inplace_final_state=True,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=slot_table,
            use_qk_l2norm_in_kernel=True,
        )
        output_blocks.copy_(
            block_output.view(bs, draft_token_num, *output_blocks.shape[2:])
        )

    def _spec_conv_qkv(
        self,
        *,
        layer_cache: Any,
        conv_states: torch.Tensor,
        cache_indices: torch.Tensor,
        mixed_blocks: torch.Tensor,
        a_blocks: torch.Tensor,
        b_blocks: torch.Tensor,
        conv_weights: torch.Tensor,
        bs: int,
        draft_token_num: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Conv + gate for one verify window. q/k/v are `[1, tokens, heads, dim]`."""
        num_k_heads = self.num_k_heads // self.tp_size
        num_v_heads = self.num_v_heads // self.tp_size
        num_tokens = bs * draft_token_num
        conv_phys = self._spec_conv_window_phys(
            layer_cache.intermediate_conv_window[0], draft_token_num
        )
        # One wide-window spec call. ATOM writes
        # [history2..historyM, draft1..draftN] -- exactly the physical row
        # behind SGLang's dedup view, so every per-step window materialises
        # for free and the live conv state is only read, never written.
        state_len = conv_states.shape[-1]
        conv_phys[:bs, :, :state_len] = conv_states[cache_indices]
        query_all, key_all, value_all = causal_conv1d_update(
            mixed_blocks.reshape(num_tokens, -1),
            conv_phys,
            conv_weights,
            num_k_heads * self.head_k_dim,
            num_v_heads * self.head_v_dim,
            self.conv1d.bias,
            self.activation,
            conv_state_indices=self._spec_conv_slot_table(bs, mixed_blocks.device),
            num_accepted_tokens=self._spec_conv_accepted(bs, mixed_blocks.device),
            query_start_loc=self._spec_cu_seqlens(
                bs, draft_token_num, mixed_blocks.device
            ),
            max_query_len=draft_token_num,
            validate_data=False,
        )
        g, beta = fused_gdn_gating(
            self.A_log,
            a_blocks.reshape(num_tokens, -1),
            b_blocks.reshape(num_tokens, -1),
            self.dt_bias,
        )
        return (
            query_all.view(1, num_tokens, num_k_heads, self.head_k_dim),
            key_all.view(1, num_tokens, num_k_heads, self.head_k_dim),
            value_all.view(1, num_tokens, num_v_heads, self.head_v_dim),
            g,
            beta,
        )

    @staticmethod
    def _spec_conv_window_phys(
        window_view: torch.Tensor, draft_token_num: int
    ) -> torch.Tensor:
        """Recover the physical wide-window buffer behind SGLang's dedup view.

        SGLang stores the conv intermediates for a linear draft chain as one
        shared `[slot, dim, D + K - 2]` row per (layer, slot) and exposes an
        overlapping `as_strided` view of logical shape `[slot, D, dim, K - 1]`
        with `view[s, t, d, w] = phys[s, d, t + w]` (see `MambaPool.__init__`
        and `conv_window_dedup_enabled` in SGLang's memory_pool.py). That
        physical row is exactly what ATOM's spec `causal_conv1d_update` writes,
        so recovering it lets one call replace the whole per-step loop.

        Raises when the view is not that dedup layout. DFLASH target verify on
        ROCm requires the linear-chain deduplicated layout.
        """
        if window_view.ndim != 4:
            raise RuntimeError(
                "ATOM GDN batched TARGET_VERIFY requires a rank-4 "
                "intermediate_conv_window."
            )
        num_slots, draft, dim, win = window_view.shape
        if draft != draft_token_num:
            raise RuntimeError(
                "ATOM GDN batched TARGET_VERIFY received an incompatible "
                f"draft dimension: expected {draft_token_num}, got {draft}."
            )
        shared_win = draft + win - 1
        stride_slot, stride_step, stride_dim, stride_win = window_view.stride()
        # Dedup layout aliases the step and window axes onto the shared-window
        # axis (both stride 1) and gives the dim axis the shared-window pitch.
        if (
            stride_step != 1
            or stride_win != 1
            or stride_dim != shared_win
            or stride_slot != dim * shared_win
        ):
            raise RuntimeError(
                "ATOM GDN batched TARGET_VERIFY requires SGLang's "
                "deduplicated sliding-window intermediate_conv_window "
                f"(shape={tuple(window_view.shape)} "
                f"stride={tuple(window_view.stride())})."
            )
        return window_view.as_strided(
            (num_slots, dim, shared_win),
            (dim * shared_win, shared_win, 1),
            window_view.storage_offset(),
        )

    def _spec_conv_slot_table(self, bs: int, device: torch.device) -> torch.Tensor:
        """`[bs, 1]` conv slot table; row i is batch position i, matching
        SGLang's `fused_conv_window_scatter_with_mask` source indexing."""
        cache = getattr(self, "_spec_conv_slot_cache", None)
        if cache is None:
            cache = {}
            self._spec_conv_slot_cache = cache
        key = (bs, device)
        table = cache.get(key)
        if table is None:
            table = torch.arange(bs, device=device, dtype=torch.int32).reshape(bs, 1)
            cache[key] = table
        return table

    def _spec_conv_accepted(self, bs: int, device: torch.device) -> torch.Tensor:
        """`num_accepted_tokens` of all ones: the conv kernel reads its initial
        history from column `num_accepted_tokens - 1`, and verify always starts
        from the committed live state seeded at column 0."""
        cache = getattr(self, "_spec_conv_accepted_cache", None)
        if cache is None:
            cache = {}
            self._spec_conv_accepted_cache = cache
        key = (bs, device)
        accepted = cache.get(key)
        if accepted is None:
            accepted = torch.ones(bs, device=device, dtype=torch.int32)
            cache[key] = accepted
        return accepted

    def _spec_cu_seqlens(
        self, bs: int, draft_token_num: int, device: torch.device
    ) -> torch.Tensor:
        """`[bs + 1]` block boundaries, cached for CUDA graph address stability."""
        cache = getattr(self, "_spec_cu_seqlens_cache", None)
        if cache is None:
            cache = {}
            self._spec_cu_seqlens_cache = cache
        key = (bs, draft_token_num, device)
        cu_seqlens = cache.get(key)
        if cu_seqlens is None:
            cu_seqlens = torch.arange(
                0,
                (bs + 1) * draft_token_num,
                draft_token_num,
                device=device,
                dtype=torch.int32,
            )
            cache[key] = cu_seqlens
        return cu_seqlens


class GDNAttentionBackend:
    @staticmethod
    def get_name() -> str:
        return "ROCM_GDN_ATTENTION"

    @staticmethod
    def get_impl_cls() -> type[GatedDeltaNet]:
        return SGLangGatedDeltaNet


@dataclass(frozen=True)
class SGLangGDNForwardContext:
    """Precomputed ATOM forward-context state derived from SGLang metadata."""

    forward_batch: Any
    gdn_metadata: GDNAttentionMetadata | None
    kv_cache_data: dict[str, KVCacheTensor]
    context: Context
    num_tokens: int
    # What every GDN layer of this forward reads after its compute.
    step: _GDNStep | None = None

    @staticmethod
    def _linear_attn_backend(attn_backend: Any) -> Any:
        return getattr(attn_backend, "linear_attn_backend", attn_backend)

    @staticmethod
    def _resolve_attn_backend(forward_batch: Any) -> Any:
        return resolve_attn_backend(forward_batch)

    @staticmethod
    def _patch_forward_batch_pools(forward_batch: Any, attn_backend: Any) -> None:
        for attr in ("token_to_kv_pool", "req_to_token_pool"):
            if getattr(forward_batch, attr, None) is None:
                pool = getattr(attn_backend, attr, None)
                if pool is not None:
                    try:
                        setattr(forward_batch, attr, pool)
                    except Exception:  # noqa: BLE001, S110
                        pass

    @staticmethod
    def _build_kv_cache_tensors(
        forward_batch: Any, attn_backend: Any
    ) -> dict[str, KVCacheTensor]:
        pool = resolve_mamba_req_pool(forward_batch, attn_backend)
        if pool is None or getattr(pool, "mamba_map", None) is None:
            try:
                from sglang.srt.model_executor.forward_context import (
                    get_req_to_token_pool,
                    has_forward_context,
                )

                if has_forward_context():
                    pool = get_req_to_token_pool()
            except Exception:  # noqa: BLE001 - forward context is optional
                pool = None
        if pool is None:
            return {}

        mamba_map = getattr(pool, "mamba_map", None)
        if mamba_map is None:
            return {}

        views = _pool_layer_views(pool)
        if views is not None and views.kv_tensors is not None:
            # Same pool state, so the same layer views as the last build; only
            # the decode policy (bound on first use) could change the result.
            cached_view, first_temporal, cached = views.kv_tensors
            policy = (
                _ensure_flydsl_policy(first_temporal)
                if first_temporal is not None
                else None
            )
            if bool(policy is not None and policy.decode) == cached_view:
                return dict(cached)

        first_temporal = None
        layer_caches: list[tuple[int, Any]] = []
        for layer_id in mamba_map:
            layer_cache = _mamba_layer_cache(pool, layer_id)
            layer_caches.append((layer_id, layer_cache))
            if first_temporal is None:
                first_temporal = layer_cache.temporal
        policy = (
            _ensure_flydsl_policy(first_temporal)
            if first_temporal is not None
            else None
        )
        use_kv_view = bool(policy is not None and policy.decode)

        out: dict[str, KVCacheTensor] = {}
        for layer_id, layer_cache in layer_caches:
            v_cache = layer_cache.temporal
            if use_kv_view:
                v_cache = v_cache.transpose(-1, -2)
            out[f"layer_{layer_id}"] = KVCacheTensor(
                layer_num=layer_id,
                k_cache=layer_cache.conv[0],
                v_cache=v_cache,
                k_scale=None,
                v_scale=None,
                # Slot-addressed recurrent state, not paged KV -- see
                # `KVCacheTensor.per_request_state`.
                per_request_state=True,
            )
        if views is not None:
            views.kv_tensors = (use_kv_view, first_temporal, out)
            return dict(out)
        return out

    @staticmethod
    def _build_context(forward_batch: Any) -> tuple[Context, int]:
        mode = forward_batch.forward_mode
        # TARGET_VERIFY is is_extend() by default, so it must be checked
        # first. DRAFT_EXTEND_V2 is not is_extend() unless
        # include_draft_extend_v2=True. Both are packed rectangles
        # (bs * draft tokens), not a seq_lens_sum prefill.
        is_draft_ext = bool(
            callable(getattr(mode, "is_draft_extend_v2", None))
            and mode.is_draft_extend_v2()
        )
        if mode.is_target_verify() or is_draft_ext:
            is_prefill = False
            num_tokens = int(forward_batch.positions.numel())
        elif mode.is_extend():
            is_prefill = True
            num_tokens = int(forward_batch.seq_lens_sum)
        else:
            is_prefill = bool(mode.is_prefill())
            num_tokens = int(forward_batch.batch_size)
        atom_config = get_current_atom_config()
        enable_dp_attention = bool(getattr(atom_config, "enable_dp_attention", False))
        global_forward_mode = getattr(forward_batch, "global_forward_mode", None)
        effective_forward_mode = (
            global_forward_mode if global_forward_mode is not None else mode
        )
        running_tokens_are_unified = not enable_dp_attention or bool(
            effective_forward_mode.is_decode_or_idle()
        )
        return (
            Context(
                positions=forward_batch.positions,
                is_prefill=is_prefill,
                is_dummy_run=mode.is_idle(),
                scheduled_bs=forward_batch.batch_size,
                running_bs=forward_batch.batch_size,
                # `num_tokens` is already this step's flat row count, so no
                # per-request multiplier is needed (nor available here).
                scheduled_tokens=num_tokens,
                running_tokens=num_tokens,
                running_tokens_are_unified=running_tokens_are_unified,
            ),
            num_tokens,
        )

    @staticmethod
    def _build_gdn_metadata(
        forward_batch: Any, linear_backend: Any
    ) -> GDNAttentionMetadata | None:
        mode = forward_batch.forward_mode
        if mode.is_target_verify():
            # SGLangGatedDeltaNet fills SGLang's transactional snapshots using
            # ATOM's stepwise kernels. Keep the outer ATOM context active for
            # MoE, norms and collectives without native GDN metadata.
            return None

        bs = forward_batch.batch_size
        fm = getattr(linear_backend, "forward_metadata", None)
        query_start_loc = getattr(fm, "query_start_loc", None)
        idx = getattr(fm, "mamba_cache_indices", None)
        live_slots = idx
        # Extend only: SGLang's host sum of extend_seq_lens, i.e. the value of
        # query_start_loc[-1].
        num_query_tokens = getattr(fm, "logical_num_tokens", None)
        if query_start_loc is None or idx is None:
            reconstructed = reconstruct_linear_metadata(forward_batch, linear_backend)
            if reconstructed is None:
                return None
            query_start_loc, idx = reconstructed
            num_query_tokens = None
        device = query_start_loc.device
        # Decode cuda-graph: keep Hybrid's static buffers (already pad=-1 on
        # dead rows in ``_replay_metadata``). Cloning here bakes a throwaway
        # address into the full decode graph and Native GDN keeps writing the
        # capture-time slots → GPU fault under radix/extra_buffer recycle.
        # Prefill/extend is eager; pad-sentinel clone is fine there.
        if mode.is_decode_or_idle():
            # Never ``.to()`` / clone here — that bakes a throwaway address into
            # the full decode graph. Hybrid ``state_indices_list`` is already
            # int32 on device; ``_align_flydsl_decode_slots`` enforces the view.
            if query_start_loc.device != device:
                raise RuntimeError(
                    "GDN decode query_start_loc must already live on the Hybrid "
                    f"cuda-graph device ({device}), got {query_start_loc.device}"
                )
            idx = _align_flydsl_decode_slots(idx, bs)
        else:
            idx = idx.to(dtype=torch.int32, device=device)
            idx, query_start_loc = apply_gdn_pad_sentinels(
                forward_batch, idx, query_start_loc, mode, bs
            )
        common_kwargs = {
            "num_spec_decodes": 0,
            "num_spec_decode_tokens": 0,
            "spec_query_start_loc": None,
            "non_spec_query_start_loc": query_start_loc,
            "spec_state_indices_tensor": None,
            "non_spec_state_indices_tensor": idx,
            # SGLang owns the mamba slots on this path and never forks a
            # request's state, so the slots the state is read from are the ones
            # it is written to. GatedDeltaNet.forward indexes this
            # unconditionally, so leaving it at its `None` default makes the
            # non-verify path raise as soon as a GDN layer runs.
            "non_spec_state_indices_in_tensor": idx,
            "spec_sequence_masks": None,
            "spec_token_indx": None,
            "non_spec_token_indx": None,
            "num_accepted_tokens": None,
        }

        if mode.is_decode_or_idle():
            return GDNAttentionMetadata(
                num_prefills=0,
                num_prefill_tokens=0,
                num_decodes=bs,
                num_decode_tokens=bs,
                num_actual_tokens=bs,
                has_initial_state=None,
                nums_dict=None,
                batch_ptr=None,
                token_chunk_offset_ptr=None,
                **common_kwargs,
            )

        if mode.is_extend():
            if replayssm_enabled():
                # Chunked prefill rewrites the checkpoint in place. The ring
                # cursor, shared by every GDN layer, has to drop the previous
                # tenant's records once per forward or the next verify folds
                # them on top of that new state.
                note_full_state_write(None if live_slots is None else live_slots[:bs])
            # SGLang's seq_lens_sum includes cached prefix tokens for some
            # hybrid batches; GDN only receives the active query tokens.
            if num_query_tokens is None:
                num_query_tokens = query_start_loc[-1].item()
            seq_sum = int(num_query_tokens)
            epl = forward_batch.extend_prefix_lens
            has_initial_state = None if epl is None else epl > 0
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(query_start_loc, seq_sum)
            )
            # GatedDeltaNet's prefill reads these once-per-step fields (#2534)
            # instead of casting per layer. The native builder fills them; a
            # None here makes the non-FlyDSL prefill (e.g. ReplaySSM under MTP)
            # and the state write-back fail. Read and write slots are the same.
            idx_i64 = idx.long()
            no_initial_state = None if has_initial_state is None else ~has_initial_state
            return GDNAttentionMetadata(
                num_prefills=bs,
                num_prefill_tokens=seq_sum,
                num_decodes=0,
                num_decode_tokens=0,
                num_actual_tokens=seq_sum,
                has_initial_state=has_initial_state,
                nums_dict=nums_dict,
                batch_ptr=batch_ptr,
                token_chunk_offset_ptr=token_chunk_offset_ptr,
                no_initial_state=no_initial_state,
                non_spec_state_indices_i64=idx_i64,
                non_spec_state_indices_in_i64=idx_i64,
                flydsl_prefill_metadata=_flydsl_prefill_metadata(
                    query_start_loc, forward_batch
                ),
                # SGLangGatedDeltaNet snapshots radix-cache states out of `h`.
                keep_intermediate_states=bool(
                    getattr(fm, "has_mamba_track_mask", False)
                ),
                **common_kwargs,
            )

        logger.warning(
            "SGLang GDN forward context: unsupported forward_mode=%s; GDN metadata skipped.",
            mode,
        )
        return None

    @classmethod
    def build(cls, forward_batch_or_metadata: Any) -> SGLangGDNForwardContext | None:
        from atom.plugin.sglang.runtime import (
            SGLangForwardBatchMetadata,
        )

        metadata = SGLangForwardBatchMetadata.build(forward_batch_or_metadata)
        if metadata is None or metadata.forward_batch is None:
            return None

        forward_batch = metadata.forward_batch
        attn_backend = cls._resolve_attn_backend(forward_batch)
        if attn_backend is None:
            logger.warning(
                "SGLang GDN forward context: no active SGLang attention backend; "
                "GDN metadata skipped."
            )
            return None

        cls._patch_forward_batch_pools(forward_batch, attn_backend)
        linear_backend = cls._linear_attn_backend(attn_backend)
        kv_cache_data = cls._build_kv_cache_tensors(forward_batch, linear_backend)
        if not kv_cache_data:
            return None

        gdn_metadata = cls._build_gdn_metadata(forward_batch, linear_backend)
        if gdn_metadata is None and not forward_batch.forward_mode.is_target_verify():
            return None

        context, num_tokens = cls._build_context(forward_batch)
        return cls(
            forward_batch=forward_batch,
            gdn_metadata=gdn_metadata,
            kv_cache_data=kv_cache_data,
            context=context,
            num_tokens=num_tokens,
            step=_GDNStep(forward_batch, linear_backend),
        )

    @classmethod
    @contextmanager
    def bind(cls, forward_batch_or_metadata: Any) -> Iterator[None]:
        forward_context = cls.build(forward_batch_or_metadata)
        if forward_context is None:
            yield
            return

        prev_kv = _forward_kv_cache_context.kv_cache_data
        current_context = get_forward_context()
        reuse_current_context = current_context.context is not None
        prev_attn_metadata = current_context.attn_metadata
        prev_context_kv = current_context.kv_cache_data
        active_kv = forward_context.kv_cache_data
        if reuse_current_context:
            active_kv = dict(prev_context_kv or {})
            active_kv.update(forward_context.kv_cache_data)
        step = getattr(forward_context, "step", None)
        step_token = None if step is None else _GDN_STEP.set(step)
        try:
            set_kv_cache_data(active_kv)
            attn_md = (
                copy.copy(prev_attn_metadata)
                if reuse_current_context and prev_attn_metadata is not None
                else AttentionMetaData()
            )
            attn_md.gdn_metadata = forward_context.gdn_metadata
            if reuse_current_context:
                # SGLangPluginRuntime already created the cross-rank-consistent
                # Context and DPMetadata. Rebuilding it here would issue a second
                # CPU all-reduce only on ranks where GDN metadata exists; idle
                # ranks skip this binder and would never join that collective.
                # Preserve all outer attention fields while injecting GDN data.
                current_context.attn_metadata = attn_md
                current_context.kv_cache_data = active_kv
            else:
                set_forward_context(
                    attn_metadata=attn_md,
                    atom_config=get_current_atom_config(),
                    context=forward_context.context,
                    running_tokens_per_dp_rank=DPMetadata.gather_running_tokens(
                        get_current_atom_config().parallel_config,
                        forward_context.num_tokens,
                    ),
                )
            yield
        finally:
            if step_token is not None:
                _GDN_STEP.reset(step_token)
            if reuse_current_context:
                current_context.attn_metadata = prev_attn_metadata
                current_context.kv_cache_data = prev_context_kv
            else:
                reset_forward_context()
            set_kv_cache_data(prev_kv if prev_kv is not None else {})
