# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import copy
import logging
from collections.abc import Iterator
from contextlib import contextmanager
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
from atom.model_ops.fla_ops.replayssm import replayssm_gated_delta_rule
from atom.model_ops.mamba_ops.causal_conv1d import causal_conv1d_update
from atom.plugin.sglang.attention_backend.backend_resolver import (
    reconstruct_linear_metadata,
    resolve_attn_backend,
    resolve_mamba_req_pool,
)
from atom.plugin.sglang.attention_backend.gdn_replayssm import (
    note_full_state_write,
    prepare_layer,
    replayssm_enabled,
)
from atom.plugin.sglang.patches.qwen4_exp_recognition_patch import (
    apply_gdn_pad_sentinels,
)
from atom.utils import envs
from atom.utils.forward_context import (
    AttentionMetaData,
    Context,
    _forward_kv_cache_context,
    get_forward_context,
    reset_forward_context,
    set_forward_context,
    set_kv_cache_data,
)

logger = logging.getLogger(__name__)


def _align_flydsl_decode_slots(index: torch.Tensor, batch: int) -> torch.Tensor:
    """Contiguous int32 prefix of length ``batch`` for FlyDSL decode."""
    if index.shape[0] != batch:
        index = index[:batch]
    if index.dtype != torch.int32 or not index.is_contiguous():
        index = index.to(dtype=torch.int32).contiguous()
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


def _ensure_flydsl_policy(state: torch.Tensor):
    """Bind Native `select_policy` once, same as `_build_gdn_cache_tensor`."""
    impls = _flydsl_linear_impls()
    if not impls:
        return None
    policy = impls[0].gdn_flydsl_policy
    if policy is not None:
        return policy
    from atom.model_ops.fla_ops.gdn_flydsl import select_policy

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
        if forward_batch is None or not forward_batch.forward_mode.is_target_verify():
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
            # Chunked prefill rewrites the checkpoint in place. The ring
            # cursor has to drop the previous tenant's records or the next
            # verify folds them on top of that new state.
            mode = None if forward_batch is None else forward_batch.forward_mode
            draft_extend = bool(
                mode is not None
                and callable(getattr(mode, "is_draft_extend_v2", None))
                and mode.is_draft_extend_v2()
            )
            if (
                replayssm_enabled()
                and mode is not None
                and mode.is_extend()
                and not draft_extend
            ):
                note_full_state_write(self._verify_cache_slots(forward_batch))
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

    def _verify_cache_slots(self, forward_batch: Any) -> torch.Tensor | None:
        attn_backend = SGLangGDNForwardContext._resolve_attn_backend(forward_batch)
        if attn_backend is None:
            return None
        linear = SGLangGDNForwardContext._linear_attn_backend(attn_backend)
        metadata = getattr(linear, "forward_metadata", None)
        indices = getattr(metadata, "mamba_cache_indices", None)
        if indices is None:
            return None
        return indices[: int(forward_batch.batch_size)]

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
        once the accept length is known and scatters only the conv window.
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

        first_temporal = None
        layer_caches: list[tuple[int, Any]] = []
        for layer_id in mamba_map:
            layer_cache = pool.mamba2_layer_cache(layer_id)
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
        if query_start_loc is None or idx is None:
            reconstructed = reconstruct_linear_metadata(forward_batch, linear_backend)
            if reconstructed is None:
                return None
            query_start_loc, idx = reconstructed
        device = query_start_loc.device
        idx = idx.to(dtype=torch.int32, device=device)
        # Native GDN pad clone. Hybrid already pads Flash like 2.4T;
        # drop this after Native GDN consumes Hybrid buffers directly.
        # See apply_gdn_pad_sentinels in qwen4_exp_recognition_patch.py.
        idx, query_start_loc = apply_gdn_pad_sentinels(
            forward_batch, idx, query_start_loc, mode, bs
        )
        if mode.is_decode_or_idle():
            # FlyDSL checks the slot vector against q's batch axis. The graph
            # buffer is the capture bucket; keep a contiguous int32 prefix.
            idx = _align_flydsl_decode_slots(idx, bs)
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
            # SGLang's seq_lens_sum includes cached prefix tokens for some
            # hybrid batches; GDN only receives the active query tokens.
            seq_sum = int(query_start_loc[-1].item())
            epl = forward_batch.extend_prefix_lens
            has_initial_state = None if epl is None else epl > 0
            nums_dict, batch_ptr, token_chunk_offset_ptr = (
                compute_causal_conv1d_metadata(query_start_loc)
            )
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
                flydsl_prefill_metadata=_flydsl_prefill_metadata(
                    query_start_loc, forward_batch
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
                    num_tokens=forward_context.num_tokens,
                )
            yield
        finally:
            if reuse_current_context:
                current_context.attn_metadata = prev_attn_metadata
                current_context.kv_cache_data = prev_context_kv
            else:
                reset_forward_context()
            set_kv_cache_data(prev_kv if prev_kv is not None else {})
