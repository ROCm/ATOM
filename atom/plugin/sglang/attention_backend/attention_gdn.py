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
from atom.model_ops.mamba_ops.causal_conv1d import causal_conv1d_update
from atom.plugin.sglang.attention_backend.backend_resolver import (
    reconstruct_linear_metadata,
    resolve_attn_backend,
    resolve_mamba_req_pool,
)
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
            return super().forward(mixed_qkv, b, a, core_attn_out, layer_name)

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
        ssm_states = layer_cache.temporal
        mixed_blocks = mixed_qkv.view(bs, draft_token_num, -1)
        a_blocks = a.view(bs, draft_token_num, -1)
        b_blocks = b.view(bs, draft_token_num, -1)
        output_blocks = core_attn_out.view(
            bs, draft_token_num, *core_attn_out.shape[1:]
        )
        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(-1)
        )

        self._verify_batched_ssm(
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
        num_k_heads = self.num_k_heads // self.tp_size
        num_v_heads = self.num_v_heads // self.tp_size
        num_tokens = bs * draft_token_num
        k_dim = num_k_heads * self.head_k_dim
        v_dim = num_v_heads * self.head_v_dim

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
            k_dim,
            v_dim,
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

        # intermediate_ssm[:, step] is indexed by batch position (see SGLang's
        # fused_mamba_state_scatter_with_mask: src[:, i, step_indices[i]]), so
        # slot = i * draft_token_num + step over the flattened per-layer view.
        pool = layer_cache.intermediate_ssm
        if not pool.is_contiguous():
            raise RuntimeError(
                "ATOM GDN batched TARGET_VERIFY requires a contiguous "
                "intermediate_ssm buffer."
            )
        flat_pool = pool.view(pool.shape[0] * pool.shape[1], *pool.shape[2:])
        slot_table = self._spec_ssm_slot_table(bs, draft_token_num, mixed_blocks.device)
        # The kernel loads h0 once, before its internal step loop, so seeding
        # step 0's slot with the live state and letting step 0 overwrite that
        # same slot is safe.
        pool[:bs, 0] = ssm_states[cache_indices]

        cu_seqlens = self._spec_cu_seqlens(bs, draft_token_num, mixed_blocks.device)
        block_output, _ = fused_recurrent_gated_delta_rule(
            q=query_all.view(1, num_tokens, num_k_heads, self.head_k_dim),
            k=key_all.view(1, num_tokens, num_k_heads, self.head_k_dim),
            v=value_all.view(1, num_tokens, num_v_heads, self.head_v_dim),
            g=g,
            beta=beta,
            initial_state=flat_pool,
            inplace_final_state=True,
            cu_seqlens=cu_seqlens,
            ssm_state_indices=slot_table,
            use_qk_l2norm_in_kernel=True,
        )
        output_blocks.copy_(
            block_output.view(bs, draft_token_num, *output_blocks.shape[2:])
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

        out: dict[str, KVCacheTensor] = {}
        for layer_id in mamba_map:
            layer_cache = pool.mamba2_layer_cache(layer_id)
            layer_name = f"layer_{layer_id}"
            out[layer_name] = KVCacheTensor(
                layer_num=layer_id,
                k_cache=layer_cache.conv[0],
                v_cache=layer_cache.temporal,
                k_scale=None,
                v_scale=None,
            )
        return out

    @staticmethod
    def _build_context(forward_batch: Any) -> tuple[Context, int]:
        mode = forward_batch.forward_mode
        is_prefill = mode.is_prefill()
        if mode.is_extend():
            num_tokens = int(forward_batch.seq_lens_sum)
        elif mode.is_target_verify():
            # Total flattened tokens, i.e. batch_size * draft_token_num.
            # `numel()` is right whether SGLang hands us the usual flat
            # `[bs * draft]` positions or a `[bs, draft]` view; `shape[-1]`
            # would report just the per-sequence length for the latter.
            num_tokens = int(forward_batch.positions.numel())
        else:
            num_tokens = int(forward_batch.batch_size)
        atom_config = get_current_atom_config()
        enable_dp_attention = bool(getattr(atom_config, "enable_dp_attention", False))
        global_forward_mode = getattr(forward_batch, "global_forward_mode", None)
        effective_forward_mode = (
            global_forward_mode if global_forward_mode is not None else mode
        )
        dp_uniform_decode = not enable_dp_attention or bool(
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
                dp_uniform_decode=dp_uniform_decode,
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
        gdn_metadata = cls._build_gdn_metadata(forward_batch, linear_backend)
        if gdn_metadata is None and not forward_batch.forward_mode.is_target_verify():
            return None

        kv_cache_data = cls._build_kv_cache_tensors(forward_batch, linear_backend)
        if not kv_cache_data:
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
        try:
            set_kv_cache_data(forward_context.kv_cache_data)
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
                current_context.kv_cache_data = forward_context.kv_cache_data
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
