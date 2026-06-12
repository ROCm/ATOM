# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek MLA wrapper for SGLang plugin mode.

This adapter keeps the model-side entry at ``self.mla_attn(...)`` and owns the
SGLang-specific runtime dispatch for DeepSeek MLA. It is intentionally shaped
closer to the vLLM plugin path than the older model-side monkey-patched entry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch import nn

if TYPE_CHECKING:
    from atom.models.deepseek_v2 import DeepseekV2MLAAttention


class SGLangDeepseekMLAAttention(nn.Module):
    """Enter SGLang DeepSeek MLA runtime through ``self.mla_attn(...)``."""

    def __init__(
        self,
        owner_attn: "DeepseekV2MLAAttention",
        base_attn: nn.Module,
    ) -> None:
        super().__init__()
        # Keep a non-module back reference. Registering owner_attn as a child
        # module would create owner_attn -> mla_attn(wrapper) -> owner_attn and
        # make nn.Module.train/eval recurse forever.
        object.__setattr__(self, "owner_attn", owner_attn)
        self.base_attn = base_attn

    @property
    def attn(self):
        return getattr(self.base_attn, "attn", self.base_attn)

    def _get_forward_batch(self, kwargs: dict[str, Any]):
        forward_batch = kwargs.get("forward_batch", None)
        if forward_batch is None:
            from atom.plugin.sglang.runtime import (
                get_current_forward_batch,
            )

            forward_batch = get_current_forward_batch()
            kwargs["forward_batch"] = forward_batch
        if forward_batch is None:
            raise RuntimeError(
                "forward_batch is required for SGLang DeepSeek MLA wrapper"
            )
        return forward_batch

    def _infer_total_tokens(self, forward_batch, tensor: torch.Tensor) -> int:
        if hasattr(forward_batch, "input_ids") and forward_batch.input_ids is not None:
            return int(forward_batch.input_ids.shape[0])
        if hasattr(forward_batch, "positions") and forward_batch.positions is not None:
            return int(forward_batch.positions.shape[0])
        if hasattr(forward_batch, "seq_lens_sum"):
            return int(forward_batch.seq_lens_sum)
        return int(tensor.shape[0])

    def _maybe_all_gather(
        self,
        tensor: torch.Tensor | None,
        *,
        total_tokens: int,
        input_scattered: bool,
    ):
        if tensor is None or not input_scattered:
            return tensor
        from sglang.srt.distributed import get_tp_group

        output = tensor.new_empty((total_tokens, *tensor.shape[1:]))
        get_tp_group().all_gather_into_tensor(output, tensor)
        return output

    def _gather_runtime_inputs(
        self,
        q_input: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
        q_scale: torch.Tensor | None,
        *,
        forward_batch,
        input_scattered: bool,
    ):
        total_tokens = self._infer_total_tokens(forward_batch, q_input)
        q_input = self._maybe_all_gather(
            q_input,
            total_tokens=total_tokens,
            input_scattered=input_scattered,
        )
        kv_c_normed = self._maybe_all_gather(
            kv_c_normed,
            total_tokens=total_tokens,
            input_scattered=input_scattered,
        )
        k_pe = self._maybe_all_gather(
            k_pe,
            total_tokens=total_tokens,
            input_scattered=input_scattered,
        )
        positions = self._maybe_all_gather(
            positions,
            total_tokens=total_tokens,
            input_scattered=input_scattered,
        )
        q_scale = self._maybe_all_gather(
            q_scale,
            total_tokens=total_tokens,
            input_scattered=input_scattered,
        )
        return q_input, kv_c_normed, k_pe, positions, q_scale

    def _project_q(
        self,
        q_input: torch.Tensor,
        q_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        attn = self.owner_attn
        from atom.plugin.sglang.models.deepseek_mla_forward import (
            _q_b_proj_with_optional_scale,
            _unwrap_linear_output,
        )

        if attn.q_lora_rank is not None:
            q = _q_b_proj_with_optional_scale(attn, q_input, q_scale)
        else:
            q = (
                attn.q_proj(q_input, q_scale)
                if q_scale is not None
                else attn.q_proj(q_input)
            )
        return _unwrap_linear_output(q).view(-1, attn.num_local_heads, attn.qk_head_dim)

    def _make_dummy_output(self, q_input: torch.Tensor) -> torch.Tensor:
        attn = self.owner_attn
        return torch.empty(
            (q_input.shape[0], attn.hidden_size),
            device=q_input.device,
            dtype=torch.bfloat16,
        )

    def _forward_absorbed(
        self,
        q_input: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
        q_scale: torch.Tensor | None,
        *,
        forward_batch,
    ) -> torch.Tensor:
        attn = self.owner_attn
        from aiter import dtypes
        from atom.model_ops.attention_mla import fused_qk_rope_concat_and_cache_mla
        from atom.plugin.sglang.models.deepseek_mla_forward import (
            _get_sglang_radix_attn,
            mla_absorbed_bmm,
            mla_v_up_proj,
        )
        from sglang.srt.layers.attention.nsa.utils import nsa_use_prefill_cp

        q = self._project_q(q_input, q_scale)
        k_nope = kv_c_normed.unsqueeze(1)
        k_pe = k_pe.unsqueeze(1)
        q_nope, q_pe = q.split([attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)
        q_nope_out = mla_absorbed_bmm(
            attn, q_nope, attn.w_kc, attn.w_scale, attn.w_scale_k, attn.kv_lora_rank
        )

        if (
            attn.rotary_emb is not None
            and not attn.use_fused_qk_rope_concat_and_cache_mla
        ):
            q_pe, k_pe = attn.rotary_emb(positions, q_pe, k_pe)

        if nsa_use_prefill_cp(forward_batch):
            latent_cache = torch.cat([k_nope.squeeze(1), k_pe.squeeze(1)], dim=-1)
            k_nope, k_pe = attn.rebuild_cp_kv_cache(
                latent_cache, forward_batch, k_nope, k_pe
            )

        save_kv_cache = True
        topk_indices = None
        q_descale = None
        if (
            getattr(attn, "use_nsa", False)
            and getattr(attn, "indexer", None) is not None
        ):
            topk_indices = attn.indexer.topk_indices_buffer[: q_input.shape[0]]
        if attn.use_fused_qk_rope_concat_and_cache_mla:
            mla_attn = _get_sglang_radix_attn(self.base_attn)
            kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(mla_attn.layer_id)
            q_cache_scale = getattr(mla_attn, "q_scale", None)
            if q_cache_scale is None:
                q_cache_scale = mla_attn.k_scale
            q_out_dtype = (
                dtypes.fp8 if attn.kv_cache_dtype == "fp8_e4m3" else q_nope_out.dtype
            )
            q_descale = q_cache_scale if attn.kv_cache_dtype == "fp8_e4m3" else None
            q = torch.empty(
                (
                    q_nope_out.shape[0],
                    attn.num_local_heads,
                    attn.kv_lora_rank + attn.qk_rope_head_dim,
                ),
                dtype=q_out_dtype,
                device=q_nope_out.device,
            )
            fused_qk_rope_concat_and_cache_mla(
                q_nope_out,
                q_pe,
                k_nope,
                k_pe,
                kv_cache,
                q,
                forward_batch.out_cache_loc,
                mla_attn.k_scale,
                q_cache_scale,
                positions,
                attn.rotary_emb.cos_cache,
                attn.rotary_emb.sin_cache,
                is_neox=attn.rotary_emb.is_neox_style,
                is_nope_first=True,
            )
            k = None
            v = None
            save_kv_cache = False
        else:
            q = torch.cat([q_nope_out, q_pe], dim=-1)
            k = torch.cat([k_nope, k_pe], dim=-1)
            v = k_nope

        extra_kwargs: dict[str, Any] = {}
        if topk_indices is not None:
            extra_kwargs["topk_indices"] = topk_indices
        attn_output = self.base_attn(
            q,
            k,
            v,
            forward_batch=forward_batch,
            save_kv_cache=save_kv_cache,
            q_scale=q_descale,
            **extra_kwargs,
        )
        attn_output = attn_output.view(-1, attn.num_local_heads, attn.kv_lora_rank)
        attn_bmm_output = mla_v_up_proj(
            attn, attn_output, attn.w_vc, attn.w_scale, attn.w_scale_v, attn.v_head_dim
        )
        return attn.o_proj(attn_bmm_output)

    def _forward_non_absorbed(
        self,
        q_input: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
        q_scale: torch.Tensor | None,
        *,
        forward_batch,
    ) -> torch.Tensor:
        attn = self.owner_attn
        from atom.plugin.sglang.models.deepseek_mla_forward import (
            _concat_mha_k_for_non_absorbed,
            _set_mla_kv_buffer_for_non_absorbed,
            _unwrap_linear_output,
            try_fused_mxfp4_kv_b_proj_fp8,
            use_sglang_fp8_prefill_attn,
        )

        q = self._project_q(q_input, q_scale)
        _, q_pe = q.split([attn.qk_nope_head_dim, attn.qk_rope_head_dim], dim=-1)

        kv_a = kv_c_normed
        k_pe = k_pe.unsqueeze(1)
        if attn.rotary_emb is not None:
            q_pe, k_pe = attn.rotary_emb(positions, q_pe, k_pe)
        q[..., attn.qk_nope_head_dim :] = q_pe

        _set_mla_kv_buffer_for_non_absorbed(attn, kv_a, k_pe, forward_batch)

        extend_prefix_lens_cpu = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        extend_no_prefix = (
            False if extend_prefix_lens_cpu is None else not any(extend_prefix_lens_cpu)
        )
        fused_kv = None
        if (
            use_sglang_fp8_prefill_attn()
            and forward_batch.forward_mode.is_extend_without_speculative()
            and extend_no_prefix
        ):
            fused_kv = try_fused_mxfp4_kv_b_proj_fp8(
                attn.kv_b_proj,
                kv_a,
                k_pe,
                num_heads=attn.num_local_heads,
                qk_nope_head_dim=attn.qk_nope_head_dim,
                v_head_dim=attn.v_head_dim,
            )

        if fused_kv is not None:
            k, v = fused_kv
        else:
            kv = _unwrap_linear_output(attn.kv_b_proj(kv_a)).view(
                -1, attn.num_local_heads, attn.qk_nope_head_dim + attn.v_head_dim
            )
            k_nope = kv[..., : attn.qk_nope_head_dim]
            v = kv[..., attn.qk_nope_head_dim :]
            k = _concat_mha_k_for_non_absorbed(attn, k_nope, k_pe)

        attn_output = attn.attn_non_absorbed(
            q,
            k,
            v,
            forward_batch=forward_batch,
            save_kv_cache=False,
        )
        attn_output = attn_output.reshape(-1, attn.num_local_heads * attn.v_head_dim)
        return attn.o_proj(attn_output)

    def forward(
        self,
        q_input: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        positions: torch.Tensor,
        q_scale: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        attn = self.owner_attn
        forward_batch = self._get_forward_batch(kwargs)

        from atom.plugin.sglang.models.deepseek_mla_forward import (
            _can_run_non_absorbed_mla_now,
        )
        from sglang.srt.layers.communicator import get_attn_tp_context

        attn_tp_context = get_attn_tp_context()
        with attn_tp_context.maybe_input_scattered(forward_batch):
            q_input, kv_c_normed, k_pe, positions, q_scale = (
                self._gather_runtime_inputs(
                    q_input,
                    kv_c_normed,
                    k_pe,
                    positions,
                    q_scale,
                    forward_batch=forward_batch,
                    input_scattered=attn_tp_context.input_scattered,
                )
            )

            from atom.utils.forward_context import get_forward_context

            if get_forward_context().context.is_dummy_run:
                return self._make_dummy_output(q_input)

            use_non_absorbed = (
                forward_batch.forward_mode.is_extend_without_speculative()
            )
            if not use_non_absorbed and forward_batch.forward_mode.is_draft_extend():
                extend_prefix_lens_cpu = getattr(
                    forward_batch, "extend_prefix_lens_cpu", None
                )
                use_non_absorbed = extend_prefix_lens_cpu is not None and not any(
                    extend_prefix_lens_cpu
                )

            if use_non_absorbed:
                if _can_run_non_absorbed_mla_now(attn, forward_batch):
                    attn.current_sgl_plugin_attn_path = "non_absorbed"
                    return self._forward_non_absorbed(
                        q_input,
                        kv_c_normed,
                        k_pe,
                        positions,
                        q_scale,
                        forward_batch=forward_batch,
                    )
                attn.current_sgl_plugin_attn_path = "absorbed_fallback"
            else:
                attn.current_sgl_plugin_attn_path = "absorbed"

            return self._forward_absorbed(
                q_input,
                kv_c_normed,
                k_pe,
                positions,
                q_scale,
                forward_batch=forward_batch,
            )
