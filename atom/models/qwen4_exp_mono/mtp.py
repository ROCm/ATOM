# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MTP target attention chain, with explicit PTPC and ReplaySSM seams."""

import os
from dataclasses import dataclass

import torch
from aiter import dtypes
from aiter.tuned_gemm import tgemm

from atom.model_ops.qwen4_exp.ops.gated import is_per_token_fp8_linear
from atom.model_ops.qwen4_exp.ops.hc_fused import hc_combine_norm
from atom.models.qwen4_exp_mono import native_qsa
from atom.models.qwen4_exp_mono.dims import Dims
from atom.models.qwen4_exp_mono.kernels import mtp_gdn, mtp_hc_ptpc
from atom.models.qwen4_exp_mono.plan import MtpPlan
from atom.models.qwen4_exp_mono.replayssm import replayssm_gated_delta_rule
from atom.utils.forward_context import get_forward_context

_GDN_PROJECTION_CONV = os.getenv("QWEN_GDN_PROJECTION_CONV", "0") == "1"
_GDN_FINISH = os.getenv("QWEN_GDN_FINISH", "0") == "1"


@dataclass(frozen=True)
class Workspace:
    plan: MtpPlan
    hc: object
    mixed: torch.Tensor
    quantized: torch.Tensor
    scale: torch.Tensor
    raw: torch.Tensor
    payload: torch.Tensor
    gdn: tuple[torch.Tensor, ...]
    core: torch.Tensor
    ba_context: object = None
    projection_conv: object = None

    @classmethod
    def allocate(cls, rows, device):
        dims = Dims(1)
        from atom.models.qwen4_exp_mono.ba.binding import BaProjection

        ba_context = (
            BaProjection(device)
            if rows == 4 and os.environ.get("QWEN_HC_BA") == "1"
            else None
        )
        projection_conv = None
        if rows == 4:
            import json

            from atom.models.qwen4_exp_mono.gdn_projection.binding import ProjectionConv

            projection_conv = ProjectionConv(device)
            print(
                "QWEN_PROJECTION_CONV_BOUND "
                + json.dumps(
                    {
                        "rows": rows,
                        "enabled": _GDN_PROJECTION_CONV,
                        "resources": projection_conv.resources,
                    }
                ),
                flush=True,
            )
        bf = lambda *shape: torch.empty(shape, device=device, dtype=torch.bfloat16)
        return cls(
            MtpPlan(rows),
            mtp_hc_ptpc.build(rows),
            bf(rows, 2560),
            torch.empty((rows, 2560), device=device, dtype=dtypes.fp8),
            torch.empty((rows, 1), device=device, dtype=torch.float32),
            bf(rows, 4),
            torch.zeros(rows * 2560, device=device, dtype=torch.int32),
            (
                bf(1, rows, dims.gdn_key_heads, 128),
                bf(1, rows, dims.gdn_key_heads, 128),
                bf(1, rows, dims.gdn_value_heads, 128),
                torch.empty(
                    (1, rows, dims.gdn_value_heads), device=device, dtype=torch.float32
                ),
                bf(1, rows, dims.gdn_value_heads),
            ),
            torch.zeros(
                (rows, dims.gdn_value_heads, 128), device=device, dtype=torch.bfloat16
            ),
            ba_context,
            projection_conv,
        )

    def mix_attention(self, hc, streams, pending, ba_weight=None):
        block, previous = pending if pending is not None else (None, None)
        streams, normed = hc_combine_norm(
            streams, hc.hc_norm.weight, hc.hc_norm.variance_epsilon, 4, block, previous
        )
        part = tgemm.mm(normed, hc.fused_w_cat, otype=torch.bfloat16)
        values = {
            "part": part.data_ptr(),
            "normed": normed.data_ptr(),
            "weight": hc.input_mix_weight_up.weight.data_ptr(),
            "raw": self.raw.data_ptr(),
            "stride": normed.stride(0),
            "mixed": self.mixed.data_ptr(),
            "q": self.quantized.data_ptr(),
            "scale": self.scale.data_ptr(),
            "payload": self.payload.data_ptr(),
        }
        if self.ba_context is not None and ba_weight is not None:
            self.ba_context.run(values, ba_weight)
        else:
            self.hc(*mtp_hc_ptpc.ABI.pack(values), stream=torch.cuda.current_stream())
        return self.mixed, (self.quantized, self.scale), (streams, self.raw)

    def attention(self, layer, task, positions, mixed, quantized, ba_prepared=False):
        if not task.is_gdn:
            prepared = native_qsa.prepare(
                layer.self_attn,
                positions,
                mixed,
                quantized=(
                    quantized
                    if is_per_token_fp8_linear(layer.self_attn.qkv_proj)
                    else None
                ),
                fuse_decode_index=True,
                index_query_length=self.plan.query_length,
            )
            return native_qsa.finish(layer.self_attn, prepared)
        module = layer.linear_attn
        # Eligibility is settled before any convolution or Replay state write.
        if (
            _GDN_PROJECTION_CONV
            and self.projection_conv is not None
            and module.quantized_inputs
        ):
            context = get_forward_context()
            metadata = context.attn_metadata.gdn_metadata
            impl = module.attn.impl
            cache = context.kv_cache_data[f"layer_{impl.layer_num}"]
            if (
                metadata.num_actual_tokens == 4
                and metadata.num_spec_decodes == 1
                and cache.replay_buf_k.shape[2] == 16
                and metadata.replayssm_route in ("auto", "serial")
                and is_per_token_fp8_linear(module.in_proj_qkvz)
                and is_per_token_fp8_linear(module.out_proj)
            ):
                b, a = (
                    self.ba_context.output if ba_prepared else module.in_proj_ba(mixed)
                ).chunk(2, dim=-1)
                z = self.projection_conv(
                    quantized[0],
                    quantized[1],
                    {
                        "weight": module.in_proj_qkvz.weight,
                        "weight_scale": module.in_proj_qkvz.weight_scale,
                    },
                    impl,
                    a,
                    b,
                    cache.k_cache,
                    metadata,
                    self.gdn,
                )
                q, k, v, g, beta = self.gdn
                from atom.models.qwen4_exp_mono.gdn_finish import (
                    forward as gdn_finish_forward,
                )

                return gdn_finish_forward(
                    module,
                    z,
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    ckpt=cache.v_cache,
                    buf_k=cache.replay_buf_k,
                    buf_u=cache.replay_buf_u,
                    buf_g=cache.replay_buf_g,
                    write_pos=metadata.write_pos,
                    slot_idx=metadata.slot_idx[:1],
                    cu_seqlens=metadata.spec_query_start_loc[:2],
                    max_query_len=4,
                    use_qk_l2norm_in_kernel=True,
                    is_kda=False,
                    route=metadata.replayssm_route,
                    o=self.core,
                )
        if module.quantized_inputs:
            projected = module.in_proj_qkvz(quantized[0], x_scale=quantized[1])
            mixed_qkv, z = projected.split([module.conv_dim, module.value_dim], dim=-1)
            b, a = (
                self.ba_context.output if ba_prepared else module.in_proj_ba(mixed)
            ).chunk(2, dim=-1)
        else:
            mixed_qkv, z, b, a = module.in_proj_qkvzba(mixed).split(
                [
                    module.conv_dim,
                    module.value_dim,
                    module.num_v_heads,
                    module.num_v_heads,
                ],
                dim=-1,
            )
        context = get_forward_context()
        metadata = context.attn_metadata.gdn_metadata
        impl = module.attn.impl
        cache = context.kv_cache_data[f"layer_{impl.layer_num}"]
        q, k, v, g, beta = mtp_gdn.prepare(
            impl, mixed_qkv, a, b, cache, metadata, self.gdn
        )
        n = metadata.num_actual_tokens
        requests = metadata.num_spec_decodes
        if (
            _GDN_FINISH
            and self.plan.rows == 4
            and n == 4
            and requests == 1
            and cache.replay_buf_k.shape[2] == 16
            and is_per_token_fp8_linear(module.out_proj)
        ):
            from atom.models.qwen4_exp_mono.gdn_finish import (
                forward as gdn_finish_forward,
            )

            return gdn_finish_forward(
                module,
                z.view_as(self.core),
                q=q[:, :n],
                k=k[:, :n],
                v=v[:, :n],
                g=g[:, :n],
                beta=beta[:, :n],
                ckpt=cache.v_cache,
                buf_k=cache.replay_buf_k,
                buf_u=cache.replay_buf_u,
                buf_g=cache.replay_buf_g,
                write_pos=metadata.write_pos,
                slot_idx=metadata.slot_idx[:requests],
                cu_seqlens=metadata.spec_query_start_loc[: requests + 1],
                max_query_len=self.plan.query_length,
                use_qk_l2norm_in_kernel=True,
                is_kda=False,
                route=metadata.replayssm_route,
                o=self.core[:n],
            )
        replayssm_gated_delta_rule(
            q=q[:, :n],
            k=k[:, :n],
            v=v[:, :n],
            g=g[:, :n],
            beta=beta[:, :n],
            ckpt=cache.v_cache,
            buf_k=cache.replay_buf_k,
            buf_u=cache.replay_buf_u,
            buf_g=cache.replay_buf_g,
            write_pos=metadata.write_pos,
            slot_idx=metadata.slot_idx[:requests],
            cu_seqlens=metadata.spec_query_start_loc[: requests + 1],
            max_query_len=self.plan.query_length,
            use_qk_l2norm_in_kernel=True,
            is_kda=False,
            route=metadata.replayssm_route,
            o=self.core[:n],
        )
        z = z.view_as(self.core)
        # Native already fuses sigmoid/norm/quant, with two BF16 roundings and
        # a per-token FP32 scale. Reuse that device stage at the GEMM seam.
        if is_per_token_fp8_linear(module.out_proj):
            x, scale = module.norm.forward_quant(
                self.core, z, module.out_proj.params_dtype
            )
            return module.out_proj(x, x_scale=scale)
        return module.out_proj(module.norm(self.core, z))
