# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""GEMM-fusion runtime (default OFF; see ``GemmFusionConfig.profile``).

Importing this package is cheap and does not touch aiter or register ops; the
custom op lives in ``ops`` and is imported only by a model that adopts a site.
"""

from .hooks import (  # noqa: F401
    GemmFusionDeviceError,
    GemmFusionRecoveryTimeout,
    after_graph_capture,
    before_graph_capture,
    bind_kv_cache,
    check_launch_gate,
    check_now,
    check_step,
    init_gemm_fusion,
    profile_enabled,
    shutdown,
)
from .site import (  # noqa: F401
    GemmFusionBindError,
    GemmFusionContractError,
    GemmFusionSite,
)


def declare_m3_qkv_site(attn_module, *, prefix: str, sparse: bool):
    """Declare the QKV GEMM + SP exchange + norm/rope/cache site for a
    MiniMax-M3 attention module (``MiniMaxM3Attention`` / ``...SparseAttention``).

    Returns ``None`` (and the model keeps its existing forward) unless the
    feature is ON *and* this layer can use the fp8-output SP path the op
    reproduces. Sets ``attn_module.attn.qkv_site`` (a plain attribute, not a
    submodule, so the linear is not duplicated in the state dict).
    """
    from . import stages as S

    if not profile_enabled():
        return None
    # registers torch.ops.aiter.gemm_fusion_qkv_attention
    from . import ops  # noqa: F401

    attn = attn_module.attn
    if not getattr(attn, "_m3_sp_fp8_output_supported", False) or getattr(
        attn, "_m3_tp_replicated_o_proj", False
    ):
        return None
    # Same gate as _minimax_m3_attend_and_project's FP8 branch (q width 8192 and
    # bf16 are already folded into _m3_sp_fp8_output_supported).
    from atom.models.minimax_m3.model import _linear_consumes_per_token_fp8

    o_proj = attn_module.o_proj
    if o_proj.input_size != 8192 or not _linear_consumes_per_token_fp8(o_proj):
        return None
    skip = bool(getattr(attn_module, "skip_index_topk", False))
    head_dim = attn_module.head_dim
    kwargs = {"q": attn_module.q_size, "kv": attn_module.kv_size, "head_dim": head_dim}
    if sparse:
        kwargs.update(
            iq=attn_module.index_q_size,
            ik=attn_module.idx_head_dim,
            idx_head_dim=attn_module.idx_head_dim,
        )
    fields = S.qkv_fields(**kwargs)
    stages = [
        S.sp_exchange(fields, drop=("iq", "ik") if (sparse and skip) else ()),
        S.head_norm_rope_cache_from_attention(
            attn,
            index=S.index_cache_signature(attn.impl) if (sparse and not skip) else None,
        ),
    ]
    site = GemmFusionSite(
        f"{prefix}.qkv",
        attn_module.qkv_proj,
        stages,
        S.native_fns(stages, attn),
        attn=attn,
        q_size=attn_module.q_size,
        kv_size=attn_module.kv_size,
        index_q_size=getattr(attn_module, "index_q_size", 0),
        index_k_size=getattr(attn_module, "idx_head_dim", 0) if sparse else 0,
        sparse=sparse,
        skip_index=skip,
    )
    attn.qkv_site = site
    return site
