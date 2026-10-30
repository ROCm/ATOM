import torch
import triton
import triton.language as tl

from atom.model_ops.fla_ops.replayssm import (
    UT_MIN_QUERY_LEN,
    _replay_dot_mode,
    _replayssm_ut_fwd_kernel,
    _serial_launch_config,
    flush_threshold_ok,
)
from atom.models.qwen4_exp_mono.replayssm import _replayssm_fwd_kernel


@triton.jit(do_not_specialize=["T_MAX", "CAP"])
def publish_device(
    buf_k,
    buf_g,
    write_pos,
    slot_idx,
    cu_seqlens,
    T_MAX,
    CAP,
    K: tl.constexpr,
    BK: tl.constexpr,
    BT: tl.constexpr,
    stride_bufk_slot: tl.constexpr,
    stride_bufk_pos: tl.constexpr,
    stride_bufk_hv: tl.constexpr,
    stride_bufg_slot: tl.constexpr,
    stride_bufg_pos: tl.constexpr,
    stride_bufg_hv: tl.constexpr,
    IS_KDA: tl.constexpr,
    i_n,
    i_hv,
):
    T = tl.load(cu_seqlens + i_n + 1) - tl.load(cu_seqlens + i_n)
    slot = tl.load(slot_idx + i_n).to(tl.int64)
    if T == 0 or slot < 0:
        return
    h = tl.maximum(tl.load(write_pos + slot), 0).to(tl.int64)
    if h + 2 * T_MAX <= CAP:
        return
    o_t = tl.arange(0, BT)
    o_k = tl.arange(0, BK)
    mask = (o_t[:, None] < T) & (o_k[None, :] < K)
    p_k = (
        buf_k
        + slot * stride_bufk_slot
        + i_hv * stride_bufk_hv
        + o_t[:, None] * stride_bufk_pos
        + o_k[None, :]
    )
    b_k = tl.load(p_k + h * stride_bufk_pos, mask=mask, other=0)
    p_g = buf_g + slot * stride_bufg_slot + i_hv * stride_bufg_hv
    if IS_KDA:
        p_g = p_g + o_t[:, None] * stride_bufg_pos + o_k[None, :]
        b_g = tl.load(p_g + h * stride_bufg_pos, mask=mask, other=0)
    else:
        p_g = p_g + o_t * stride_bufg_pos
        b_g = tl.load(p_g + h * stride_bufg_pos, mask=o_t < T, other=0)
    tl.debug_barrier()
    tl.store(p_k, b_k, mask=mask)
    if IS_KDA:
        tl.store(p_g, b_g, mask=mask)
    else:
        tl.store(p_g, b_g, mask=o_t < T)


def replay_unpublished(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ckpt: torch.Tensor,
    buf_k: torch.Tensor,
    buf_u: torch.Tensor,
    buf_g: torch.Tensor,
    write_pos: torch.Tensor,
    slot_idx: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_query_len: int,
    scale: float | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
    route: str = "auto",
    o: torch.Tensor | None = None,
) -> torch.Tensor:
    """ReplaySSM forward for one linear-attention layer.

    Args:
        q, k: ``[1, T_tot, H, K]``
        v: ``[1, T_tot, HV, V]``
        g: ``[1, T_tot, HV]`` (GDN) or ``[1, T_tot, HV, K]`` (KDA)
        beta: ``[1, T_tot, HV]`` or ``[1, T_tot, HV, V]``
        ckpt: ``[num_slots, HV, K, V]`` checkpoint pool (mutated on flush)
        buf_k / buf_u / buf_g: record buffers, see ``replayssm_buffer_shapes``
        write_pos: ``[num_slots]`` int32 committed-record cursor; advance with
            :func:`replayssm_commit` once per forward, never here
        slot_idx: ``[N]`` int32, ``PAD_SLOT_ID`` for idle entries
        max_query_len: the ``T`` used by the flush predicate; must match the
            value passed to :func:`replayssm_commit` in the same step
        route: ``"serial"`` walks tokens one at a time; ``"ut"`` uses the
            chunked delta-rule UT transform (scalar gate, non-headwise beta,
            multi-token steps only); ``"auto"`` picks ``"ut"`` when eligible.
        o: optional contiguous ``[T_tot, HV, V]`` destination (e.g. a slice of
            the layer's output buffer); the kernel writes into it directly
            instead of a fresh tensor, saving the caller a device copy.

    Returns:
        ``o`` of shape ``[1, T_tot, HV, V]`` -- same leading-batch convention
        as :func:`fused_recurrent_gated_delta_rule`, so this is a drop-in
        replacement at the call sites in ``attention_gdn.py``.
    """
    assert max_query_len == 4 and (not is_kda) and (route in ("auto", "serial"))
    assert q.shape[0] == 1, "varlen layout expected (B == 1)"
    _, T_tot, H, K = q.shape
    HV, V = (v.shape[2], v.shape[3])
    N = cu_seqlens.numel() - 1
    cap = buf_k.shape[2]
    if scale is None:
        scale = K ** (-0.5)
    assert flush_threshold_ok(
        cap, max_query_len
    ), f"replayssm cache_len={cap} must be >= 2*max_query_len={2 * max_query_len}; otherwise the record buffer overruns"
    BK = triton.next_power_of_2(K)
    assert triton.cdiv(K, BK) == 1, "K must fit one block"
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    if o is None:
        o = q.new_empty(1, T_tot, HV, V)
    else:
        assert o.shape == (T_tot, HV, V) and o.is_contiguous() and (o.dtype == q.dtype)
        o = o.unsqueeze(0)
    if route not in ("auto", "serial", "ut"):
        raise ValueError(f"unknown replayssm route {route!r}; expected auto|serial|ut")
    ut_eligible = not is_kda and beta.ndim != v.ndim and (max_query_len > 1)
    if route == "auto":
        route = "ut" if ut_eligible and max_query_len >= UT_MIN_QUERY_LEN else "serial"
    if route == "ut":
        if not ut_eligible:
            raise ValueError(
                "replayssm route='ut' requires a scalar gate (not KDA), non-headwise beta, and max_query_len > 1"
            )
        BV_UT = triton.next_power_of_2(V)
        BT = triton.next_power_of_2(max_query_len)
        BH_UT = max(16, triton.next_power_of_2(cap - max_query_len))
        _replayssm_ut_fwd_kernel[1, N * HV](
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            o=o,
            ckpt=ckpt,
            buf_k=buf_k,
            buf_u=buf_u,
            buf_g=buf_g,
            write_pos=write_pos,
            slot_idx=slot_idx,
            cu_seqlens=cu_seqlens,
            scale=scale,
            N=N,
            T_TOT=T_tot,
            T_MAX=max_query_len,
            CAP=cap,
            H=H,
            HV=HV,
            K=K,
            V=V,
            BK=BK,
            BV=BV_UT,
            BH=BH_UT,
            BT=BT,
            stride_ckpt_slot=ckpt.stride(0),
            stride_bufk_slot=buf_k.stride(0),
            stride_bufk_pos=buf_k.stride(2),
            stride_bufk_hv=buf_k.stride(1),
            stride_bufu_slot=buf_u.stride(0),
            stride_bufu_pos=buf_u.stride(2),
            stride_bufu_hv=buf_u.stride(1),
            stride_bufg_slot=buf_g.stride(0),
            stride_bufg_pos=buf_g.stride(2),
            stride_bufg_hv=buf_g.stride(1),
            USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
            DOT_MODE=_replay_dot_mode(buf_u.dtype),
            num_warps=4 if BT >= 16 else 2,
            num_stages=2,
        )
        return o
    BV, num_warps, num_stages = _serial_launch_config(N, HV, V, max_query_len, is_kda)
    NV = triton.cdiv(V, BV)
    BH = max(16, triton.next_power_of_2(cap - max_query_len))
    _replayssm_fwd_kernel[NV, N * HV](
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        o=o,
        ckpt=ckpt,
        buf_k=buf_k,
        buf_u=buf_u,
        buf_g=buf_g,
        write_pos=write_pos,
        slot_idx=slot_idx,
        cu_seqlens=cu_seqlens,
        scale=scale,
        N=N,
        T_TOT=T_tot,
        T_MAX=max_query_len,
        CAP=cap,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        BH=BH,
        stride_ckpt_slot=ckpt.stride(0),
        stride_bufk_slot=buf_k.stride(0),
        stride_bufk_pos=buf_k.stride(2),
        stride_bufk_hv=buf_k.stride(1),
        stride_bufu_slot=buf_u.stride(0),
        stride_bufu_pos=buf_u.stride(2),
        stride_bufu_hv=buf_u.stride(1),
        stride_bufg_slot=buf_g.stride(0),
        stride_bufg_pos=buf_g.stride(2),
        stride_bufg_hv=buf_g.stride(1),
        stride_beta_tok=beta.stride(1),
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        IS_KDA=is_kda,
        IS_BETA_HEADWISE=beta.ndim == v.ndim,
        DOT_MODE=_replay_dot_mode(buf_u.dtype),
        T1_FAST=max_query_len == 1,
        T1_TILED=max_query_len == 1 and (not is_kda),
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return o
