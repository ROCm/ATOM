# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DeepSeek-V4 DSpark semi-autoregressive block drafter for ATOM.

DSpark (DeepSeek-AI, 2026) is a speculative-decoding draft model. It is stored
inside the V4 checkpoint under the same ``mtp.*`` namespace as serial MTP, but
it is a DIFFERENT architecture and is routed here, never to serial MTP.

Two mechanisms (paper §3):

1. Semi-Autoregressive Generation (§3.1)
   - A heavy PARALLEL backbone (``dspark_block_size`` DSpark layers = V4 decoder
     layers with mHC + sliding-window attention over a private rolling target-KV
     window) produces all base logits ``U_1..U_gamma`` in one forward pass.
   - A lightweight SEQUENTIAL Markov head injects intra-block token dependency
     via a low-rank transition bias ``B = W1 @ W2`` (rank ``dspark_markov_rank``),
     sampling left-to-right.  Final per-position distribution (paper Eq. 4/5):

         p_k(v | x_<k) = softmax_v( U_k(v) + B(x_{k-1}, v) )

     Because the bias is added inside a per-position softmax (local correction,
     not global normalization), per-token probabilities remain exact, which is
     required for lossless speculative verification.

2. Confidence head (§3.2.1)
   - A per-position scalar ``c_k = sigma(w^T [h_k ; W1[x_{k-1}]])`` estimating the
     conditional survival probability (token k accepted | prefix accepted),
     consumed by the (Phase-2) hardware-aware scheduler.

Phase 1 scope: lossless block draft generation with a STATIC verify length.
The confidence head is computed and exposed, but the confidence-scheduled
verification (STS calibration + hardware-aware prefix scheduler) is Phase 2.

Checkpoint layout (DeepSeek-V4-Pro-DSpark):
  mtp.{0,1,2}.*              3 DSpark backbone layers (attn + MoE + mHC)
  mtp.0.main_proj / main_norm   inject concat of target layers [58,59,60]
  mtp.2.markov_head.markov_w1/w2   Markov low-rank transition (rank 512)
  mtp.2.confidence_head.proj       confidence head [1, hidden+rank]
  mtp.2.hc_head_{fn,base,scale}, mtp.2.norm   final mHC reduction + norm
  (embed + lm_head are shared with the target via share_with_target)
"""

from typing import Optional

import torch
from torch import nn


class DSparkMarkovHead(nn.Module):
    """Low-rank first-order Markov transition bias (paper §3.1, Eq. 5).

    The full ``V x V`` transition matrix is factorized as ``B = W1 @ W2`` with
    ``W1 in R^{V x r}`` (embedding lookup of the previous token) and
    ``W2 in R^{V x r}`` (logit projection).  Given the previously sampled token
    ``x_{k-1}``, the bias added to position ``k``'s base logits is

        B(x_{k-1}, :) = W1[x_{k-1}] @ W2^T   in R^V

    ``r = dspark_markov_rank`` (512 for V4-Pro-DSpark) keeps both storage and
    per-step compute small, so the sequential sampling loop stays lightweight
    relative to the parallel backbone.

    Checkpoint shapes: markov_w1.weight [V, r], markov_w2.weight [V, r].
    Both are nn.Embedding-style [V, r] tables; the logit projection uses
    ``W1[x] @ W2.weight^T`` ( == @ W2 with W2 viewed as [r, V] ).
    """

    def __init__(self, vocab_size: int, rank: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.rank = rank
        # W1: per-token embedding lookup table [V, r].
        self.markov_w1 = nn.Embedding(vocab_size, rank)
        # W2: logit projection stored as [V, r] (matches checkpoint); applied as
        # embed @ W2.weight^T to produce a [*, V] bias. fp32 for precision parity
        # with the reference (the bias enters a softmax that gates acceptance).
        self.markov_w2 = nn.Embedding(vocab_size, rank)

    def forward(self, token_ids: torch.Tensor):
        """Compute the per-position transition bias and the Markov embedding.

        Args:
            token_ids: [*]  ids of the previously sampled token x_{k-1}.

        Returns:
            logits_bias: [*, V]  bias to add to base logits at the next position.
            markov_embed: [*, r]  W1[x_{k-1}], reused by the confidence head.
        """
        markov_embed = self.markov_w1(token_ids)  # [*, r]
        # bias = W1[x] @ W2^T : [*, r] x [r, V] -> [*, V]. fp32 matmul.
        logits_bias = torch.matmul(
            markov_embed.float(), self.markov_w2.weight.float().t()
        )
        return logits_bias, markov_embed


class DSparkConfidenceHead(nn.Module):
    """Per-position survival-probability estimator (paper §3.2.1, Eq. 7).

        c_k = sigma( w^T [ h_k ; W1[x_{k-1}] ] )

    Input is the concatenation of the backbone hidden state ``h_k`` (dim) and the
    Markov embedding ``W1[x_{k-1}]`` (rank), so the projection weight has shape
    ``[1, hidden + rank]`` (checkpoint: confidence_head.proj.weight [1, 7680]).

    The raw sigmoid output is the per-position conditional acceptance estimate.
    Phase 2 applies Sequential Temperature Scaling (STS) on the cumulative
    product before feeding the hardware-aware scheduler.
    """

    def __init__(self, hidden_size: int, rank: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size + rank, 1, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, markov_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Args:
            hidden_states: [*, hidden]  backbone hidden h_k.
            markov_embeds: [*, rank]    W1[x_{k-1}].
        Returns:
            confidence: [*]  sigmoid survival probability in (0, 1).
        """
        # Confidence is computed in fp32 (the checkpoint head is fp32 and the
        # downstream scheduler needs calibrated absolute probabilities).
        x = torch.cat([hidden_states, markov_embeds], dim=-1).float()
        logit = torch.nn.functional.linear(x, self.proj.weight.float()).squeeze(-1)
        return torch.sigmoid(logit)


# ---------------------------------------------------------------------------
# Numerical helpers (mirror the public DSpark HF reference; see vLLM PR #46965).
# These are deliberately plain-torch so they run on ROCm without new kernels.
# GPU-VERIFY: on-device, the dense fmha below can be swapped for
# aiter.fmha_fwd_with_sink_asm (q block=5, kv=window128+block) for speed.
# ---------------------------------------------------------------------------


def _linear_out(output):
    """ATOM quantized linears may return (tensor, scale); take the tensor."""
    return output[0] if isinstance(output, tuple) else output


def _count_dspark_stages(model_path, default: int = 0) -> int:
    """Count distinct ``mtp.{i}.*`` stages in the checkpoint index.

    DSpark stores its backbone as ``mtp.0 .. mtp.{N-1}`` in the V4 checkpoint
    (N=3 for V4-Pro-DSpark). We must build exactly N stages or the last stage's
    Markov/confidence-head weights get dropped at load. The HF config's
    ``num_nextn_predict_layers`` is unrelated (it is 1, a serial-MTP field).
    """
    import json
    import os
    import re

    if not model_path:
        return default
    idx_path = os.path.join(model_path, "model.safetensors.index.json")
    try:
        with open(idx_path) as f:
            weight_map = json.load(f)["weight_map"]
    except Exception:
        return default
    stages = set()
    for name in weight_map:
        m = re.match(r"^mtp\.(\d+)\.", name)
        if m:
            stages.add(int(m.group(1)))
    return (max(stages) + 1) if stages else default


def _apply_dspark_rope_hf_torch(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """Plain-torch reference for DSpark HF-style (GPT-J interleaved) RoPE.

    Kept as a kernel-free, inspectable reference. The production path
    (``_apply_dspark_rope_hf``) dispatches to a fused Triton kernel on GPU.
    """
    if rope_dim <= 0:
        return x
    out = x.clone()
    rope = x[..., -rope_dim:].float()
    cos_sin = cos_sin_cache.index_select(0, positions.reshape(-1)).float()
    cos, sin = cos_sin.chunk(2, dim=-1)
    # Broadcast cos/sin over an optional head axis.
    if rope.dim() == 3:
        cos = cos[:, None, :]
        sin = sin[:, None, :]
    rope_pairs = rope.view(*rope.shape[:-1], -1, 2)
    even = rope_pairs[..., 0]
    odd = rope_pairs[..., 1]
    rotated = torch.stack(
        (even * cos - odd * sin, odd * cos + even * sin), dim=-1
    ).flatten(-2)
    out[..., -rope_dim:] = rotated.to(out.dtype)
    return out


_FORWARD_ROPE_KERNEL = None


def _forward_rope_gptj_kernel_impl():
    """Lazily build (and cache) the forward GPT-J RoPE Triton kernel.

    Mirrors ``atom/model_ops/v4_kernels/inverse_rope.py`` but applies the FORWARD
    rotation. Same ``reuse_freqs_front_part`` cache layout: cos/sin hold rd//2
    entries, each shared by an adjacent pair (index = d // 2). fp32 math for
    parity with the torch reference; result cast back to the input dtype.
    """
    global _FORWARD_ROPE_KERNEL
    if _FORWARD_ROPE_KERNEL is not None:
        return _FORWARD_ROPE_KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _kernel(
        x_ptr, cos_ptr, sin_ptr, pos_ptr,
        stride_x_s, stride_x_h, stride_x_d,
        stride_cos_s, stride_cos_d,
        S,
        BLOCK_S: tl.constexpr, BLOCK_RD: tl.constexpr, BLOCK_RD_HALF: tl.constexpr,
    ):
        pid_h = tl.program_id(0)
        pid_s = tl.program_id(1)
        s_offs = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
        d_offs = tl.arange(0, BLOCK_RD)
        s_mask = s_offs < S
        pos = tl.load(pos_ptr + s_offs, mask=s_mask)
        # GPT-J reuse_freqs_front_part: cos/sin index = d // 2 (pair-shared).
        d_cos_offs = d_offs // 2
        cos_offs = pos[:, None] * stride_cos_s + d_cos_offs[None, :] * stride_cos_d
        cos_mask = s_mask[:, None] & (d_cos_offs < BLOCK_RD_HALF)[None, :]
        cos = tl.load(cos_ptr + cos_offs, mask=cos_mask).to(tl.float32)
        sin = tl.load(sin_ptr + cos_offs, mask=cos_mask).to(tl.float32)
        x_offs = (
            s_offs[:, None] * stride_x_s
            + pid_h * stride_x_h
            + d_offs[None, :] * stride_x_d
        )
        x_mask = s_mask[:, None] & (d_offs < BLOCK_RD)[None, :]
        x = tl.load(x_ptr + x_offs, mask=x_mask).to(tl.float32)
        # Forward GPT-J: out[2i]   =  x[2i]*cos - x[2i+1]*sin
        #                out[2i+1] =  x[2i]*sin + x[2i+1]*cos
        # Negate ODDs of (x*sin), then flip pairs (inverse kernel negates evens).
        x_sin = x * sin
        odd_mask = (d_offs % 2 == 1)[None, :]
        x_negated = tl.where(odd_mask, -x_sin, x_sin)
        x_negated = tl.reshape(x_negated, (BLOCK_S, BLOCK_RD_HALF, 2))
        x_negated = tl.flip(x_negated, 2)
        x_rotated = tl.reshape(x_negated, (BLOCK_S, BLOCK_RD))
        out = x * cos + x_rotated
        out = out.to(x_ptr.dtype.element_ty)
        tl.store(x_ptr + x_offs, out, mask=x_mask)

    _FORWARD_ROPE_KERNEL = _kernel
    return _kernel


def _apply_dspark_rope_hf(
    x: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rope_dim: int,
) -> torch.Tensor:
    """Apply DSpark HF-style (GPT-J interleaved) RoPE to the trailing rope lanes.

    Args:
        x: [N, H, head_dim] or [N, head_dim]; rope acts on last ``rope_dim`` dims.
        positions: [N] int absolute positions.
        cos_sin_cache: [max_pos, rope_dim] concatenated cos||sin.
        rope_dim: number of trailing rope dimensions.

    Dispatches to a fused Triton kernel on GPU (forward analog of the existing
    ``inverse_rope`` kernel). Set ``ATOM_DSPARK_ROPE_TORCH=1`` to force the
    plain-torch reference above.
    """
    if rope_dim <= 0:
        return x
    import os

    if os.environ.get("ATOM_DSPARK_ROPE_TORCH", "0") == "1" or not x.is_cuda:
        return _apply_dspark_rope_hf_torch(x, positions, cos_sin_cache, rope_dim)
    import triton

    kernel = _forward_rope_gptj_kernel_impl()
    out = x.clone()
    # Normalize to [S, H, rd] over the rope slice (add a head axis if absent).
    squeeze_head = out.dim() == 2
    rope_view = (out.unsqueeze(1) if squeeze_head else out)[..., -rope_dim:]
    S, H, rd = rope_view.shape
    # cos_sin_cache is [max_pos, rope_dim] = concat(cos[rd//2] || sin[rd//2]).
    cos = cos_sin_cache[:, : rope_dim // 2].contiguous()
    sin = cos_sin_cache[:, rope_dim // 2 :].contiguous()
    pos = positions.reshape(-1).to(torch.int32)
    BLOCK_RD = triton.next_power_of_2(rd)
    BLOCK_S = 32
    grid = (H, triton.cdiv(S, BLOCK_S))
    kernel[grid](
        rope_view,
        cos,
        sin,
        pos,
        rope_view.stride(0),
        rope_view.stride(1),
        rope_view.stride(2),
        cos.stride(0),
        cos.stride(1),
        S,
        BLOCK_S=BLOCK_S,
        BLOCK_RD=BLOCK_RD,
        BLOCK_RD_HALF=BLOCK_RD // 2,
        num_warps=4,
    )
    return out


def _fake_fp8_e4m3_inplace(x: torch.Tensor, block_size: int = 64) -> None:
    """In-place FP8 E4M3 fake-quant with power-of-two block scales (DSpark QAT).

    The HF DSpark module is QAT-trained: the non-RoPE KV lanes are quant/dequant
    through FP8 E4M3 at inference to match training numerics. Keeps the rolling
    KV cache in its native dtype (only the values pass through the round-trip).
    """
    if x.numel() == 0:
        return
    if x.shape[-1] % block_size != 0:
        raise ValueError(
            "DSpark fake-FP8 block size must divide the last dim: "
            f"{x.shape[-1]} % {block_size} != 0."
        )
    view = x.reshape(-1, x.shape[-1] // block_size, block_size)
    amax = view.abs().amax(dim=-1, keepdim=True).clamp_min(1.0e-4)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448.0)))
    quant = torch.clamp(view / scale, -448.0, 448.0).to(torch.float8_e4m3fn)
    view.copy_(quant.to(view.dtype) * scale.to(view.dtype))


def _apply_dspark_kv_qat_(kv: torch.Tensor, rope_dim: int) -> None:
    non_rope = kv[..., :-rope_dim] if rope_dim > 0 else kv
    _fake_fp8_e4m3_inplace(non_rope, block_size=64)


def _dspark_block_topk_idxs(
    B: int, T: int, W: int, valid_target: torch.Tensor, device
) -> torch.Tensor:
    """Encode the (window-validity + block-causal) attention mask as gather
    indices into the combined ``[window ++ draft]`` KV (length ``W+T``).

    For draft query position ``m`` (0..T-1) the attended columns are:
      * every VALID rolling-window slot  -> global index ``w`` (0..W-1)
      * draft-block slots ``0..m``       -> global index ``W + j``  (block-causal)
    All other entries are ``-1`` (the fused sparse_attn kernel skips them).

    Returns: topk_idxs [B, T, W+T] int32, suitable for ``sparse_attn``.
    """
    # Window columns: keep the global slot index where valid, else -1. Same for
    # every draft position m -> broadcast over T.
    win_idx = torch.arange(W, device=device)
    win_cols = torch.where(valid_target, win_idx.view(1, W), win_idx.new_full((1,), -1))
    win_cols = win_cols.view(B, 1, W).expand(B, T, W)  # [B, T, W]
    # Draft columns: block-causal. position m attends draft j<=m -> index W+j.
    j = torch.arange(T, device=device)
    causal = j.view(1, T) <= j.view(T, 1)  # [T(m), T(j)]
    draft_cols = torch.where(causal, (W + j).view(1, T), j.new_full((1,), -1))
    draft_cols = draft_cols.view(1, T, T).expand(B, T, T)  # [B, T, T]
    return torch.cat([win_cols, draft_cols], dim=-1).to(torch.int32)  # [B, T, W+T]


def _dspark_block_sparse_attention_torch(
    q: torch.Tensor,           # [B, T, H, D]
    kv: torch.Tensor,          # [B, W + T, D]  (window target-KV ++ draft-block KV)
    attn_sink: torch.Tensor,   # [H]
    valid_target: torch.Tensor,  # [B, W] bool: which window slots hold real KV
    scale: float,
) -> torch.Tensor:             # [B, T, H, D]
    """Plain-torch reference: dense block attention over (window ++ draft block).

    Kept as a kernel-free, inspectable reference. The production path
    (``_dspark_block_sparse_attention``) dispatches to the fused flash kernel.
    """
    B, T, H, D = q.shape
    W = kv.shape[1] - T
    # Scores: [B, H, T, W+T]  (broadcast single KV head over H query heads).
    scores = torch.einsum("bthd,bsd->bhts", q.float(), kv.float()) * scale
    # Mask construction.
    neg_inf = torch.finfo(scores.dtype).min
    # Window slots: valid_target [B, W] -> [B, 1, 1, W].
    win_mask = valid_target.view(B, 1, 1, W)
    # Draft-block slots: block-causal, position t attends to draft cols <= t.
    # block_causal[t, s] = (s <= t).
    draft_cols = torch.arange(T, device=q.device)
    block_causal = (draft_cols.view(1, T) <= draft_cols.view(T, 1))  # [T, T]
    block_mask = block_causal.view(1, 1, T, T).expand(B, 1, T, T)
    full_mask = torch.cat([win_mask.expand(B, 1, T, W), block_mask], dim=-1)
    scores = scores.masked_fill(~full_mask, neg_inf)
    # Attention sink: one extra always-on column per head with zero value.
    sink = attn_sink.float().view(1, H, 1, 1).expand(B, H, T, 1)
    scores_with_sink = torch.cat([scores, sink], dim=-1)
    probs = torch.softmax(scores_with_sink, dim=-1)
    probs = probs[..., :-1]  # drop the sink column (its value is 0)
    out = torch.einsum("bhts,bsd->bthd", probs, kv.float())
    return out.to(q.dtype)


def _dspark_block_sparse_attention(
    q: torch.Tensor,           # [B, T, H, D]
    kv: torch.Tensor,          # [B, W + T, D]  (window target-KV ++ draft-block KV)
    attn_sink: torch.Tensor,   # [H]
    valid_target: torch.Tensor,  # [B, W] bool: which window slots hold real KV
    scale: float,
) -> torch.Tensor:             # [B, T, H, D]
    """Per-block attention over (rolling target window ++ draft block).

    DSpark is MQA: a single shared KV head broadcast to all H query heads. Each
    draft query position t attends to all valid target-window slots plus the
    draft-block KV up to and including its own position (block-causal), with a
    per-head attention sink contributing to the softmax denominator only.

    The (window-validity + block-causal) mask is encoded as gather indices and
    dispatched to ATOM's fused flash ``sparse_attn`` (Triton + torch fallback,
    both sink+MQA aware and tuned for head_dim>=256). This avoids materializing
    the [B,H,T,W+T] fp32 score matrix. Set ``ATOM_DSPARK_ATTN_TORCH=1`` to force
    the plain-torch reference above.
    """
    import os

    if os.environ.get("ATOM_DSPARK_ATTN_TORCH", "0") == "1" or not q.is_cuda:
        return _dspark_block_sparse_attention_torch(
            q, kv, attn_sink, valid_target, scale
        )
    from atom.model_ops.sparse_attn_v4 import sparse_attn

    B, T, _, _ = q.shape
    W = kv.shape[1] - T
    topk_idxs = _dspark_block_topk_idxs(B, T, W, valid_target, q.device)
    # sparse_attn requires matching fp16/bf16 dtypes for q and kv; sink is fp32.
    return sparse_attn(
        q.contiguous(),
        kv.to(q.dtype).contiguous(),
        attn_sink.float(),
        topk_idxs,
        scale,
    )


# ---------------------------------------------------------------------------
# DSpark backbone layer + draft wrapper.
#
# These reuse the DeepSeek-V4 decoder layer machinery (attention linears, MoE,
# mHC) but run a DSpark-specific attention path: a private rolling target-KV
# window (size = sliding_window) plus the draft-block KV, dense block-causal
# attention with an attention sink, and a BF16 inverse-RoPE output projection.
#
# GPU-VERIFY: every method below that touches aiter / V4 attention submodules
# must be validated on an MI3xx device against the reference DSpark outputs.
# The numerics are kept in plain torch so they are kernel-free and inspectable.
# ---------------------------------------------------------------------------

# Heavy ATOM imports are deferred to module load only when the real engine pulls
# this in (unit tests import the heads/helpers above without these).
try:
    from atom.models.deepseek_v4 import (  # noqa: E402
        Block,
        DeepseekV4Args,
        HCState,
        make_v4_quant_config,
    )
    from atom.model_ops.layernorm import RMSNorm  # noqa: E402
    from atom.model_ops.linear import ReplicatedLinear  # noqa: E402
    from atom.model_ops.v4_kernels.state_writes import swa_write  # noqa: E402

    _ATOM_V4_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only in the stubbed test sandbox
    _ATOM_V4_AVAILABLE = False
    Block = object  # type: ignore


class DSparkLayer(Block):  # type: ignore[misc]
    """One DSpark backbone stage: a V4 decoder block with a DSpark attention path.

    Inherits ``Block`` to reuse the attention linears (wqkv_a/wq_b/wo_a/wo_b,
    q_norm/kv_norm/attn_sink/rotary_emb), the MoE FFN, and the full mHC
    (``fuse_hc``/``hc_pre``/``hc_post``) machinery. Only the attention *compute*
    is replaced: instead of V4's paged sparse attention, DSpark attends a draft
    block over its private rolling target-KV window.

    Stage-specific extras (loaded from the checkpoint):
      stage 0 (mtp.0):   main_proj [hidden*len(target_layers) -> hidden] + main_norm
                         (injects the concatenated target hidden states)
      stage last (mtp.2): markov_head, confidence_head, hc_head_{fn,base,scale}, norm
    """

    def __init__(
        self,
        layer_id: int,
        args: "DeepseekV4Args",
        *,
        stage_id: int,
        num_stages: int,
        markov_rank: int,
        target_layer_ids: tuple,
        block_size: int,
        prefix: str = "",
        alt_stream=None,
        indexer_stream=None,
    ):
        super().__init__(
            layer_id, args, prefix=prefix, alt_stream=alt_stream,
            indexer_stream=indexer_stream,
        )
        self.stage_id = stage_id
        self.num_stages = num_stages
        self.block_size = block_size
        self.window_size = args.window_size

        if stage_id == 0:
            self.main_proj = ReplicatedLinear(
                args.dim * len(target_layer_ids),
                args.dim,
                bias=False,
                quant_config=args.quant_config,
                prefix=f"{prefix}.main_proj",
            )
            self.main_norm = RMSNorm(args.dim, args.norm_eps)

        if stage_id == num_stages - 1:
            from atom.model_ops.utils import atom_parameter

            self.norm = RMSNorm(args.dim, args.norm_eps)
            self.markov_head = DSparkMarkovHead(args.vocab_size, markov_rank)
            self.confidence_head = DSparkConfidenceHead(args.dim, markov_rank)
            hc_mult = args.hc_mult
            self.hc_head_fn = atom_parameter(
                torch.empty(hc_mult, hc_mult * args.dim, dtype=torch.float32)
            )
            self.hc_head_base = atom_parameter(
                torch.empty(hc_mult, dtype=torch.float32)
            )
            self.hc_head_scale = atom_parameter(
                torch.empty(1, dtype=torch.float32)
            )

        # Private rolling target-KV window: [max_seqs, window, head_dim]. MQA, so
        # a single shared KV head. Bound lazily (max_seqs known at runtime); the
        # 1-slot buffer is a placeholder until reset_kv_cache(max_seqs) is called.
        self.register_buffer(
            "dspark_kv_cache",
            torch.zeros(1, args.window_size, self.attn.head_dim),
            persistent=False,
        )

    def reset_kv_cache(self, max_num_seqs: int, device, dtype) -> None:
        """(Re)allocate the private rolling target-KV window."""
        self.dspark_kv_cache = torch.zeros(
            max_num_seqs,
            self.window_size,
            self.attn.head_dim,
            device=device,
            dtype=dtype,
        )

    # ---- DSpark attention path (replaces Block.attn's paged sparse attn) -----

    def _compute_main_kv(
        self, main_x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Project one target hidden state per request into a rolling-window KV
        row (post kv_norm + RoPE + QAT). main_x: [B, dim] -> [B, head_dim]."""
        a = self.attn
        qr_kv = _linear_out(a.wqkv_a(main_x))
        _, kv = torch.split(qr_kv, [a.q_lora_rank, a.head_dim], dim=-1)
        kv = a.kv_norm(kv).view(-1, 1, a.head_dim)
        kv = _apply_dspark_rope_hf(
            kv, positions.reshape(-1),
            torch.cat([a.rotary_emb.cos_cache.squeeze(), a.rotary_emb.sin_cache.squeeze()], dim=-1),
            a.rope_head_dim,
        )
        _apply_dspark_kv_qat_(kv, a.rope_head_dim)
        return kv.view(-1, a.head_dim)

    def precompute_context_kv(
        self,
        main_x: torch.Tensor,       # [T, dim]  target hidden(s)
        positions: torch.Tensor,    # [T]
        cache_indices: torch.Tensor,  # [B]  rows into dspark_kv_cache
        cu_seqlens_q: torch.Tensor | None = None,  # [B+1]; None => one row/req
        write_per_batch: int = 1,
    ) -> None:
        """Write target-KV row(s) into each request's rolling window (pos % window).

        Two modes:
          * decode (default): main_x is [B, dim], one anchor token per request.
            ``cu_seqlens_q`` is the identity ramp and ``write_per_batch=1``.
          * prefill warmup: main_x is the flat [T, dim] ragged batch of all
            scheduled tokens; ``cu_seqlens_q`` ([B+1]) delimits per-request
            spans and ``write_per_batch = min(max_seqlen, window)`` so the last
            ``min(seq_len, window)`` tokens of every request seed the window.
            Without this, a request's first draft (right after prefill) sees an
            almost-empty window and rejects early; warming it lifts first-block
            acceptance to the steady-state level.

        Uses the cudagraph-safe ``swa_write`` Triton ring-buffer kernel (the same
        primitive the V4 target uses for its SWA cache). A plain advanced-index
        write (``cache[rows, slots] = kv``) is NOT cudagraph-safe: it modifies an
        nn.Module buffer with runtime indices during forward, which trips the
        compiler's buffer-mutation guard and cannot be safely graph-replayed.
        ``swa_write`` derives all indices inside the kernel from static-shaped
        inputs (it picks the last ``min(tok_n_b, write_per_batch)`` rows of each
        seq via ``cu_seqlens_q`` diff), so it replays correctly.
        """
        main_kv = self._compute_main_kv(main_x, positions)  # [T, head_dim]
        main_kv = main_kv.to(self.dspark_kv_cache.dtype).contiguous()
        if cu_seqlens_q is None:
            B = main_kv.shape[0]
            cu_seqlens_q = torch.arange(
                B + 1, device=main_kv.device, dtype=torch.int32
            )
        swa_write(
            kv=main_kv,                                   # [T, head_dim]
            positions=positions.to(torch.int32),          # [T]
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),    # [B+1] per-req spans
            state_slot_per_seq=cache_indices.to(torch.int32),  # [B] cache rows
            swa_kv=self.dspark_kv_cache,                   # [num_slots, window, D]
            cache_size=self.window_size,
            write_per_batch=write_per_batch,
        )

    def dspark_attention(
        self,
        x: torch.Tensor,            # [B, T, dim]  per-block hidden (post attn_norm)
        positions: torch.Tensor,    # [B]  anchor position per request
        cache_indices: torch.Tensor,  # [B]
    ) -> torch.Tensor:              # [B, T, dim]
        """Block attention over (rolling target window ++ draft block KV)."""
        a = self.attn
        B, T, _ = x.shape
        flat = x.reshape(B * T, -1)
        qr_kv = _linear_out(a.wqkv_a(flat))
        qr, kv = torch.split(qr_kv, [a.q_lora_rank, a.head_dim], dim=-1)
        # q_norm runs in fused_quant mode: it returns (qr_fp8, qr_scale) so the
        # downstream wq_b can skip its own input quant (x_scale=qr_scale).
        qr_normed = a.q_norm(qr)
        if isinstance(qr_normed, tuple):
            qr_q, qr_scale = qr_normed
            q = _linear_out(a.wq_b(qr_q, x_scale=qr_scale))
        else:
            q = _linear_out(a.wq_b(qr_normed))
        q = q.view(B * T, a.n_local_heads, a.head_dim)
        # Per-head Q RMSNorm (weightless), matching the V4 reference.
        q = q * torch.rsqrt(q.float().square().mean(-1, keepdim=True) + a.eps).to(q.dtype)
        kv = a.kv_norm(kv).view(B * T, 1, a.head_dim)

        # Draft positions: anchor+1 .. anchor+T.
        draft_pos = positions.view(B, 1) + torch.arange(
            1, T + 1, device=x.device, dtype=positions.dtype
        ).view(1, T)
        cos_sin = torch.cat(
            [a.rotary_emb.cos_cache.squeeze(), a.rotary_emb.sin_cache.squeeze()], dim=-1
        )
        q = _apply_dspark_rope_hf(q, draft_pos.reshape(-1), cos_sin, a.rope_head_dim)
        kv = _apply_dspark_rope_hf(kv, draft_pos.reshape(-1), cos_sin, a.rope_head_dim)
        _apply_dspark_kv_qat_(kv, a.rope_head_dim)
        q = q.view(B, T, a.n_local_heads, a.head_dim)
        kv = kv.view(B, T, a.head_dim)

        # Assemble [window ++ draft block] KV and the window-validity mask.
        rows = cache_indices.long()
        window_kv = self.dspark_kv_cache[rows]  # [B, W, head_dim]
        all_kv = torch.cat([window_kv, kv], dim=1)  # [B, W+T, head_dim]
        W = self.window_size
        slot_ids = torch.arange(W, device=x.device).view(1, W)
        valid_target = slot_ids <= positions.view(B, 1)  # filled slots only

        out = _dspark_block_sparse_attention(
            q, all_kv, a.attn_sink[: a.n_local_heads], valid_target, a.softmax_scale
        )  # [B, T, n_heads, head_dim]

        # Output projection: mirror DeepseekV4Attention.forward_impl's output
        # stage exactly (deepseek_v4.py:1922-1930): inverse-RoPE on the rope
        # lanes, grouped output-LoRA einsum with the BF16 wo_a weight, then wo_b.
        # GPU-VERIFY: numerics validated against the V4 reference output stage.
        o = out.reshape(B * T, a.n_local_heads, a.head_dim).contiguous()
        rd = a.rope_head_dim
        # Remove the absolute-position contribution carried in via value-side RoPE.
        a.rotary_emb.inverse(draft_pos.reshape(-1), o[..., -rd:])
        o = o.view(B * T, a.n_local_groups, -1)
        wo_a = a.wo_a.weight.view(a.n_local_groups, a.o_lora_rank, -1)
        o = torch.einsum("sgd,grd->sgr", o, wo_a)
        out_final = _linear_out(a.wo_b(o.flatten(1))).view(B, T, -1)
        return out_final

    def forward_block(
        self,
        x: torch.Tensor,            # [B, T, hc, dim] (stage 0) or [B, T, dim]
        positions: torch.Tensor,    # [B]
        cache_indices: torch.Tensor,  # [B]
        hc_state: "HCState | None",
    ):
        """Run one DSpark stage over a [B, T] block, returning updated hc_state.

        Mirrors Block.forward but routes attention through dspark_attention and
        keeps the [B, T] block flattened to [B*T] for the mHC + MoE ops.
        """
        B = positions.shape[0]
        T = self.block_size
        # ----- Attention sub-layer with mHC mixing -----
        if hc_state is None:
            residual = x.reshape(B * T, self.hc_mult, x.shape[-1])
            hc_state = HCState(residual=residual, post_mix=None, comb_mix=None, x_prev=None)
        hc_state = self.fuse_hc(
            hc_state, self.hc_attn_fn, self.hc_attn_scale, self.hc_attn_base,
            self.attn_norm.weight, self.norm_eps,
        )
        attn_in = hc_state.x_prev.view(B, T, -1)
        attn_out = self.dspark_attention(attn_in, positions, cache_indices)
        hc_state.x_prev = attn_out.reshape(B * T, -1)
        # ----- FFN sub-layer with mHC mixing -----
        hc_state = self.fuse_hc(
            hc_state, self.hc_ffn_fn, self.hc_ffn_scale, self.hc_ffn_base,
            self.ffn_norm.weight, self.norm_eps,
        )
        hc_state.x_prev = self.ffn(hc_state.x_prev)
        return hc_state


class DeepseekV4DSpark(nn.Module):
    """Top-level DSpark draft wrapper (mirrors DeepseekV4MTP's contract).

    Owns the DSpark backbone layers (loaded from the V4 checkpoint's ``mtp.*``
    namespace via the standard load_model path with spec_decode=True) and shares
    ``embed`` / ``head`` with the target through ``share_with_target``.

    The EagleProposer drives drafting through ``forward_spec``: a single parallel
    backbone pass produces base logits, then ``forward_head`` runs the sequential
    Markov loop to sample the block left-to-right and emit confidence scores.
    """

    # Disk `mtp.{i}.*` -> wrapper param `model.mtp.{i}.*` (same as V4 MTP).
    if _ATOM_V4_AVAILABLE:
        from atom.model_loader.loader import WeightsMapper

        weights_mapper = WeightsMapper(orig_to_new_prefix={"mtp.": "model.mtp."})
    weights_mapping = {
        ".gate.bias": ".gate.e_score_correction_bias",
        ".scale": ".weight_scale_inv",
    }
    packed_modules_mapping = {
        "attn.wq_a": ("attn.wqkv_a", 0),
        "attn.wkv": ("attn.wqkv_a", 1),
        "compressor.wkv": ("compressor.wkv_gate", 0),
        "compressor.wgate": ("compressor.wkv_gate", 1),
        "shared_experts.w1": ("shared_experts.gate_up_proj", 0),
        "shared_experts.w3": ("shared_experts.gate_up_proj", 1),
    }

    def __init__(self, config: "Config", prefix: str = "") -> None:
        super().__init__()
        self.atom_config = config
        self.hf_config = config.hf_config
        self.args = DeepseekV4Args.from_hf_config(self.hf_config)
        self.args.quant_config = make_v4_quant_config(
            self.hf_config,
            online_quant_config=getattr(config, "online_quant_config", None),
        )
        self.atom_config.quant_config = self.args.quant_config

        self.block_size = int(getattr(self.hf_config, "dspark_block_size"))
        self.markov_rank = int(getattr(self.hf_config, "dspark_markov_rank"))
        self.noise_token_id = int(getattr(self.hf_config, "dspark_noise_token_id"))
        self.target_layer_ids = tuple(
            int(i) for i in getattr(self.hf_config, "dspark_target_layer_ids")
        )
        # Number of DSpark backbone stages = number of mtp.{i}.* blocks actually
        # present in the checkpoint (3 for V4-Pro-DSpark). num_nextn_predict_layers
        # is 1 in the HF config (a serial-MTP convention) and must NOT be used
        # here, or stages mtp.1/mtp.2 (which hold the Markov + confidence heads)
        # get no home and their weights are silently dropped.
        self.num_stages = _count_dspark_stages(
            getattr(config, "model", None),
            default=int(getattr(self.hf_config, "dspark_num_layers", 0) or 0),
        )
        if self.num_stages <= 0:
            raise ValueError(
                "Could not determine DSpark stage count from the checkpoint; "
                "set dspark_num_layers in the config."
            )

        self.model = _DSparkInner(
            self.atom_config,
            self.args,
            num_stages=self.num_stages,
            markov_rank=self.markov_rank,
            target_layer_ids=self.target_layer_ids,
            block_size=self.block_size,
            noise_token_id=self.noise_token_id,
        )

    # ---- weight-loading hooks (same contract as DeepseekV4MTP) --------------

    def remap_mtp_weight_name(self, name: str) -> "str | None":
        return name if "mtp." in name else None

    @property
    def disable_fused_shared_loading(self) -> bool:
        for m in self.model.modules():
            if m.__class__.__name__ == "MoE":
                return not getattr(m, "_fuse_shared_into_routed", True)
        return False

    def get_expert_mapping(self):
        from atom.model_ops.moe import FusedMoE

        num_fused_shared = 0
        for m in self.model.modules():
            if m.__class__.__name__ == "FusedMoE":
                num_fused_shared = getattr(m, "num_fused_shared_experts", 0)
                break
        return FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="w1",
            ckpt_down_proj_name="w2",
            ckpt_up_proj_name="w3",
            num_experts=self.args.n_routed_experts + num_fused_shared,
        )

    def share_with_target(self, target_base: nn.Module, loaded: set) -> None:
        """Bind embed/head to the already-loaded target instances (no reload)."""
        self.model.embed = target_base.model.embed
        self.model.head = target_base.model.head

    def reset_kv_cache(self, max_num_seqs: int, device, dtype) -> None:
        for layer in self.model.layers:
            layer.reset_kv_cache(max_num_seqs, device, dtype)

    # ---- drafting entry points (called by the proposer) --------------------

    def precompute_context_kv(
        self,
        main_hidden,
        positions,
        cache_indices,
        cu_seqlens_q=None,
        write_per_batch: int = 1,
    ) -> None:
        """Populate every stage's rolling target-KV window from target hidden.

        Decode: one anchor row per request (defaults). Prefill warmup: pass the
        flat ragged batch with ``cu_seqlens_q`` + ``write_per_batch`` so the last
        ``min(seq_len, window)`` tokens of each request seed the window.
        """
        self.model.precompute_context_kv(
            main_hidden, positions, cache_indices, cu_seqlens_q, write_per_batch
        )

    def forward_spec(
        self,
        input_ids: torch.Tensor,    # [B]  anchor token per request (x0)
        main_hidden: torch.Tensor,  # [B, dim*len(target_layers)] concat target hidden
        positions: torch.Tensor,    # [B]  anchor position per request
        cache_indices: torch.Tensor,  # [B] rows into the rolling KV cache
    ):
        """One DSpark draft block: parallel backbone + sequential Markov head.

        Returns:
            draft_token_ids: [B, block_size]
            confidence: [B, block_size]
        """
        return self.model.forward_spec(
            input_ids, main_hidden, positions, cache_indices
        )


class _DSparkInner(nn.Module):
    """Inner module owning the DSpark backbone layers; embed/head set externally."""

    def __init__(
        self,
        atom_config: "Config",
        args: "DeepseekV4Args",
        *,
        num_stages: int,
        markov_rank: int,
        target_layer_ids: tuple,
        block_size: int,
        noise_token_id: int,
    ):
        super().__init__()
        self.args = args
        self.block_size = block_size
        self.noise_token_id = noise_token_id
        self.hc_mult = args.hc_mult
        # ModelRunner reads this to bind draft attention KV slots after the
        # target's layers (parity with V4 MTP), though DSpark uses a private
        # rolling KV cache rather than the paged pool.
        self.mtp_start_layer_idx = args.n_layers
        self.mtp = nn.ModuleList(
            [
                DSparkLayer(
                    args.n_layers + i,
                    args,
                    stage_id=i,
                    num_stages=num_stages,
                    markov_rank=markov_rank,
                    target_layer_ids=target_layer_ids,
                    block_size=block_size,
                    prefix=f"mtp.{i}",
                )
                for i in range(num_stages)
            ]
        )
        self.layers = self.mtp  # alias for reset_kv_cache iteration
        self.embed = None  # set by share_with_target
        self.head = None

    def precompute_context_kv(
        self,
        main_hidden,
        positions,
        cache_indices,
        cu_seqlens_q=None,
        write_per_batch: int = 1,
    ) -> None:
        # Stage 0 owns main_proj/main_norm; project once, reuse the rolling-KV
        # write per stage (each stage has its own kv cache + attn linears).
        stage0 = self.mtp[0]
        main_x = _linear_out(stage0.main_proj(main_hidden))
        main_x = stage0.main_norm(main_x)
        for layer in self.mtp:
            layer.precompute_context_kv(
                main_x, positions, cache_indices, cu_seqlens_q, write_per_batch
            )

    def forward_spec(self, input_ids, main_hidden, positions, cache_indices):
        B = input_ids.shape[0]
        T = self.block_size
        stage0 = self.mtp[0]
        # Inject target context: project concat target hidden -> dim, norm.
        main_x = stage0.main_proj(main_hidden)
        main_x = stage0.main_norm(main_x)  # [B, dim]  (used as rolling-KV source)

        # Build the draft block input ids: [anchor, noise, noise, ...].
        draft_ids = input_ids.new_full((B, T), self.noise_token_id)
        draft_ids[:, 0] = input_ids
        x = self.embed(draft_ids.reshape(-1)).view(B, T, -1)  # [B, T, dim]
        x = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)  # [B, T, hc, dim]

        # ----- Parallel backbone: run all stages over the block in one pass ---
        hc_state = None
        for layer in self.mtp:
            hc_state = layer.forward_block(x, positions, cache_indices, hc_state)
            x = hc_state.x_prev.view(B, T, -1)  # stage output feeds next stage

        # ----- Final mHC reduction + norm -> base logits (parallel) ----------
        last = self.mtp[-1]
        # hc_post the final residual to [B*T, dim], then last.hc_head reduce.
        residual = hc_state.residual  # [B*T, hc, dim]
        reduced = last.hc_post(
            hc_state.x_prev, residual, hc_state.post_mix, hc_state.comb_mix
        )  # [B*T, hc, dim]
        # Sigmoid-gated mHC head reduction to [B*T, dim] (reuse target head math).
        hidden = self.head.hc_head(
            reduced, last.hc_head_fn, last.hc_head_scale, last.hc_head_base
        )
        hidden = last.norm(hidden).view(B, T, -1)  # [B, T, dim]
        base_logits = self.head.get_logits(
            hidden.reshape(B * T, -1)
        ).view(B, T, -1)  # [B, T, vocab]

        # ----- Sequential Markov head: sample the block left-to-right ---------
        return self.forward_head(base_logits, hidden, input_ids)

    def forward_head(self, base_logits, hidden, anchor_ids):
        """Apply the Markov transition bias position-by-position and sample.

        paper Eq.5:  logits_k <- U_k + B(x_{k-1}, .) ;  x_k <- sample(logits_k)
        Confidence:  c_k = sigma(proj([h_k ; W1[x_{k-1}]]))
        """
        B, T, _ = base_logits.shape
        last = self.mtp[-1]
        out_ids = anchor_ids.new_empty(B, T + 1)
        out_ids[:, 0] = anchor_ids
        markov_embeds = []
        for k in range(T):
            bias, m_embed = last.markov_head(out_ids[:, k])  # [B, V], [B, r]
            logits_k = base_logits[:, k].float() + bias
            out_ids[:, k + 1] = logits_k.argmax(dim=-1)  # greedy (temp handled upstream)
            markov_embeds.append(m_embed)
        confidence = last.confidence_head(
            hidden, torch.stack(markov_embeds, dim=1)
        )  # [B, T]
        return out_ids[:, 1:], confidence
