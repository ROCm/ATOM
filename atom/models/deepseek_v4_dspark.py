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

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

if TYPE_CHECKING:
    from atom.config import Config


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

    Input is the concatenation of ``h_k`` (dim) and the Markov embedding
    ``W1[x_{k-1}]`` (rank), so the projection weight has shape
    ``[1, hidden + rank]`` (checkpoint: confidence_head.proj.weight [1, 7680]).

    ``h_k`` is the PRE-norm mHC head reduction, not the post-norm tensor the LM
    head consumes; feeding the normed one miscalibrates every score.

    The sigmoid lives here rather than in the caller (as in the reference):
    VerifyScheduler consumes ``c_k`` as an absolute probability, so the (0, 1)
    range is part of this module's contract.
    """

    def __init__(self, hidden_size: int, rank: int):
        super().__init__()
        self.proj = nn.Linear(hidden_size + rank, 1, bias=False)

    def forward(
        self, hidden_states: torch.Tensor, markov_embeds: torch.Tensor
    ) -> torch.Tensor:
        """Args:
            hidden_states: [*, hidden]  PRE-norm mHC head reduction h_k.
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
    view = x.view(-1, x.shape[-1] // block_size, block_size)
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
    """Encode the window-validity attention mask as gather indices into the
    combined ``[window ++ draft]`` KV (length ``W+T``).

    For draft query position ``m`` (0..T-1) the attended columns are:
      * every VALID rolling-window slot  -> global index ``w`` (0..W-1)
      * EVERY draft-block slot           -> global index ``W + j`` (j = 0..T-1)
    Invalid window slots are ``-1`` (the fused sparse_attn kernel skips them).

    Attention inside the draft block is BIDIRECTIONAL: the block is decoded in
    one parallel pass, so position is carried by RoPE, not by a causal mask.

    Returns: topk_idxs [B, T, W+T] int32, suitable for ``sparse_attn``.
    """
    # Built int32 end to end: the index ramps start as int32 so the `cat` lands
    # directly in the output dtype. Defaulting to int64 and casting afterwards
    # would materialize the full [B, T, W+T] block twice (once int64, once int32).
    #
    # Window columns: keep the global slot index where valid, else -1. Same for
    # every draft position m -> broadcast over T.
    win_idx = torch.arange(W, device=device, dtype=torch.int32)
    win_cols = torch.where(valid_target, win_idx.view(1, W), win_idx.new_full((1,), -1))
    win_cols = win_cols.view(B, 1, W).expand(B, T, W)  # [B, T, W]
    # Draft columns: every query row attends the WHOLE block -> index W+j.
    j = torch.arange(T, device=device, dtype=torch.int32)
    draft_cols = (W + j).view(1, 1, T).expand(B, T, T)  # [B, T, T]
    return torch.cat([win_cols, draft_cols], dim=-1)  # [B, T, W+T] int32


@dataclass(frozen=True)
class _DSparkBlockPlan:
    """Per-block invariants shared by every DSpark stage.

    ``dspark_attention`` runs once per stage, but these three tensors depend only
    on ``(positions, T, W)`` — never on stage weights — so they are built once per
    ``forward_spec`` and reused. Recomputing them per stage rebuilt the
    ``[B, T, W+T]`` gather-index block once per stage for identical values.
    """

    draft_pos: torch.Tensor  # [B, T]      anchor+1 .. anchor+T
    valid_target: torch.Tensor  # [B, W]      rolling-window slot validity
    topk_idxs: torch.Tensor  # [B, T, W+T] sparse_attn gather indices


def _build_block_plan(
    positions: torch.Tensor,  # [B] anchor position per request
    T: int,  # draft width
    W: int,  # rolling window size
    is_dummy_run: bool,
) -> _DSparkBlockPlan:
    """Build the per-block invariants once for the whole DSpark backbone."""
    B = positions.shape[0]
    device = positions.device
    offsets = torch.arange(1, T + 1, device=device, dtype=positions.dtype)
    draft_pos = positions.view(B, 1) + offsets.view(1, T)
    if is_dummy_run:
        # warmup runs BEFORE allocate_kv_cache: no window is bound yet, so every
        # slot is invalid (the draft output is discarded anyway).
        valid_target = torch.zeros(B, W, dtype=torch.bool, device=device)
    else:
        # slot s valid iff its absolute position (anchor-(W-1)+s) >= 0.
        slot_ids = torch.arange(W, device=device).view(1, W)
        valid_target = slot_ids >= (W - 1) - positions.view(B, 1)
    return _DSparkBlockPlan(
        draft_pos=draft_pos,
        valid_target=valid_target,
        topk_idxs=_dspark_block_topk_idxs(B, T, W, valid_target, device),
    )


def _dspark_block_sparse_attention_torch(
    q: torch.Tensor,  # [B, T, H, D]
    kv: torch.Tensor,  # [B, W + T, D]  (window target-KV ++ draft-block KV)
    attn_sink: torch.Tensor,  # [H]
    valid_target: torch.Tensor,  # [B, W] bool: which window slots hold real KV
    scale: float,
) -> torch.Tensor:  # [B, T, H, D]
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
    # Draft-block slots: bidirectional — every draft position attends the whole
    # block (see _dspark_block_topk_idxs; the block is decoded in parallel).
    block_mask = q.new_ones(1, 1, T, T, dtype=torch.bool).expand(B, 1, T, T)
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
    q: torch.Tensor,  # [B, T, H, D]
    kv: torch.Tensor,  # [B, W + T, D]  (window target-KV ++ draft-block KV)
    attn_sink: torch.Tensor,  # [H]
    valid_target: torch.Tensor,  # [B, W] bool: which window slots hold real KV
    topk_idxs: torch.Tensor,  # [B, T, W+T] int32 gather indices (from the block plan)
    scale: float,
) -> torch.Tensor:  # [B, T, H, D]
    """Per-block attention over (rolling target window ++ draft block).

    DSpark is MQA: a single shared KV head broadcast to all H query heads. Each
    draft query position t attends to all valid target-window slots plus the
    ENTIRE draft-block KV (bidirectional within the block), with a per-head
    attention sink contributing to the softmax denominator only.

    The window-validity mask is encoded as gather indices
    (``topk_idxs``, built once per block by :func:`_build_block_plan`) and
    dispatched to ATOM's fused flash ``sparse_attn`` (Triton + torch fallback,
    both sink+MQA aware and tuned for head_dim>=256). This avoids materializing
    the [B,H,T,W+T] fp32 score matrix. Set ``ATOM_DSPARK_ATTN_TORCH=1`` to force
    the plain-torch reference above (it re-derives the mask from
    ``valid_target`` and ignores ``topk_idxs``).
    """
    import os

    if os.environ.get("ATOM_DSPARK_ATTN_TORCH", "0") == "1" or not q.is_cuda:
        return _dspark_block_sparse_attention_torch(
            q, kv, attn_sink, valid_target, scale
        )
    from atom.model_ops.sparse_attn_v4 import sparse_attn

    # sparse_attn requires matching fp16/bf16 dtypes for q and kv; sink is fp32.
    # Asserted rather than cast: q and kv reach here from different sources (the
    # qk-norm/RoPE output vs the [paged window ++ draft block] concat), so a
    # mismatch is a real bug — a silent `.to()` would hide it behind a per-step
    # GPU copy.
    assert kv.dtype == q.dtype, (
        f"DSpark block attention needs matching q/kv dtypes, "
        f"got q={q.dtype} kv={kv.dtype}."
    )
    return sparse_attn(q, kv, attn_sink.float(), topk_idxs, scale)


# ---------------------------------------------------------------------------
# DSpark backbone layer + draft wrapper.
#
# These reuse the DeepSeek-V4 decoder layer machinery (attention linears, MoE,
# mHC) but run a DSpark-specific attention path: a private rolling target-KV
# window (size = sliding_window) plus the draft-block KV, dense attention that
# is bidirectional within the draft block, with an attention sink, and a BF16
# inverse-RoPE output projection.
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
    from atom.model_ops.v4_kernels.state_writes import (  # noqa: E402
        dspark_paged_window_gather,
        swa_write,
    )
    from atom.model_ops.v4_kernels.qk_norm_rope_maybe_quant import (  # noqa: E402
        qk_norm_rope_maybe_quant,
    )

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
            layer_id,
            args,
            prefix=prefix,
            alt_stream=alt_stream,
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
            self.hc_head_scale = atom_parameter(torch.empty(1, dtype=torch.float32))

        # PAGED-SWA: draft window KV lives in a paged pool bound by
        # DeepseekV4AttentionMetadataBuilder.build_kv_cache_tensor at
        # allocate_kv_cache; see precompute_context_kv / dspark_attention.
        #
        # Mark this attn as a DSpark draft layer so the builder always binds it a
        # PRIVATE bf16 SWA pool, even under an fp8 target KV cache. DSpark's block
        # attention runs bf16 (no fused fp8 kernel for its [window ++ draft-block]
        # shape), so an fp8 draft window is a measured net regression. The target
        # KV cache is unaffected (still fp8).
        self.attn.dspark_draft = True

    def reset_kv_cache(self, max_num_seqs: int, device, dtype) -> None:
        """No-op: draft KV is paged into the shared pool (bound at
        allocate_kv_cache), not a private per-layer ring. Kept for eagle.py's
        `hasattr(model, "reset_kv_cache")` call contract."""
        return

    # ---- DSpark attention path (replaces Block.attn's paged sparse attn) -----

    def _compute_main_kv(
        self, main_x: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """Project target hidden states into rolling-window KV rows (post
        kv_norm + RoPE + QAT). main_x: [T, dim] -> [T, head_dim].

        ``positions`` is the caller's padded forward buffer and may be longer
        than ``main_x``; only its first T entries are used.

        The NoPE lanes are fake-quantized through fp8 E4M3 (DSpark QAT numerics)
        then stored bf16 — matching the QAT-trained draft's expected KV values."""
        a = self.attn
        qr_kv = _linear_out(a.wqkv_a(main_x))
        _, kv = torch.split(qr_kv, [a.q_lora_rank, a.head_dim], dim=-1)
        kv = a.kv_norm(kv).view(-1, 1, a.head_dim)
        rope_dim = a.rope_head_dim
        # RoPE via the shared aiter fused kernel (rope_cached_positions, GPT-J
        # interleaved = the same rotate_style=1 layout the draft used to apply via
        # its own triton kernel). In-place on the rope-slice only (aiter handles
        # the non-contiguous [..., -rope_dim:] slice via strides); `kv` is a fresh
        # local tensor so the in-place write is safe. Guard rope_dim > 0 so
        # rope_dim == 0 doesn't turn `[..., -rope_dim:]` into the whole head
        # (-0 == 0).
        if rope_dim:
            a.rotary_emb.forward(positions.view(-1)[: kv.shape[0]], kv[..., -rope_dim:])
        _apply_dspark_kv_qat_(kv, rope_dim)
        return kv.view(-1, a.head_dim)

    def precompute_context_kv(
        self,
        main_x: torch.Tensor,  # [T, dim]  target hidden(s)
        positions: torch.Tensor,  # [T]
        cu_seqlens_q: torch.Tensor,  # [B+1] int32 per-req spans into main_x
        write_per_batch: int,
    ) -> None:
        """Write target-KV rows into each request's rolling window.

        ``main_x`` is the flat [T, dim] ragged batch of every scheduled token and
        ``cu_seqlens_q`` ([B+1]) delimits the per-request spans; the last
        ``min(seq_len, write_per_batch)`` rows of each span are written. Callers
        pass ``write_per_batch = window_size``, which covers a prefill tail and a
        decode verify span alike.

        Every scheduled row is written on every step: the window must hold a row
        for every position it spans, and the read side gathers slots by absolute
        position regardless of whether they were ever written. See
        :meth:`DSparkProposer.propose` for why writing rejected rows is safe.

        PAGED-SWA: the draft window KV now lives in the shared paged pool
        (``self.attn.swa_kv``, this draft layer's slice of ``unified_kv``),
        content-addressed by ``swa_block_tables`` exactly like the V4 target SWA
        (#1417). ``swa_write`` is the same cudagraph-safe Triton kernel the target
        uses: it derives all indices in-kernel from ``cu_seqlens_q`` +
        ``positions`` (no advanced-index buffer-mutation, no ``.item()`` sync), so
        it graph-replays correctly. The physical destination comes entirely from
        ``swa_block_tables``, so no per-request state slot is needed here.
        """
        from atom.utils.forward_context import get_forward_context

        fc = get_forward_context()
        # warmup_model runs BEFORE allocate_kv_cache, so `self.attn.swa_kv` /
        # `swa_block_size` are unbound and `swa_block_tables` is absent. Same
        # short-circuit as the V4 target (deepseek_v4.py is_dummy_run guard):
        # skip the paged SWA write on dummy runs — warmup discards draft output.
        if fc.context.is_dummy_run:
            return
        attn_md = fc.attn_metadata
        a = self.attn
        main_kv = self._compute_main_kv(main_x, positions)  # [T, head_dim]
        # The SWA pool is allocated bf16 for DSpark draft layers regardless of the
        # target's kv_cache_dtype, while main_kv carries the model dtype — two
        # independent sources. Assert instead of casting so a mismatch surfaces
        # here rather than as a silent per-step copy (or a silently reinterpreted
        # store inside the swa_write kernel, which has no dtype guard).
        assert main_kv.dtype == a.swa_kv.dtype, (
            f"DSpark draft KV dtype {main_kv.dtype} != SWA pool dtype "
            f"{a.swa_kv.dtype}."
        )
        B = cu_seqlens_q.shape[0] - 1
        swa_write(
            main_kv,  # [T, head_dim]
            positions,  # [T] int64
            cu_seqlens_q,  # [B+1] int32, per-req spans
            attn_md.swa_block_tables[:B],  # [B, max_blocks]
            a.swa_kv,  # [num_pages, head_dim]
            a.swa_block_size,
            write_per_batch,
        )

    def dspark_attention(
        self,
        x: torch.Tensor,  # [B, T, dim]  per-block hidden (post attn_norm)
        positions: torch.Tensor,  # [B]  anchor position per request
        plan: "_DSparkBlockPlan",  # per-block invariants, shared across stages
    ) -> torch.Tensor:  # [B, T, dim]
        """Block attention over (rolling target window ++ draft block KV)."""
        a = self.attn
        B, T, _ = x.shape
        flat = x.view(B * T, -1)
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
        # q stays 2-D [B*T, H*D], kv 2-D [B*T, D] — the fused kernel wants 2-D.

        # Draft positions (anchor+1 .. anchor+T) come from the shared block plan.
        draft_pos = plan.draft_pos
        rope_dim = a.rope_head_dim
        # Per-head weightless Q RMSNorm + weighted KV RMSNorm + GPT-J RoPE in ONE
        # fused kernel — the same `qk_norm_rope_maybe_quant` the V4 target runs
        # every layer (bf16 path: quant off, no SWA fusion — DSpark scatters its
        # window separately). Replaces the draft's hand-written weightless Q-norm
        # + kv_norm + the `rotary_emb.forward` (_V4RoPE) RoPE launch. `kv` is
        # passed PRE-norm; the kernel applies kv_norm.weight internally.
        qkn = qk_norm_rope_maybe_quant(
            q,
            kv,
            a.kv_norm.weight,
            a.rotary_emb.cos_cache,
            a.rotary_emb.sin_cache,
            draft_pos.view(-1),
            a.n_local_heads,
            a.head_dim,
            rope_dim,
            a.eps,
            quant_q=False,
            quant_k=False,
            prefix=f"{a.layer_name}.dspark_qk_norm_rope",
        )
        q = qkn.q_sa.view(B, T, a.n_local_heads, a.head_dim)
        kv = qkn.kv.view(B * T, 1, a.head_dim)
        _apply_dspark_kv_qat_(kv, rope_dim)
        kv = kv.view(B, T, a.head_dim)

        # Assemble the [window ++ draft block] KV. The window-validity mask and
        # gather indices are stage-invariant and come from the block plan; only
        # the KV gather is per-stage (each stage owns its own swa_kv slice).
        # PAGED-SWA: gather the dense [B, W, head_dim] rolling window from the
        # shared paged pool (this draft layer's swa_kv slice), addressed by
        # swa_block_tables — the same content-addressing the write used.
        from atom.utils.forward_context import get_forward_context

        fc = get_forward_context()
        W = self.window_size
        if fc.context.is_dummy_run:
            # warmup runs BEFORE allocate_kv_cache → swa_kv / swa_block_tables
            # unbound. All-zero window so the forward still compiles at shape
            # (draft output is discarded).
            window_kv = kv.new_zeros(B, W, a.head_dim)
        else:
            attn_md = fc.attn_metadata
            window_kv = dspark_paged_window_gather(
                a.swa_kv,  # [num_pages, head_dim]
                attn_md.swa_block_tables[:B],  # [B, max_blocks]
                positions,  # [B] anchor positions
                W,
                a.swa_block_size,
            )  # [B, W, head_dim]
        all_kv = torch.cat([window_kv, kv], dim=1)  # [B, W+T, head_dim]

        out = _dspark_block_sparse_attention(
            q,
            all_kv,
            a.attn_sink[: a.n_local_heads],
            plan.valid_target,
            plan.topk_idxs,
            a.softmax_scale,
        )  # [B, T, n_heads, head_dim]

        # Output projection: mirror DeepseekV4Attention.forward_impl's output
        # stage exactly (deepseek_v4.py:1922-1930): inverse-RoPE on the rope
        # lanes, grouped output-LoRA einsum with the BF16 wo_a weight, then wo_b.
        # GPU-VERIFY: numerics validated against the V4 reference output stage.
        o = out.view(B * T, a.n_local_heads, a.head_dim)
        rope_dim = a.rope_head_dim
        # Remove the absolute-position contribution carried in via value-side RoPE.
        a.rotary_emb.inverse(draft_pos.view(-1), o[..., -rope_dim:])
        o = o.view(B * T, a.n_local_groups, -1)
        wo_a = a.wo_a.weight.view(a.n_local_groups, a.o_lora_rank, -1)
        o = torch.einsum("sgd,grd->sgr", o, wo_a)
        out_final = _linear_out(a.wo_b(o.flatten(1))).view(B, T, -1)
        return out_final

    def forward_block(
        self,
        x: torch.Tensor,  # [B, T, hc, dim] (stage 0) or [B, T, dim]
        positions: torch.Tensor,  # [B]
        plan: "_DSparkBlockPlan",  # per-block invariants, shared across stages
        hc_state: "HCState | None",
    ):
        """Run one DSpark stage over a [B, T] block, returning updated hc_state.

        Mirrors Block.forward but routes attention through dspark_attention and
        keeps the [B, T] block flattened to [B*T] for the mHC + MoE ops.
        """
        B = positions.shape[0]
        T = x.shape[1]
        # ----- Attention sub-layer with mHC mixing -----
        if hc_state is None:
            residual = x.view(B * T, self.hc_mult, x.shape[-1])
            hc_state = HCState(
                residual=residual, post_mix=None, comb_mix=None, x_prev=None
            )
        hc_state = self.fuse_hc(
            hc_state,
            self.hc_attn_fn,
            self.hc_attn_scale,
            self.hc_attn_base,
            self.attn_norm.weight,
            self.norm_eps,
        )
        attn_in = hc_state.x_prev.view(B, T, -1)
        attn_out = self.dspark_attention(attn_in, positions, plan)
        hc_state.x_prev = attn_out.view(B * T, -1)
        # ----- FFN sub-layer with mHC mixing -----
        hc_state = self.fuse_hc(
            hc_state,
            self.hc_ffn_fn,
            self.hc_ffn_scale,
            self.hc_ffn_base,
            self.ffn_norm.weight,
            self.norm_eps,
        )
        hc_state.x_prev = self.ffn(hc_state.x_prev)
        return hc_state


class DeepseekV4DSpark(nn.Module):
    """Top-level DSpark draft wrapper (mirrors DeepseekV4MTP's contract).

    Owns the DSpark backbone layers (loaded from the V4 checkpoint's ``mtp.*``
    namespace via the standard load_model path with spec_decode=True) and shares
    ``embed`` / ``head`` with the target through ``share_with_target``.

    The DSparkProposer drives drafting through ``forward_spec``: a single parallel
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
            # Target parity (deepseek_v4.py): without model_path the
            # wo_a-is-BF16-on-disk probe returns False, and a ckpt shipping BF16
            # wo_a would give the draft FP8+scale params and garbage attention.
            model_path=getattr(config, "model", None),
            online_quant_config=getattr(config, "online_quant_config", None),
        )
        self.atom_config.quant_config = self.args.quant_config

        self.block_size = int(getattr(self.hf_config, "dspark_block_size"))
        # Rolling target-KV window width. Exposed on the wrapper (top level) so the
        # proposer never reaches through `self.model.model.mtp[0]` to read it.
        self.window_size = int(self.args.window_size)
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
        cu_seqlens_q,
        write_per_batch: int,
    ) -> None:
        """Populate every stage's rolling target-KV window from target hidden.

        Pass the flat ragged batch of all scheduled tokens with the real
        ``cu_seqlens_q`` and ``write_per_batch = window_size``. See
        :meth:`DSparkLayer.precompute_context_kv`.
        """
        self.model.precompute_context_kv(
            main_hidden, positions, cu_seqlens_q, write_per_batch
        )

    def forward_spec(
        self,
        input_ids: torch.Tensor,  # [B]  anchor token per request (x0)
        positions: torch.Tensor,  # [B]  anchor position per request
        num_draft: "int | None" = None,  # draft width (defaults to block_size)
    ):
        """One DSpark draft block: parallel backbone + sequential Markov head.

        Takes no target hidden: the target context reaches the block through the
        rolling KV window, which ``precompute_context_kv`` must have populated
        for this step beforehand.

        ``num_draft`` selects the draft width; when the verify horizon
        (num_speculative_tokens) exceeds ``dspark_block_size`` the caller passes
        the larger width and the block is drafted at that width in one pass.

        Returns:
            draft_token_ids: [B, num_draft]
            confidence: [B, num_draft]
        """
        return self.model.forward_spec(input_ids, positions, num_draft=num_draft)


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
        cu_seqlens_q,
        write_per_batch: int,
    ) -> None:
        # Stage 0 owns main_proj/main_norm; project once, reuse the rolling-KV
        # write per stage (each stage has its own kv cache + attn linears).
        stage0 = self.mtp[0]
        main_x = _linear_out(stage0.main_proj(main_hidden))
        main_x = stage0.main_norm(main_x)
        for layer in self.mtp:
            layer.precompute_context_kv(
                main_x, positions, cu_seqlens_q, write_per_batch
            )

    def forward_spec(self, input_ids, positions, num_draft=None):
        B = input_ids.shape[0]
        # Draft width defaults to the training block size but may be widened up to
        # the rolling window when num_speculative_tokens > block_size (the weights
        # are draft-width-agnostic; positions past the block size are RoPE-
        # extrapolated). Cap at window_size so the [window ++ draft] KV stays sane.
        T = int(num_draft) if num_draft is not None else self.block_size
        # No main_proj here: the target context reaches the draft block through
        # the rolling KV window (written by precompute_context_kv and gathered in
        # each stage's attention), not through this forward's activations.

        # Build the draft block input ids: [anchor, noise, noise, ...].
        draft_ids = input_ids.new_full((B, T), self.noise_token_id)
        draft_ids[:, 0] = input_ids
        x = self.embed(draft_ids.view(-1)).view(B, T, -1)  # [B, T, dim]
        x = x.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)  # [B, T, hc, dim]

        # Per-block invariants (draft positions, window validity, gather indices)
        # depend only on (positions, T, W), so build them once and share across
        # every stage instead of recomputing them inside each stage's attention.
        from atom.utils.forward_context import get_forward_context

        plan = _build_block_plan(
            positions,
            T,
            self.mtp[0].window_size,
            get_forward_context().context.is_dummy_run,
        )

        # ----- Parallel backbone: run all stages over the block in one pass ---
        hc_state = None
        for layer in self.mtp:
            hc_state = layer.forward_block(x, positions, plan, hc_state)
            x = hc_state.x_prev.view(B, T, -1)  # stage output feeds next stage

        # ----- Final mHC reduction + norm -> base logits (parallel) ----------
        last = self.mtp[-1]
        # hc_post the final residual to [B*T, dim], then last.hc_head reduce.
        residual = hc_state.residual  # [B*T, hc, dim]
        reduced = last.hc_post(
            hc_state.x_prev, residual, hc_state.post_mix, hc_state.comb_mix
        )  # [B*T, hc, dim]
        # Sigmoid-gated mHC head reduction to [B*T, dim] (reuse target head math).
        # `norm` applies to the LM head input only: the confidence head takes the
        # PRE-norm reduction (matching the reference).
        hc_hidden = self.head.hc_head(
            reduced, last.hc_head_fn, last.hc_head_scale, last.hc_head_base
        )  # [B*T, dim]
        base_logits = self.head.get_logits(last.norm(hc_hidden)).view(
            B, T, -1
        )  # [B, T, vocab]

        # ----- Sequential Markov head: sample the block left-to-right ---------
        return self.forward_head(base_logits, hc_hidden.view(B, T, -1), input_ids)

    def forward_head(self, base_logits, hc_hidden, anchor_ids):
        """Apply the Markov transition bias position-by-position and sample.

        paper Eq.5:  logits_k <- U_k + B(x_{k-1}, .) ;  x_k <- sample(logits_k)
        Confidence:  c_k = sigma(proj([h_k ; W1[x_{k-1}]]))

        ``hc_hidden`` is the PRE-norm mHC reduction [B, T, dim] (the confidence
        head's ``h_k``); ``base_logits`` came from the post-norm tensor.
        """
        B, T, _ = base_logits.shape
        last = self.mtp[-1]
        out_ids = anchor_ids.new_empty(B, T + 1)
        out_ids[:, 0] = anchor_ids
        markov_embeds = []
        for k in range(T):
            bias, m_embed = last.markov_head(out_ids[:, k])  # [B, V], [B, r]
            logits_k = base_logits[:, k].float() + bias
            out_ids[:, k + 1] = logits_k.argmax(
                dim=-1
            )  # greedy (temp handled upstream)
            markov_embeds.append(m_embed)
        confidence = last.confidence_head(
            hc_hidden, torch.stack(markov_embeds, dim=1)
        )  # [B, T]
        return out_ids[:, 1:], confidence
