# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# The MiniMax-H3 architecture and its packed-sequence serving contract follow
# the reference implementation in sgl-project/sglang
# (python/sglang/multimodal_gen/runtime/models/dits/minimax_h3.py,
# Apache-2.0). This is an independent implementation for ATOM: it drops the
# tensor-parallel machinery (the DiT runs at tp_size == 1, with Ulysses as the
# parallel axis), uses ATOM's UlyssesGroup for the sequence<->head all-to-all,
# and routes attention through aiter.

"""MiniMax-H3 packed-token audio-video DiT.

Forward contract, verified against a live 4-rank run at 1344x768x124f. Note the
mix of global and rank-local inputs -- this is the main source of porting risk:

    x                       [1, S, 96]      global   (24 latent x 1x2x2 patch)
    audio_x                 [1, S, 32]      global
    img_position_ids        [1, S, 3]       global
    inverse_indices         [S]             global
    update_mask             [n_img]         global
    rope_cache              ([S/W, 96], ..) LOCAL
    block_token_tags        [S/W]           LOCAL
    block_combined_indices  [S/W]           LOCAL
    local_embedding_layout  row/global ids  LOCAL
    prompt_embeds           [T, H]          replicated
    packed_seq_params       cu_seqlens      global
    -> video [n_img, 96] fp32, audio [n_audio, 32] fp32

Precision is not uniform and must not be "cleaned up": patch projections, the
timestep embedding and both output heads run fp32; the block stack runs bf16.
"""

import math
from typing import Any

import torch
from torch import nn

from atom.diffusion.attention import (
    AttentionBackend,
    packed_varlen_attention,
    resolve_attention_backend,
)
from atom.diffusion.models.minimax_h3.arch import (
    MINIMAX_H3_ADALN_MODALITY_NUM,
    MiniMaxH3DiTArchConfig,
)
from atom.diffusion.ulysses import UlyssesGroup

_BF16 = torch.bfloat16
_FP32 = torch.float32


def _linear(
    in_features: int,
    out_features: int,
    *,
    bias: bool,
    dtype: torch.dtype,
) -> nn.Linear:
    """Linear factory for the DiT.

    The DiT runs at ``tp_size == 1`` -- Ulysses parallelises the sequence, not
    the hidden dim -- so ATOM's Column/Row parallel wrappers would degenerate to
    a plain matmul. Swap this factory for ``atom.model_ops.linear`` if TP or
    quantization is ever wanted.
    """
    return nn.Linear(in_features, out_features, bias=bias, dtype=dtype)


def _norm(size: int, *, eps: float, dtype: torch.dtype = _BF16) -> nn.RMSNorm:
    # torch.nn.RMSNorm upcasts reduced-precision input for the variance
    # reduction, which is the fp32-accumulate semantic the checkpoint expects.
    return nn.RMSNorm(size, eps=eps, dtype=dtype)


def reorder_grouped_qkv_to_qkv(
    weight: torch.Tensor,
    *,
    num_query_groups: int,
    heads_per_group: int,
    head_dim: int,
) -> torch.Tensor:
    """Reorder the checkpoint's interleaved-grouped QKV into [Q; K; V].

    The checkpoint stores ``num_query_groups`` blocks of
    ``(heads_per_group + 2) * head_dim`` rows, so a plain three-way split of the
    fused tensor is silently wrong.
    """
    per_group = (heads_per_group + 2) * head_dim
    expected_out = num_query_groups * per_group
    if weight.shape[0] != expected_out:
        raise ValueError(
            f"grouped qkv weight has output dim {weight.shape[0]}, "
            f"expected {expected_out}"
        )
    rest = weight.shape[1:]
    grouped = weight.reshape(num_query_groups, per_group, *rest)
    q, k, v = torch.split(
        grouped, [heads_per_group * head_dim, head_dim, head_dim], dim=1
    )
    return torch.cat(
        [
            q.reshape(num_query_groups * heads_per_group * head_dim, *rest),
            k.reshape(num_query_groups * head_dim, *rest),
            v.reshape(num_query_groups * head_dim, *rest),
        ],
        dim=0,
    )


def _modulate_scale_shift(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """x * (1 + scale[idx]) + shift[idx], gathered per token modality."""
    return (
        x * (1.0 + scale.index_select(0, indices)) + shift.index_select(0, indices)
    ).to(dtype)


def _norm_modulate(
    norm: nn.RMSNorm,
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """RMSNorm then indexed scale/shift, in one kernel where aiter has it.

    Unfused, the normalised activation is written and immediately re-read, and
    both modulation tables are materialised at [tokens, hidden] by the gather.
    Measured 5.2x on a 63,232 x 5376 row block.
    """
    if x.device.type == "cuda" and x.dtype is _BF16:
        try:
            from aiter.ops.triton.fusions.fused_rmsnorm_indexed_adaln import (
                fused_rmsnorm_indexed_adaln,
            )
        except ImportError:
            pass
        else:
            return fused_rmsnorm_indexed_adaln(
                x.contiguous(), norm.weight, shift, scale, indices, eps=norm.eps
            )
    return _modulate_scale_shift(norm(x), shift, scale, indices, dtype=_BF16)


def _modulate_gate(
    x: torch.Tensor,
    gate: torch.Tensor,
    other: torch.Tensor,
    indices: torch.Tensor,
    *,
    dtype: torch.dtype,
) -> torch.Tensor:
    """x + gate[idx] * other, gathered per token modality."""
    return (x + gate.index_select(0, indices) * other).to(dtype)


def _silu_mul(hidden: torch.Tensor) -> torch.Tensor:
    gate, up = hidden.chunk(2, dim=-1)
    return nn.functional.silu(gate) * up


def _rope_qk(
    q: torch.Tensor, k: torch.Tensor, cos_sin_cache: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """3-D NeoX RoPE on q and k from a packed cos|sin cache ``[T, rot_dim]``.

    Only the leading ``rot_dim`` head dims rotate; the rest passes through.
    """
    half = cos_sin_cache.shape[-1] // 2
    cos_half, sin_half = cos_sin_cache.split(half, dim=-1)
    cos = torch.cat((cos_half, cos_half), dim=-1).unsqueeze(1)
    sin = torch.cat((sin_half, sin_half), dim=-1).unsqueeze(1)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        x_rot, x_pass = x[..., : 2 * half], x[..., 2 * half :]
        x1, x2 = torch.chunk(x_rot, 2, dim=-1)
        rotated = x_rot * cos + torch.cat((-x2, x1), dim=-1) * sin
        return torch.cat((rotated, x_pass), dim=-1)

    return rotate(q), rotate(k)


def _qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm: nn.RMSNorm,
    k_norm: nn.RMSNorm,
    rope_cache: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """QK-Norm then 3-D RoPE, in one kernel where aiter has it.

    Unfused this is a norm plus, per tensor, a slice, a negate, two
    concatenates and two multiplies over [tokens, heads, 128] -- and the cos/sin
    row is broadcast into a full-size temporary. Measured 9.3x fused, and the
    kernel writes through the qkv projection's own storage, so the split views
    are never materialised.
    """
    if (
        rope_cache is not None
        and q.device.type == "cuda"
        and q.dtype is _BF16
        and q.stride(-1) == 1
        and k.stride(-1) == 1
    ):
        try:
            from aiter.ops.triton.rope.fused_qk_norm_rope_cached import (
                fused_qk_norm_rope_cached,
            )
        except ImportError:
            pass
        else:
            return fused_qk_norm_rope_cached(
                q, k, q_norm.weight, k_norm.weight, rope_cache, eps=q_norm.eps
            )

    # The fallback norm is deliberately ungated on ``is_cuda``: that is True on
    # ROCm, and the reference uses it to reach a CUDA-only JIT that cannot
    # build here. Gate on capability, never on ``is_cuda``.
    q, k = q_norm(q), k_norm(k)
    return (q, k) if rope_cache is None else _rope_qk(q, k, rope_cache)


def rope_cos_sin_cache(freqs: torch.Tensor, *, dtype: torch.dtype) -> torch.Tensor:
    """Build the activation-dtype cos|sin cache from raw rope frequencies."""
    half = freqs.shape[-1] // 2
    return (
        torch.cat((torch.cos(freqs[:, :half]), torch.sin(freqs[:, :half])), dim=-1)
        .to(dtype=dtype, copy=False)
        .contiguous()
    )


class MiniMaxH3Rope(nn.Module):
    """3-D RoPE over (t, h, w); rotates 96 of 128 head dims."""

    # theta, verified against the checkpoint's own rope.inv_freq to 7e-9.
    ROPE_THETA = 10000.0

    def __init__(self, inv_freq_len: int) -> None:
        super().__init__()
        # Initialised, not torch.empty: an unloaded model would otherwise read
        # uninitialised memory and emit NaN velocities intermittently.
        index = torch.arange(inv_freq_len, dtype=_FP32)
        self.register_buffer(
            "inv_freq",
            1.0 / (self.ROPE_THETA ** (index / inv_freq_len)),
            persistent=True,
        )

    def forward(self, img_position_ids: torch.Tensor) -> torch.Tensor:
        """[1, S, 3] (t, h, w) -> freqs [S, 6 * inv_freq_len]."""
        if img_position_ids.dim() != 3 or img_position_ids.shape[0] != 1:
            raise ValueError(
                f"img_position_ids must be [1, S, 3], got "
                f"{list(img_position_ids.shape)}"
            )
        pos = img_position_ids[0].to(_FP32)
        per_axis = pos.unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        t_f, h_f, w_f = per_axis.unbind(dim=1)
        half = torch.cat((t_f, h_f, w_f), dim=-1)
        return torch.cat((half, half), dim=-1)


class MiniMaxH3TimeEmbedder(nn.Module):
    """Sinusoidal timestep embedding, fp32 throughout."""

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        self.frequency_embedding_size = arch.timestep_input_dim
        self.proj_in = _linear(
            arch.timestep_input_dim,
            arch.time_embed_hidden_size,
            bias=True,
            dtype=_FP32,
        )
        self.proj_out = _linear(
            arch.time_embed_hidden_size, arch.time_embed_dim, bias=True, dtype=_FP32
        )
        self._frequency_cache: torch.Tensor | None = None

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """[M] -> [M, time_embed_dim] fp32. Cosine values precede sine."""
        half = self.frequency_embedding_size // 2
        freqs = self._frequency_cache
        if freqs is None or freqs.device != t.device:
            freqs = torch.exp(
                -math.log(10000.0)
                * torch.arange(half, dtype=_FP32, device=t.device)
                / half
            )
            self._frequency_cache = freqs
        args = t.to(_FP32)[:, None] * freqs[None]
        t_freq = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.proj_out(nn.functional.silu(self.proj_in(t_freq)))


class MiniMaxH3Attention(nn.Module):
    """Packed varlen attention with an optional Ulysses head/sequence trade."""

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        *,
        attn_backend: "AttentionBackend | str | None" = None,
    ) -> None:
        super().__init__()
        self.attn_backend = resolve_attention_backend(attn_backend)
        self.num_heads = arch.num_attention_heads
        self.head_dim = arch.attention_head_dim
        self.inner_dim = arch.inner_dim
        self.softmax_scale = self.head_dim**-0.5

        self.qkv_proj = _linear(
            arch.hidden_size, 3 * self.inner_dim, bias=False, dtype=_BF16
        )
        self.q_norm = _norm(self.head_dim, eps=arch.qk_norm_eps)
        self.k_norm = _norm(self.head_dim, eps=arch.qk_norm_eps)
        self.out_proj = _linear(
            self.inner_dim, arch.hidden_size, bias=False, dtype=_BF16
        )

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        pad_from: int | None = None,
    ) -> torch.Tensor:
        """Non-causal varlen attention over the packed sequence.

        Backend choice lives in :mod:`atom.diffusion.attention`; ASM is
        the default and Triton is what reproduces the sglang reference exactly.
        """
        return packed_varlen_attention(
            q,
            k,
            v,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            pad_from=pad_from,
            softmax_scale=self.softmax_scale,
            backend=self.attn_backend,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        pad_from: int | None = None,
        ulysses: UlyssesGroup,
    ) -> torch.Tensor:
        """[T_local, H] -> [T_local, H].

        qkv / QK-Norm / RoPE run on this rank's rows with all heads; the
        all-to-all then trades sequence for heads so attention sees the whole
        packed sequence with ``num_heads / world`` local heads. ``cu_seqlens``
        therefore keeps global packed-document semantics on every rank.
        """
        total = x.shape[0]
        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.inner_dim, dim=-1)
        q = q.view(total, self.num_heads, self.head_dim)
        k = k.view(total, self.num_heads, self.head_dim)
        v = v.view(total, self.num_heads, self.head_dim)

        q, k = _qk_norm_rope(q, k, self.q_norm, self.k_norm, rope_cache)

        if ulysses.enabled:
            q = ulysses.scatter_heads(q)
            k = ulysses.scatter_heads(k)
            v = ulysses.scatter_heads(v)

        out = self._attend(
            q, k, v, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen, pad_from=pad_from
        )

        if ulysses.enabled:
            out = ulysses.gather_heads(out)

        return self.out_proj(out.reshape(total, self.num_heads * self.head_dim))


class MiniMaxH3MLP(nn.Module):
    """Fused gate/up projection, SiLU-gated, then down projection."""

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        self.fc1 = _linear(
            arch.hidden_size, 2 * arch.ffn_hidden_size, bias=False, dtype=_BF16
        )
        self.fc2 = _linear(
            arch.ffn_hidden_size, arch.hidden_size, bias=False, dtype=_BF16
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(_silu_mul(self.fc1(x)))


class MiniMaxH3AdalnProj(nn.Module):
    """Zero-init projection from the timestep embedding to AdaLN vectors.

    Per block: [M, t_dim] -> [M, 3*6H] -> view(M*3, 6H) -> chunk(6).
    Final layer: one modality, expand_ratio 2.
    """

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        out_features: int,
        *,
        expand_ratio: int,
        modality_num: int,
    ) -> None:
        super().__init__()
        if out_features != expand_ratio * arch.hidden_size * modality_num:
            raise ValueError(
                f"adaln out_features mismatch: {out_features} != "
                f"{expand_ratio}*{arch.hidden_size}*{modality_num}"
            )
        self.expand_ratio = expand_ratio
        self.modality_num = modality_num
        self.hidden_size = arch.hidden_size
        self.linear = _linear(arch.time_embed_dim, out_features, bias=True, dtype=_BF16)

    def forward(self, adaln_input: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.linear(adaln_input)
        m = x.shape[0]
        x = x.view(m * self.modality_num, self.expand_ratio * self.hidden_size)
        return tuple(x.chunk(self.expand_ratio, dim=-1))


class MiniMaxH3TokenRefinerBlock(nn.Module):
    """Pre-norm transformer block, no AdaLN and no RoPE."""

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        *,
        attn_backend: "AttentionBackend | str | None" = None,
    ) -> None:
        super().__init__()
        self.norm1 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.attn = MiniMaxH3Attention(arch, attn_backend=attn_backend)
        self.mlp = MiniMaxH3MLP(arch)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        pad_from: int | None = None,
        ulysses: UlyssesGroup,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            rope_cache=None,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            pad_from=pad_from,
            ulysses=ulysses,
        )
        return x + self.mlp(self.norm2(x))


class MiniMaxH3TokenRefiner(nn.Module):
    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        *,
        attn_backend: "AttentionBackend | str | None" = None,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            MiniMaxH3TokenRefinerBlock(arch, attn_backend=attn_backend)
            for _ in range(arch.token_refiner_num_layers)
        )
        self.final_norm = _norm(arch.hidden_size, eps=arch.final_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        pad_from: int | None = None,
        ulysses: UlyssesGroup,
    ) -> torch.Tensor:
        for block in self.blocks:
            x = block(
                x,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                pad_from=pad_from,
                ulysses=ulysses,
            )
        return self.final_norm(x)


class MiniMaxH3DiTBlock(nn.Module):
    """norm -> AdaLN scale/shift -> attention -> gated residual, then MLP."""

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig,
        *,
        attn_backend: "AttentionBackend | str | None" = None,
    ) -> None:
        super().__init__()
        self.norm1 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.norm2 = _norm(arch.hidden_size, eps=arch.norm_eps)
        self.attn = MiniMaxH3Attention(arch, attn_backend=attn_backend)
        self.mlp = MiniMaxH3MLP(arch)
        self.adaln_proj = MiniMaxH3AdalnProj(
            arch,
            arch.adaln_out_features,
            expand_ratio=6,
            modality_num=MINIMAX_H3_ADALN_MODALITY_NUM,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        combined_indices: torch.Tensor,
        rope_cache: torch.Tensor | None,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        pad_from: int | None = None,
        ulysses: UlyssesGroup,
        adaln_params: tuple[torch.Tensor, ...] | None = None,
    ) -> torch.Tensor:
        if adaln_params is None:
            adaln_params = self.adaln_proj(adaln_input)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaln_params

        residual = x
        h = _norm_modulate(self.norm1, x, shift_msa, scale_msa, combined_indices)
        h = self.attn(
            h,
            rope_cache=rope_cache,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            pad_from=pad_from,
            ulysses=ulysses,
        )
        x = _modulate_gate(residual, gate_msa, h, combined_indices, dtype=_BF16)

        residual = x
        h = _norm_modulate(self.norm2, x, shift_mlp, scale_mlp, combined_indices)
        h = self.mlp(h)
        return _modulate_gate(residual, gate_mlp, h, combined_indices, dtype=_BF16)


class MiniMaxH3FinalLayer(nn.Module):
    """Single-modality AdaLN, then the fp32 video and audio output heads."""

    def __init__(self, arch: MiniMaxH3DiTArchConfig) -> None:
        super().__init__()
        self.norm = _norm(arch.hidden_size, eps=arch.final_norm_eps)
        self.adaln_proj = MiniMaxH3AdalnProj(
            arch,
            arch.final_adaln_out_features,
            expand_ratio=2,
            modality_num=1,
        )
        self.video_out = _linear(
            arch.hidden_size, arch.video_patch_dim, bias=True, dtype=_FP32
        )
        self.audio_out = _linear(
            arch.hidden_size, arch.audio_latents_dim, bias=True, dtype=_FP32
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        adaln_input: torch.Tensor,
        inverse_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shift, scale = self.adaln_proj(adaln_input)
        h = _modulate_scale_shift(
            self.norm(x), shift, scale, inverse_indices, dtype=_BF16
        )
        h = h.to(_FP32)
        return self.video_out(h), self.audio_out(h)


class MiniMaxH3DiTModel(nn.Module):
    """MiniMax-H3 DiT for ATOM."""

    def __init__(
        self,
        arch: MiniMaxH3DiTArchConfig | None = None,
        ulysses: UlyssesGroup | None = None,
        *,
        attn_backend: "AttentionBackend | str | None" = None,
    ) -> None:
        super().__init__()
        self.arch = arch or MiniMaxH3DiTArchConfig()
        self.ulysses = ulysses or UlyssesGroup()
        self.arch.validate_ulysses(self.ulysses.world_size)
        # Resolved once here so every block reports the same backend, and so a
        # parity run can pin Triton without touching each submodule.
        self.attn_backend = resolve_attention_backend(attn_backend)
        attn_backend = self.attn_backend

        a = self.arch
        self.hidden_size = a.hidden_size

        self.video_patch_proj = _linear(
            a.video_patch_dim, a.hidden_size, bias=True, dtype=_FP32
        )
        self.audio_patch_proj = _linear(
            a.audio_latents_dim, a.hidden_size, bias=True, dtype=_FP32
        )
        self.condition_proj = _linear(a.text_dim, a.hidden_size, bias=True, dtype=_BF16)
        self.time_embedder = MiniMaxH3TimeEmbedder(a)
        self.rope = MiniMaxH3Rope(a.rope_inv_freq_len)
        self.token_refiner = MiniMaxH3TokenRefiner(a, attn_backend=attn_backend)
        self.blocks = nn.ModuleList(
            MiniMaxH3DiTBlock(a, attn_backend=attn_backend) for _ in range(a.num_layers)
        )
        self.final_layer = MiniMaxH3FinalLayer(a)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _pos_ids(pos_info: Any, key: str) -> torch.Tensor:
        if torch.is_tensor(pos_info):
            return pos_info.view(-1).to(torch.long)
        if isinstance(pos_info, dict) and "position_ids" in pos_info:
            return pos_info["position_ids"].view(-1).to(torch.long)
        raise TypeError(f"{key} must be a tensor or carry 'position_ids'")

    @staticmethod
    def _psp(psp: Any, field: str) -> Any:
        if isinstance(psp, dict):
            if field not in psp:
                raise KeyError(f"packed_seq_params missing {field!r}")
            return psp[field]
        if not hasattr(psp, field):
            raise AttributeError(f"packed_seq_params missing {field!r}")
        return getattr(psp, field)

    def refine_prompt_embeds(
        self,
        prompt_embeds: torch.Tensor,
        refiner_cu_seqlens: torch.Tensor,
        *,
        device: torch.device,
    ) -> torch.Tensor:
        """Project raw text embeddings and run the 2-layer token refiner."""
        text_rows = prompt_embeds.to(device=device, dtype=_BF16)
        text_embed = self.condition_proj(text_rows)
        cu = refiner_cu_seqlens.to(device=device, dtype=torch.int32)
        max_seqlen = int((cu[1:] - cu[:-1]).max().item()) if cu.numel() > 1 else 0
        # The refiner runs over the (short) text sequence on every rank; it is
        # replicated, never Ulysses-split.
        return self.token_refiner(
            text_embed,
            cu_seqlens=cu,
            max_seqlen=max_seqlen,
            ulysses=UlyssesGroup(),
        )

    def build_rope_cache(
        self, img_position_ids: torch.Tensor, row_start: int, row_stop: int
    ) -> torch.Tensor:
        """cos|sin cache for this rank's row shard, [row_stop-row_start, 96]."""
        freqs = self.rope(img_position_ids[:, row_start:row_stop])
        return rope_cos_sin_cache(freqs, dtype=_BF16)

    def _embed(
        self,
        *,
        x: torch.Tensor,
        audio_x: torch.Tensor,
        text_embed: torch.Tensor,
        unique_timesteps: torch.Tensor,
        layout: dict[str, Any],
        local_seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scatter text / video / audio embeddings into this rank's rows."""
        embeddings = torch.zeros(
            (local_seq_len, self.hidden_size), device=device, dtype=_BF16
        )

        text_start = int(layout["text_source_start"])
        text_stop = int(layout["text_source_stop"])
        text_rows = text_stop - text_start
        if text_rows:
            embeddings[:text_rows].copy_(text_embed[text_start:text_stop])

        img_global = layout["img_global_ids"].to(device)
        img_rows = layout["img_row_ids"].to(device)
        if img_rows.numel():
            x_rows = x.view(-1, x.shape[-1]).index_select(0, img_global).to(_FP32)
            embeddings.index_copy_(0, img_rows, self.video_patch_proj(x_rows).to(_BF16))

        audio_global = layout["audio_global_ids"].to(device)
        audio_rows = layout["audio_row_ids"].to(device)
        if audio_rows.numel():
            a_rows = (
                audio_x.view(-1, audio_x.shape[-1])
                .index_select(0, audio_global)
                .to(_FP32)
            )
            embeddings.index_copy_(
                0, audio_rows, self.audio_patch_proj(a_rows).to(_BF16)
            )

        return embeddings, self.time_embedder(unique_timesteps)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        x = kwargs["x"]
        audio_x = kwargs["audio_x"]
        device = x.device

        cu_seqlens = self._psp(kwargs["packed_seq_params"], "cu_seqlens_q").to(
            device=device, dtype=torch.int32
        )
        max_seqlen = int(self._psp(kwargs["packed_seq_params"], "max_seqlen_q"))
        # Trailing alignment padding, if the caller declared it. Dropping those
        # rows halves the attention grid; see packed_varlen_attention.
        psp = kwargs["packed_seq_params"]
        pad_from = psp.get("used_len") if isinstance(psp, dict) else None
        pad_from = None if pad_from is None else int(pad_from)
        refiner_cu = self._psp(kwargs["refiner_packed_seq_params"], "cu_seqlens_q")

        world = self.ulysses.world_size
        rank = self.ulysses.rank
        seq_len = int(x.shape[1])
        if seq_len % world:
            raise ValueError(
                f"packed sequence ({seq_len}) must divide across the ulysses "
                f"world size ({world})"
            )
        local_seq_len = seq_len // world
        row_start = rank * local_seq_len
        row_stop = row_start + local_seq_len

        # Text refinement is request-static: the caller may hand back an
        # already-refined tensor to keep it out of the denoise hot loop.
        refined_len = kwargs.get("refined_prompt_embeds_length")
        prompt_embeds = kwargs["prompt_embeds"]
        if refined_len is not None:
            text_len = int(
                refined_len.item() if torch.is_tensor(refined_len) else refined_len
            )
            text_embed = prompt_embeds[:text_len].to(device=device, dtype=_BF16)
            if int(text_embed.shape[-1]) != self.hidden_size:
                raise ValueError(
                    f"refined prompt embeddings must be {self.hidden_size} wide, "
                    f"got {int(text_embed.shape[-1])}"
                )
        else:
            text_embed = self.refine_prompt_embeds(
                prompt_embeds, refiner_cu, device=device
            )

        rope_cache = kwargs.get("rope_cache")
        if rope_cache is None:
            rope_cache = self.build_rope_cache(
                kwargs["img_position_ids"].to(device), row_start, row_stop
            )
        elif isinstance(rope_cache, (tuple, list)):
            # The serving contract passes (cos_sin_cache, positions); the eager
            # RoPE path indexes nothing, so the positions half is unused here.
            rope_cache = rope_cache[0]
        rope_cache = rope_cache.to(device=device, dtype=_BF16)

        layout = kwargs.get("local_embedding_layout")
        if layout is None:
            raise KeyError(
                "local_embedding_layout is required; ATOM's pipeline builds it "
                "in the packed-sequence stage"
            )

        hidden, t_emb = self._embed(
            x=x.to(device),
            audio_x=audio_x.to(device),
            text_embed=text_embed,
            unique_timesteps=kwargs["unique_timesteps"].view(-1).to(device),
            layout=layout,
            local_seq_len=local_seq_len,
            device=device,
        )
        adaln_input = nn.functional.silu(t_emb).to(_BF16)

        inverse_indices = kwargs["inverse_indices"].view(-1).to(device).long()
        block_inverse = inverse_indices[row_start:row_stop]

        block_combined = kwargs.get("block_combined_indices")
        if block_combined is None:
            block_tags = kwargs.get("block_token_tags")
            if block_tags is None:
                block_tags = (
                    kwargs["token_tags"].view(-1).to(device).long()[row_start:row_stop]
                )
            block_combined = torch.add(
                block_tags.to(device).long().clamp(min=0),
                block_inverse,
                alpha=MINIMAX_H3_ADALN_MODALITY_NUM,
            )
        block_combined = block_combined.to(device).long()

        for block in self.blocks:
            hidden = block(
                hidden,
                adaln_input=adaln_input,
                combined_indices=block_combined,
                rope_cache=rope_cache,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                pad_from=pad_from,
                ulysses=self.ulysses,
            )

        video_logits, audio_logits = self.final_layer(
            hidden, adaln_input=adaln_input, inverse_indices=block_inverse
        )

        if world > 1:
            # Rows are rank-local through the whole stack; gather before
            # selecting the live media rows, which are global indices.
            video_width = video_logits.shape[-1]
            merged = torch.cat((video_logits, audio_logits), dim=-1)
            gathered = [torch.empty_like(merged) for _ in range(world)]
            torch.distributed.all_gather(gathered, merged.contiguous())
            merged = torch.cat(gathered, dim=0)
            video_logits, audio_logits = merged.split(
                (video_width, merged.shape[-1] - video_width), dim=-1
            )

        infer_out_pos = self._pos_ids(
            kwargs["img_pos_for_infer_output_info"], "img_pos_for_infer_output_info"
        ).to(device)
        audio_pos = self._pos_ids(kwargs["audio_pos_info"], "audio_pos_info").to(device)
        video_logits = video_logits.index_select(0, infer_out_pos)
        audio_logits = audio_logits.index_select(0, audio_pos)

        if not bool(kwargs.get("skip_mask_out_condition", False)):
            update_mask = kwargs["update_mask"].view(-1).to(device)
            if update_mask.shape[0] != video_logits.shape[0]:
                raise ValueError(
                    f"update_mask length {update_mask.shape[0]} != video rows "
                    f"{video_logits.shape[0]}"
                )
            video_logits = video_logits * update_mask.unsqueeze(-1)
            update_audio_mask = kwargs.get("update_audio_mask")
            if update_audio_mask is not None:
                audio_logits = audio_logits * update_audio_mask.view(-1).unsqueeze(-1)

        return video_logits, audio_logits
