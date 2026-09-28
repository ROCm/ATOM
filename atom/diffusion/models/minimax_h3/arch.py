# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-H3 DiT architecture config.

Values mirror the released FL2VA/Ref2VA checkpoints. Verified against a live
forward on 8x MI308X: 37,760 packed tokens in 2 segments (37,712 + 48 pad) for
a 1344x768x124f request.
"""

from dataclasses import dataclass

# The packed sequence is padded up to a multiple of this before attention.
# It also bounds the usable Ulysses degree: the sequence must divide evenly
# across ranks, so degrees above 64 cannot work regardless of head count.
MINIMAX_H3_PACKED_SEQUENCE_ALIGNMENT = 64

# AdaLN conditions on three token modalities (text, audio, video).
MINIMAX_H3_ADALN_MODALITY_NUM = 3

# Parameters the checkpoint stores -- and the reference implementation runs --
# in fp32 rather than bf16. Patch/unpatch projections and the timestep
# embedding are numerically sensitive; demoting them to bf16 shifts output.
MINIMAX_H3_FP32_PARAM_NAMES = frozenset(
    {
        "video_patch_proj.weight",
        "video_patch_proj.bias",
        "audio_patch_proj.weight",
        "audio_patch_proj.bias",
        "time_embedder.proj_in.weight",
        "time_embedder.proj_in.bias",
        "time_embedder.proj_out.weight",
        "time_embedder.proj_out.bias",
        "final_layer.video_out.weight",
        "final_layer.video_out.bias",
        "final_layer.audio_out.weight",
        "final_layer.audio_out.bias",
    }
)

# Buffers kept in fp32 for the same reason.
MINIMAX_H3_FP32_BUFFER_NAMES = frozenset({"rope.inv_freq"})


@dataclass
class MiniMaxH3DiTArchConfig:
    """Static architecture of the MiniMax-H3 audio-video DiT."""

    num_layers: int = 50
    token_refiner_num_layers: int = 2
    hidden_size: int = 5376
    num_attention_heads: int = 56
    attention_head_dim: int = 128
    ffn_hidden_size: int = 14336

    latents_dim: int = 24
    audio_latents_dim: int = 32
    patch_size: tuple[int, int, int] = (1, 2, 2)
    text_dim: int = 5120

    timestep_input_dim: int = 256
    time_embed_hidden_size: int = 5376
    time_embed_dim: int = 2688

    # 18 = 6 modulation vectors (shift/scale/gate for attn and mlp) x 3
    # modalities; the final layer uses 2 x 1 modality.
    adaln_out_features: int = 18 * 5376
    final_adaln_out_features: int = 2 * 5376

    # 16 frequencies per axis over (t, h, w), concatenated twice -> 96 of the
    # 128 head dims rotate (rotary_percent 0.75).
    rope_inv_freq_len: int = 16

    norm_eps: float = 1e-5
    qk_norm_eps: float = 1e-5
    final_norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        if isinstance(self.patch_size, list):
            self.patch_size = tuple(self.patch_size)
        if len(self.patch_size) != 3:
            raise ValueError(f"patch_size must have 3 values, got {self.patch_size}")

        expected_adaln = 18 * self.hidden_size
        if self.adaln_out_features != expected_adaln:
            raise ValueError(
                f"adaln_out_features must be 18*hidden_size ({expected_adaln}), "
                f"got {self.adaln_out_features}"
            )
        if self.final_adaln_out_features != 2 * self.hidden_size:
            raise ValueError(
                f"final_adaln_out_features must be 2*hidden_size "
                f"({2 * self.hidden_size}), got {self.final_adaln_out_features}"
            )
        if self.num_attention_heads * self.attention_head_dim != self.inner_dim:
            raise ValueError("inner_dim must equal num_heads * head_dim")

    @property
    def inner_dim(self) -> int:
        return self.num_attention_heads * self.attention_head_dim

    @property
    def rope_dim(self) -> int:
        """Rotated head dims: (t, h, w) x inv_freq_len, doubled."""
        return 6 * self.rope_inv_freq_len

    @property
    def video_patch_dim(self) -> int:
        pt, ph, pw = self.patch_size
        return self.latents_dim * pt * ph * pw

    def validate_ulysses(self, world_size: int) -> None:
        """Check this architecture can be split across ``world_size`` ranks."""
        if self.num_attention_heads % world_size:
            raise ValueError(
                f"num_attention_heads ({self.num_attention_heads}) must be "
                f"divisible by ulysses world size ({world_size})"
            )
