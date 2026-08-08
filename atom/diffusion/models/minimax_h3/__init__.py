# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-H3 specific pipeline stages and helpers."""

from atom.diffusion.models.minimax_h3.condition_noise import (
    MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
    audio_cond_noise_aug_rows,
    imgvid_cond_noise_aug_rows,
)
from atom.diffusion.models.minimax_h3.denoise import (
    build_timestep_conditioning,
    run_denoise_loop,
)
from atom.diffusion.models.minimax_h3.geometry import (
    MiniMaxH3Geometry,
    align_frame_count,
    audio_latent_t,
    frame_count_from_video_latent_t,
    time_shift_sigmas,
    video_latent_t,
)
from atom.diffusion.models.minimax_h3.keyframe import (
    cover_crop_plan,
    encode_keyframe_cond_rows,
    prepare_keyframe_canvas,
    stretch_keyframe_canvas,
)
from atom.diffusion.models.minimax_h3.latent_prep import (
    build_initial_latents,
    scatter_rows_into_packed,
)
from atom.diffusion.models.minimax_h3.packed_sequence import (
    FL2VA_KEYFRAME_SIGNATURES,
    build_local_embedding_layout,
    build_packed_sequence,
    build_packed_sequence_ref2va,
    build_packed_sequence_t2va,
    resolve_keyframe_indices,
    temporal_position_span,
    validate_keyframe_signature,
)
from atom.diffusion.models.minimax_h3.packed_tokens import (
    patchify_video_latent,
    unpack_audio_tokens,
    unpatchify_video_tokens,
)

__all__ = [
    "FL2VA_KEYFRAME_SIGNATURES",
    "MINIMAX_H3_AUDIO_REF_COND_TIMESTEP",
    "MINIMAX_H3_IMGVID_COND_TIMESTEP",
    "MiniMaxH3Geometry",
    "align_frame_count",
    "audio_cond_noise_aug_rows",
    "audio_latent_t",
    "build_initial_latents",
    "build_local_embedding_layout",
    "build_packed_sequence",
    "build_packed_sequence_ref2va",
    "build_packed_sequence_t2va",
    "build_timestep_conditioning",
    "cover_crop_plan",
    "encode_keyframe_cond_rows",
    "frame_count_from_video_latent_t",
    "imgvid_cond_noise_aug_rows",
    "patchify_video_latent",
    "prepare_keyframe_canvas",
    "resolve_keyframe_indices",
    "run_denoise_loop",
    "scatter_rows_into_packed",
    "stretch_keyframe_canvas",
    "temporal_position_span",
    "time_shift_sigmas",
    "unpack_audio_tokens",
    "unpatchify_video_tokens",
    "validate_keyframe_signature",
    "video_latent_t",
]
