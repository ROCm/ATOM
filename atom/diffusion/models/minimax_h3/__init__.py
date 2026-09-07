# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-H3 specific pipeline stages and helpers."""

from atom.diffusion.models.minimax_h3.conditioning import (
    MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
    cover_crop_plan,
    encode_keyframe_cond_rows,
    imgvid_cond_noise_aug_rows,
    prepare_keyframe_canvas,
    stretch_keyframe_canvas,
)
from atom.diffusion.models.minimax_h3.denoise import (
    build_timestep_conditioning,
    run_denoise_loop,
)
from atom.diffusion.models.minimax_h3.layout import (
    FL2VA_KEYFRAME_SIGNATURES,
    MiniMaxH3Geometry,
    align_frame_count,
    audio_latent_t,
    build_initial_latents,
    build_local_embedding_layout,
    build_packed_sequence,
    build_packed_sequence_ref2va,
    patchify_video_latent,
    resolve_keyframe_indices,
    scatter_rows_into_packed,
    temporal_position_span,
    time_shift_sigmas,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    validate_keyframe_signature,
    video_latent_t,
)

__all__ = [
    "FL2VA_KEYFRAME_SIGNATURES",
    "MINIMAX_H3_AUDIO_REF_COND_TIMESTEP",
    "MINIMAX_H3_IMGVID_COND_TIMESTEP",
    "MiniMaxH3Geometry",
    "align_frame_count",
    "audio_latent_t",
    "build_initial_latents",
    "build_local_embedding_layout",
    "build_packed_sequence",
    "build_packed_sequence_ref2va",
    "build_timestep_conditioning",
    "cover_crop_plan",
    "encode_keyframe_cond_rows",
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
