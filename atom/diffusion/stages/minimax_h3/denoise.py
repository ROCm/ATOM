# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-H3 denoise loop.

State stays in row form -- video ``[Nv, 96]``, audio ``[Na, 32]`` -- and is
scattered into the padded ``[1, S, *]`` buffers each step, so the sampler never
sees the packed layout.

Video and audio run separate sigma schedules, so each step builds a per-token
timestep vector: AdaLN conditions on ``token_tag + 3 * inverse_index`` into that
step's unique timesteps. Step 0 has one (both modalities at t=0), later steps
two, and conditioned requests a third for the conditioning rows.
"""

from collections.abc import Callable

import torch

from atom.diffusion.configs.minimax_h3 import MINIMAX_H3_ADALN_MODALITY_NUM
from atom.diffusion.models.schedulers.euler_ancestral_h3 import (
    MiniMaxH3EulerAncestralEta0Scheduler,
)
from atom.diffusion.stages.minimax_h3.condition_noise import (
    MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
)
from atom.diffusion.stages.minimax_h3.latent_prep import scatter_rows_into_packed
from atom.diffusion.stages.minimax_h3.packed_sequence import (
    build_local_embedding_layout,
)

# Conditioning rows ride max(video_timestep, noise_aug) -- the same coefficient
# they were mixed with, so value and timestep agree. The max never binds on the
# released 50-step schedules (video tops out at 0.8, audio 0.941).


def build_timestep_conditioning(
    *,
    token_tags: torch.Tensor,
    img_pos: torch.Tensor,
    audio_pos: torch.Tensor,
    video_timestep: float,
    audio_timestep: float,
    cond_pos: torch.Tensor | None = None,
    condition_timestep: float = MINIMAX_H3_IMGVID_COND_TIMESTEP,
    cond_audio_pos: torch.Tensor | None = None,
    audio_condition_timestep: float = MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(unique_timesteps, inverse_indices, combined_indices)``.

    Text and padding ride the video timestep -- they carry no latent, so the
    choice only has to keep them inside the unique set. ``cond_pos`` marks
    visual conditioning, ``cond_audio_pos`` audio references (a different
    constant: they are not noise-augmented).
    """
    seq_len = int(token_tags.shape[0])
    per_token = torch.full(
        (seq_len,), float(video_timestep), dtype=torch.float32, device=device
    )
    per_token.index_fill_(0, audio_pos.to(device), float(audio_timestep))
    # img_pos is already the video timestep; index_fill on it would be a no-op.
    del img_pos
    if cond_pos is not None and int(cond_pos.numel()):
        per_token.index_fill_(
            0,
            cond_pos.to(device),
            max(float(video_timestep), float(condition_timestep)),
        )
    if cond_audio_pos is not None and int(cond_audio_pos.numel()):
        per_token.index_fill_(
            0,
            cond_audio_pos.to(device),
            max(float(audio_timestep), float(audio_condition_timestep)),
        )

    unique, inverse = torch.unique(per_token, sorted=True, return_inverse=True)
    combined = token_tags.to(device).clamp(min=0) + MINIMAX_H3_ADALN_MODALITY_NUM * (
        inverse.to(torch.long)
    )
    return unique, inverse.to(torch.long), combined


def run_denoise_loop(
    *,
    dit: Callable[..., tuple[torch.Tensor, torch.Tensor]],
    video_rows: torch.Tensor,
    audio_rows: torch.Tensor,
    packed: dict,
    cond_rows: torch.Tensor | None = None,
    cond_audio_rows: torch.Tensor | None = None,
    video_sigmas: list[float],
    audio_sigmas: list[float],
    rank_slice: tuple[int, int],
    device: torch.device | str = "cpu",
    prompt_embeds: torch.Tensor,
    refined_prompt_embeds_length: int,
    rope_cache: torch.Tensor,
    scheduler: MiniMaxH3EulerAncestralEta0Scheduler | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the loop and return final ``(video_rows, audio_rows)``.

    Runs ``len(sigmas) - 1`` steps. Conditioning rows head their region and are
    re-scattered unchanged each step, but the DiT is asymmetric: it returns only
    the *generated* video rows yet *all* audio rows, so the audio references
    must be trimmed off the prediction.
    """
    if len(video_sigmas) != len(audio_sigmas):
        raise ValueError(
            f"sigma schedules must be the same length, got "
            f"{len(video_sigmas)} and {len(audio_sigmas)}"
        )
    if len(video_sigmas) < 2:
        raise ValueError("need at least two sigmas to take a step")

    scheduler = scheduler or MiniMaxH3EulerAncestralEta0Scheduler()
    seq_len = int(packed["seq_len"])
    img_pos = packed["img_pos"].to(device)
    n_cond = int(packed.get("cond_rows", 0) or 0)
    if n_cond and cond_rows is None:
        raise ValueError(
            f"packed layout reserves {n_cond} conditioning rows but no "
            "cond_rows tensor was supplied"
        )
    if cond_rows is not None:
        if not n_cond:
            raise ValueError(
                "cond_rows supplied but the packed layout reserves none; the "
                "sequence must be built with keyframe_frame_indices"
            )
        if int(cond_rows.shape[0]) != n_cond:
            raise ValueError(
                f"cond_rows has {int(cond_rows.shape[0])} rows, layout expects "
                f"{n_cond}"
            )
        cond_rows = cond_rows.to(device)
    cond_pos = img_pos[:n_cond] if n_cond else None
    output_img_pos = img_pos[n_cond:]

    n_cond_audio = int(packed.get("cond_audio_rows", 0) or 0)
    if n_cond_audio and cond_audio_rows is None:
        raise ValueError(
            f"packed layout reserves {n_cond_audio} audio reference rows but "
            "no cond_audio_rows tensor was supplied"
        )
    if cond_audio_rows is not None:
        if not n_cond_audio:
            raise ValueError(
                "cond_audio_rows supplied but the packed layout reserves none"
            )
        if int(cond_audio_rows.shape[0]) != n_cond_audio:
            raise ValueError(
                f"cond_audio_rows has {int(cond_audio_rows.shape[0])} rows, "
                f"layout expects {n_cond_audio}"
            )
        cond_audio_rows = cond_audio_rows.to(device)
    audio_pos = packed["audio_pos"].to(device)
    cond_audio_pos = audio_pos[:n_cond_audio] if n_cond_audio else None
    token_tags = packed["token_tags"].to(device)
    cu_seqlens = packed["cu_seqlens"].to(device)
    img_position_ids = packed["img_position_ids"]
    row_start, row_stop = rank_slice

    video_rows = video_rows.to(device)
    audio_rows = audio_rows.to(device)
    max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

    layout = build_local_embedding_layout(
        img_pos=packed["img_pos"],
        audio_pos=packed["audio_pos"],
        text_pos=packed["text_pos"],
        row_start=row_start,
        row_stop=row_stop,
    )

    num_steps = len(video_sigmas) - 1
    for step in range(num_steps):
        v_sig, v_next = video_sigmas[step], video_sigmas[step + 1]
        a_sig, a_next = audio_sigmas[step], audio_sigmas[step + 1]
        v_t, a_t = 1.0 - v_sig, 1.0 - a_sig

        unique_t, inverse, combined = build_timestep_conditioning(
            token_tags=token_tags,
            img_pos=img_pos,
            audio_pos=audio_pos,
            video_timestep=v_t,
            audio_timestep=a_t,
            cond_pos=cond_pos,
            cond_audio_pos=cond_audio_pos,
            device=device,
        )

        x, audio_x = scatter_rows_into_packed(
            video_rows=(
                torch.cat((cond_rows, video_rows), dim=0)
                if cond_rows is not None
                else video_rows
            ),
            audio_rows=(
                torch.cat((cond_audio_rows, audio_rows), dim=0)
                if cond_audio_rows is not None
                else audio_rows
            ),
            img_pos=img_pos,
            audio_pos=audio_pos,
            seq_len=seq_len,
        )

        pred_video, pred_audio = dit(
            x=x,
            audio_x=audio_x,
            img_position_ids=img_position_ids.to(device),
            unique_timesteps=unique_t,
            inverse_indices=inverse,
            block_combined_indices=combined[row_start:row_stop],
            update_mask=packed["update_mask"].to(device),
            prompt_embeds=prompt_embeds,
            refined_prompt_embeds_length=refined_prompt_embeds_length,
            rope_cache=rope_cache,
            packed_seq_params={
                "cu_seqlens_q": cu_seqlens,
                "max_seqlen_q": max_seqlen,
            },
            refiner_packed_seq_params={
                "cu_seqlens_q": torch.tensor(
                    [0, refined_prompt_embeds_length], dtype=torch.int32, device=device
                ),
                "max_seqlen_q": refined_prompt_embeds_length,
            },
            local_embedding_layout=layout,
            img_pos_info={"position_ids": img_pos},
            audio_pos_info={"position_ids": audio_pos},
            text_pos_info={"position_ids": packed["text_pos"].to(device)},
            img_pos_for_infer_output_info={"position_ids": output_img_pos},
            skip_mask_out_condition=True,
        )

        if n_cond_audio:
            # The DiT predicts the reference audio rows too; drop them rather
            # than stepping them, or the references drift as the sample evolves.
            pred_audio = pred_audio[n_cond_audio:]

        video_rows, audio_rows = scheduler.step(
            visual_latent=video_rows,
            audio_latent=audio_rows,
            noise_pred_visual=pred_video.to(video_rows.dtype),
            noise_pred_audio=pred_audio.to(audio_rows.dtype),
            video_timestep=torch.tensor([v_t], dtype=torch.float32, device=device),
            audio_timestep=torch.tensor([a_t], dtype=torch.float32, device=device),
            video_sigma_curr=v_sig,
            video_sigma_next=v_next,
            audio_sigma_curr=a_sig,
            audio_sigma_next=a_next,
        )

        if progress is not None:
            progress(step + 1, num_steps)

    return video_rows, audio_rows
