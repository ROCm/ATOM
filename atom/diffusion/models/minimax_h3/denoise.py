# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-H3 denoise loop and its rectified-flow sampler."""

import math
from collections.abc import Callable

import torch

from atom.diffusion.models.minimax_h3.arch import MINIMAX_H3_ADALN_MODALITY_NUM
from atom.diffusion.models.minimax_h3.conditioning import (
    MINIMAX_H3_AUDIO_REF_COND_TIMESTEP,
    MINIMAX_H3_IMGVID_COND_TIMESTEP,
)
from atom.diffusion.models.minimax_h3.layout import (
    build_local_embedding_layout,
    scatter_rows_into_packed,
)


def _check_finite(t: torch.Tensor, name: str) -> None:
    if not bool(torch.isfinite(t).all().item()):
        raise ValueError(f"{name} must be finite")


def _check_unit_timestep(t: torch.Tensor, name: str) -> None:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not torch.is_floating_point(t):
        raise ValueError(f"{name} must be floating point")
    _check_finite(t, name)
    if bool(((t < 0) | (t > 1)).any().item()):
        raise ValueError(f"{name} must lie in [0, 1]")


def _check_sigma(value: float, name: str) -> float:
    sigma = float(value)
    if not math.isfinite(sigma):
        raise ValueError(f"{name} must be finite")
    if sigma < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return sigma


def _check_timestep_sigma_pair(
    timestep: torch.Tensor, sigma_curr: float, name: str
) -> float:
    """Enforce sigma == 1 - t.

    Cheap, and it catches the failure mode that matters: a schedule and a
    timestep drifting out of step produces a plausible video that is subtly
    wrong rather than an error.
    """
    _check_unit_timestep(timestep, f"{name}_timestep")
    sigma = _check_sigma(sigma_curr, f"{name}_sigma_curr")
    expected = 1.0 - timestep.detach().to(dtype=torch.float32)
    if not torch.allclose(
        torch.full_like(expected, sigma), expected, rtol=1e-5, atol=1e-5
    ):
        raise ValueError(f"{name}_sigma_curr must equal 1 - {name}_timestep")
    return sigma


def minimax_h3_rf_v_to_x0(
    xt: torch.Tensor, v: torch.Tensor, timestep: torch.Tensor
) -> torch.Tensor:
    """Rectified-flow velocity -> clean sample: ``x0 = xt + (1 - t) * v``."""
    if xt.shape != v.shape:
        raise ValueError(f"xt and v shapes must match: {xt.shape} vs {v.shape}")
    if not torch.is_floating_point(xt) or not torch.is_floating_point(v):
        raise ValueError("xt and v must be floating point")
    _check_finite(xt, "xt")
    _check_finite(v, "v")
    _check_unit_timestep(timestep, "timestep")

    cond_t = timestep.to(device=xt.device, dtype=xt.dtype)
    while cond_t.ndim < xt.ndim:
        cond_t = cond_t.unsqueeze(-1)
    out = xt + (1 - cond_t) * v
    _check_finite(out, "x0")
    return out


def minimax_h3_euler_eta0_step(
    state: torch.Tensor,
    denoised: torch.Tensor,
    *,
    sigma_curr: float,
    sigma_next: float,
) -> torch.Tensor:
    """One eta=0 ancestral Euler step."""
    if state.shape != denoised.shape:
        raise ValueError(
            f"state and denoised shapes must match: {state.shape} vs {denoised.shape}"
        )
    if not torch.is_floating_point(state) or not torch.is_floating_point(denoised):
        raise ValueError("state and denoised must be floating point")
    _check_finite(state, "state")
    _check_finite(denoised, "denoised")
    sigma_curr = _check_sigma(sigma_curr, "sigma_curr")
    sigma_next = _check_sigma(sigma_next, "sigma_next")
    if sigma_curr == 0.0 and sigma_next != 0.0:
        raise ValueError("sigma_next must be 0 when sigma_curr is 0")

    if sigma_curr == 0.0:
        return state

    # Accumulate the interpolation in fp32 for reduced-precision states, then
    # cast back; doing it in bf16 loses meaningful precision over 50 steps.
    compute_dtype = (
        torch.float32 if state.dtype in (torch.float16, torch.bfloat16) else state.dtype
    )
    ratio = state.new_tensor(sigma_next, dtype=compute_dtype) / state.new_tensor(
        sigma_curr, dtype=compute_dtype
    )
    out = ratio * state.to(dtype=compute_dtype) + (1.0 - ratio) * denoised.to(
        dtype=compute_dtype
    )
    out = out.to(dtype=state.dtype)
    _check_finite(out, "euler_eta0_step output")
    return out


class MiniMaxH3EulerAncestralEta0Scheduler:
    """Steps the video and audio latents together, on separate schedules."""

    def set_shift(self, flow_shift: float) -> None:
        """No-op.

        The flow shift is baked into the sigma schedule by the timestep
        preparation stage, not applied here. Kept so callers can treat this
        like the other samplers.
        """

    def step(
        self,
        *,
        visual_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        noise_pred_visual: torch.Tensor,
        noise_pred_audio: torch.Tensor,
        video_timestep: torch.Tensor,
        audio_timestep: torch.Tensor,
        video_sigma_curr: float,
        video_sigma_next: float,
        audio_sigma_curr: float,
        audio_sigma_next: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(next_visual, next_audio)``."""
        video_sigma_curr = _check_timestep_sigma_pair(
            video_timestep, video_sigma_curr, "video"
        )
        audio_sigma_curr = _check_timestep_sigma_pair(
            audio_timestep, audio_sigma_curr, "audio"
        )

        denoised_visual = minimax_h3_rf_v_to_x0(
            visual_latent, noise_pred_visual, video_timestep
        )
        denoised_audio = minimax_h3_rf_v_to_x0(
            audio_latent, noise_pred_audio, audio_timestep
        )
        return (
            minimax_h3_euler_eta0_step(
                visual_latent,
                denoised_visual,
                sigma_curr=video_sigma_curr,
                sigma_next=video_sigma_next,
            ),
            minimax_h3_euler_eta0_step(
                audio_latent,
                denoised_audio,
                sigma_curr=audio_sigma_curr,
                sigma_next=audio_sigma_next,
            ),
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
                # Where trailing alignment padding starts; lets attention drop
                # a workgroup plane it would otherwise launch for dead rows.
                "used_len": packed.get("used_len"),
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
