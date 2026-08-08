# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
#
# Sampler semantics follow the reference in sgl-project/sglang
# (runtime/models/schedulers/scheduling_minimax_h3_euler_ancestral.py,
# Apache-2.0).

"""MiniMax-H3 rectified-flow Euler-ancestral sampler at eta = 0.

Two identities define the whole thing:

    sigma(t) = 1 - t                      (the model is trained on this)
    x0       = xt + (1 - t) * v           (velocity -> clean sample)
    x_next   = r * xt + (1 - r) * x0,     r = sigma_next / sigma_curr

At eta = 0 the ancestral noise term vanishes, so the step is a plain
interpolation between the current state and the model's clean estimate.

Video and audio are stepped with independent sigma schedules -- H3 uses a
different flow shift per modality -- which is why every entry point takes
per-modality sigmas rather than one shared pair.
"""

import math

import torch


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
