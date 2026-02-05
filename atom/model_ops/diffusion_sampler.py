# SPDX-License-Identifier: Apache-2.0
"""Flow matching sampler for diffusion models."""

import torch


class FlowMatchingSampler:
    def __init__(self, num_steps: int = 50, shift: float = 1.0):
        self.num_steps = num_steps
        self.shift = shift
        self.timesteps = None

    def set_timesteps(self, num_steps: int = None, device: str = "cuda"):
        n = num_steps or self.num_steps
        sigmas = torch.linspace(1, 0, n + 1, device=device)
        sigmas = self.shift * sigmas / (1 + (self.shift - 1) * sigmas)
        self.timesteps = sigmas[:-1]
        self.sigmas = sigmas

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        step_idx: int,
    ) -> torch.Tensor:
        dt = self.sigmas[step_idx + 1] - self.sigmas[step_idx]
        return sample + model_output * dt
