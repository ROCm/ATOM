# SPDX-License-Identifier: Apache-2.0
"""Diffusion model runner for Flux."""

import torch
from atom.model_loader.loader import load_model
from atom.model_ops.diffusion_sampler import FlowMatchingSampler
from atom.models.flux_vae import AutoencoderKL
from atom.models.flux_text_encoder import FluxTextEncoder


class DiffusionModelRunner:
    def __init__(self, config, device: str = "cuda"):
        self.config = config
        self.device = device
        self.dtype = torch.bfloat16

        self.model = load_model(config).to(self.device, self.dtype)
        self.sampler = FlowMatchingSampler(num_steps=50)
        self.vae = AutoencoderKL().to(self.device, self.dtype)
        self.text_encoder = FluxTextEncoder(device=device)

    @torch.no_grad()
    def generate(
        self,
        prompts: list,
        height: int = 1024,
        width: int = 1024,
        num_steps: int = 50,
        guidance_scale: float = 3.5,
    ) -> torch.Tensor:
        B = len(prompts)
        H, W = height // 8, width // 8
        latents = torch.randn(B, 16, H, W, device=self.device, dtype=self.dtype)

        text_emb = self.text_encoder.encode(prompts)
        self.sampler.set_timesteps(num_steps, self.device)

        for i, t in enumerate(self.sampler.timesteps):
            timesteps = t.expand(B)
            noise_pred = self.model(latents, timesteps, text_emb, guidance_scale)
            latents = self.sampler.step(noise_pred, timesteps, latents, i)

        return self.vae.decode(latents).clamp(-1, 1) * 0.5 + 0.5
