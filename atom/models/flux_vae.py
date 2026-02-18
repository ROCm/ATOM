# SPDX-License-Identifier: Apache-2.0
"""VAE for Flux latent encoding/decoding."""

import torch
from torch import nn


class AutoencoderKL(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, latent_ch=16, ch=128, ch_mult=(1, 2, 4, 4)):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, 1, 1),
            *[
                nn.Sequential(
                    nn.Conv2d(
                        ch * ch_mult[max(0, i - 1)], ch * m, 3, 2 if i > 0 else 1, 1
                    ),
                    nn.SiLU(),
                )
                for i, m in enumerate(ch_mult)
            ],
            nn.Conv2d(ch * ch_mult[-1], latent_ch * 2, 3, 1, 1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_ch, ch * ch_mult[-1], 3, 1, 1),
            *[
                nn.Sequential(
                    nn.Upsample(scale_factor=2) if i > 0 else nn.Identity(),
                    nn.Conv2d(
                        ch * ch_mult[len(ch_mult) - i - 1],
                        ch * ch_mult[max(0, len(ch_mult) - i - 2)],
                        3,
                        1,
                        1,
                    ),
                    nn.SiLU(),
                )
                for i in range(len(ch_mult))
            ],
            nn.Conv2d(ch, out_ch, 3, 1, 1),
        )
        self.scale = 0.13025

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return h[:, : h.shape[1] // 2] * self.scale

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z / self.scale)
