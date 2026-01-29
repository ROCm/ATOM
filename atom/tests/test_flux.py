# SPDX-License-Identifier: Apache-2.0
"""Tests for Flux model components."""

import unittest
import torch


class MockConfig:
    hidden_size = 768
    num_attention_heads = 12
    patch_size = 2
    in_channels = 4
    out_channels = 4
    num_double_layers = 2
    num_single_layers = 2
    text_hidden_size = 768


class MockAtomConfig:
    def __init__(self):
        self.hf_config = MockConfig()
        self.quant_config = None


class TestFluxComponents(unittest.TestCase):
    def test_patch_embed(self):
        from atom.models.flux import FluxPatchEmbed

        embed = FluxPatchEmbed(in_ch=4, hidden_size=768, patch_size=2)
        x = torch.randn(2, 4, 32, 32)
        out = embed(x)
        self.assertEqual(out.shape, (2, 256, 768))

    def test_timestep_embedding(self):
        from atom.models.flux import timestep_embedding

        t = torch.tensor([0.0, 0.5, 1.0])
        emb = timestep_embedding(t, 256)
        self.assertEqual(emb.shape, (3, 256))

    def test_sampler(self):
        from atom.model_ops.diffusion_sampler import FlowMatchingSampler

        sampler = FlowMatchingSampler(num_steps=10)
        sampler.set_timesteps(10, "cpu")
        self.assertEqual(len(sampler.timesteps), 10)

    def test_vae(self):
        from atom.models.flux_vae import AutoencoderKL

        vae = AutoencoderKL(in_ch=3, out_ch=3, latent_ch=4, ch=32, ch_mult=(1, 2))
        x = torch.randn(1, 3, 64, 64)
        z = vae.encode(x)
        out = vae.decode(z)
        self.assertEqual(out.shape[0], 1)
        self.assertEqual(out.shape[1], 3)


if __name__ == "__main__":
    unittest.main()
