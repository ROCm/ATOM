# SPDX-License-Identifier: MIT

from types import SimpleNamespace

import torch

from atom.models.glm5_next_vl import (
    Glm5NextVisionTransformer,
    PackedGateUp,
    VisionAttention,
)


def _tiny_config():
    return SimpleNamespace(
        hidden_size=32,
        num_heads=4,
        patch_size=2,
        temporal_patch_size=2,
        spatial_merge_size=2,
        in_channels=3,
        depth=2,
        intermediate_size=64,
        out_hidden_size=48,
        projection_intermediate_size=80,
        rms_norm_eps=1e-5,
        swiglu_limit=10.0,
    )


def test_vision_encoder_output_matches_merged_grid_size():
    model = Glm5NextVisionTransformer(_tiny_config())
    pixels = torch.randn(16, 3 * 2 * 2 * 2)

    output = model(pixels, torch.tensor([[2, 4, 2]]))

    assert output.shape == (4, 48)


def test_attention_does_not_cross_frame_boundaries():
    torch.manual_seed(1)
    attention = VisionAttention(dim=32, heads=4, eps=1e-5)
    hidden = torch.randn(8, 32)
    cos = torch.ones(8, 8)
    sin = torch.zeros(8, 8)

    segmented = attention(hidden, cos, sin, [4, 4])
    separate = torch.cat(
        (
            attention(hidden[:4], cos[:4], sin[:4], [4]),
            attention(hidden[4:], cos[4:], sin[4:], [4]),
        )
    )

    torch.testing.assert_close(segmented, separate)


def test_packed_gate_up_loader_places_checkpoint_shards():
    layer = PackedGateUp(4, 3, bias=True)
    gate = torch.full((3, 4), 1.0)
    up = torch.full((3, 4), 2.0)

    layer.weight.weight_loader(layer.weight, gate, 0)
    layer.weight.weight_loader(layer.weight, up, 1)

    torch.testing.assert_close(layer.weight[:3], gate)
    torch.testing.assert_close(layer.weight[3:], up)
