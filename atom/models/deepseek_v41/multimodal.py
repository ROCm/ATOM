# SPDX-License-Identifier: MIT
"""Vision weight ownership and image-to-language embedding projection."""

import torch

from atom.model_ops.utils import atom_parameter

from .model import DeepseekV41ForCausalLM
from .vision import Aligner, ViT


class DeepseekV41MultimodalModel(DeepseekV41ForCausalLM):
    def __init__(self, config, *, max_length, online_quant_config=None):
        super().__init__(
            config, max_length=max_length, online_quant_config=online_quant_config
        )
        vision_config = config._multimodal_config.vision_config
        self.vision = ViT(vision_config)
        self.aligner = Aligner(vision_config, config.hidden_size)
        for name in ("image_start", "image_end", "image_newline"):
            self.register_parameter(
                name,
                atom_parameter(torch.empty(config.hidden_size, dtype=torch.bfloat16)),
            )

    def embed_input_ids(self, input_ids):
        return self.embed(input_ids)

    def get_vision_embeddings(self, pixel_values, grid_thw):
        # CPU grids drive image-local attention; no device synchronization per
        # vision layer, and no attention edges between unrelated images.
        if grid_thw.device.type != "cpu":
            raise ValueError("V4.1 image grids must remain on the CPU")
        ratio = self.aligner.downsample_ratio
        results, offset = [], 0
        for temporal, h, w in grid_thw.tolist():
            if temporal != 1:
                raise ValueError("V4.1 vision accepts still images only")
            patches = pixel_values[offset : offset + h * w]
            offset += h * w
            values = self.aligner(self.vision(patches, h, w), h, w)
            lh, lw = -(-h // ratio), -(-w // ratio)
            rows = values.view(lh, lw, -1)
            newline = self.image_newline.expand(lh, 1, -1)
            results.append(
                torch.cat(
                    (
                        self.image_start[None],
                        torch.cat((rows, newline), 1).flatten(0, 1),
                        self.image_end[None],
                    )
                )
            )
        if offset != pixel_values.shape[0]:
            raise ValueError("Image grids do not cover the supplied patches")
        return torch.cat(results)
