"""ATOM vision model exposed through RTP's multimodal worker interface."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from rtp_llm.multimodal.multimodal_mixins.base_multimodal_mixin import (
    BaseMultiModalMixin,
)
from rtp_llm.multimodal.multimodal_mixins.qwen3_5_moe.qwen3_5_moe_mixin import (
    Qwen3_5MoeImageEmbedding,
    Qwen3_5MoeVitWeight,
)
from rtp_llm.utils.base_model_datatypes import MMUrlType
from transformers import AutoProcessor

from atom.models.qwen3_5_vl import Qwen3VisionTransformer


class Qwen4ExpImageEmbedding(Qwen3_5MoeImageEmbedding):
    """Use RTP's image transport with the exact ATOM standalone vision model."""

    def __init__(self, mm_related_params):
        ckpt_path = mm_related_params.config["ckpt_path"]
        # Match standalone's AutoProcessor. Qwen3.5's override forces a slow
        # image processor and may produce different patches for the same image.
        self.mm_processor = AutoProcessor.from_pretrained(ckpt_path)
        # This checkpoint declares model_type=qwen4_exp, which is not yet
        # registered with the container's Transformers. ATOM standalone reads
        # the same vision_config fields from config.json; no HF model class is
        # needed to construct its ViT.
        with (Path(ckpt_path) / "config.json").open(encoding="utf-8") as stream:
            config = json.load(stream)
        self.visual = Qwen3VisionTransformer(
            SimpleNamespace(**config["vision_config"]),
            norm_eps=float(config["text_config"].get("rms_norm_eps", 1e-6)),
        )

    @torch.inference_mode()
    def embedding(self, data, **kwargs):
        embeddings = self._encode_images(data[0], data[1])
        return torch.cat(embeddings), torch.cat(self.get_position_ids(data[1]))

    def _encode_images(self, pixel_values, grid_thw):
        # This ViT uses unmasked attention across its entire input. Each grid
        # row must be encoded independently, including multiple images within
        # one request, or patches from different images attend to each other.
        patch_counts = grid_thw.to(device="cpu").prod(-1).tolist()
        if not patch_counts or sum(patch_counts) != pixel_values.shape[0]:
            raise ValueError("Qwen4Exp vision patches do not match image grids")
        pixels = pixel_values.to(self._device, dtype=self._data_type)
        grids = grid_thw.to(self._device)
        return [
            self.visual(patches, grid.unsqueeze(0))
            for patches, grid in zip(pixels.split(patch_counts), grids)
        ]

    @torch.inference_mode()
    def batched_embedding(self, data_list, mm_types, **kwargs):
        if not all(mm_type == MMUrlType.IMAGE for mm_type in mm_types):
            return super().batched_embedding(data_list, mm_types, **kwargs)
        outputs = []
        for data in data_list:
            pixel_values, grid_thw = data[0], data[1]
            outputs.extend(
                zip(
                    self._encode_images(pixel_values, grid_thw),
                    self.get_position_ids(grid_thw),
                )
            )
        return outputs


class Qwen4ExpVisionMixinImpl(BaseMultiModalMixin):
    def _init_multimodal(self):
        self.mm_part = Qwen4ExpImageEmbedding(self.mm_related_params)
        self.mm_related_params.vit_weights = Qwen3_5MoeVitWeight(
            {"vit": self.mm_part.visual}
        )

    def _prepare_vit_weights(self, database):
        self.mm_related_params.vit_weights.detect_ckpt_prefix(
            database.get_pretrain_tensor_names()
        )

    @classmethod
    def _get_mm_module(cls, mm_related_params, vit_config):
        return Qwen4ExpImageEmbedding(mm_related_params).visual
