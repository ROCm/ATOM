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
        pixel_values = data[0].to(self._device, dtype=self._data_type)
        grid_thw = data[1].to(self._device)
        embeddings = self.visual(pixel_values, grid_thw)
        return embeddings, self.get_position_ids(grid_thw)[0]

    @torch.inference_mode()
    def batched_embedding(self, data_list, mm_types, **kwargs):
        if not all(mm_type == MMUrlType.IMAGE for mm_type in mm_types):
            return super().batched_embedding(data_list, mm_types, **kwargs)
        pixel_values = torch.cat([data[0] for data in data_list]).to(
            self._device, dtype=self._data_type
        )
        grid_thw = torch.cat([data[1] for data in data_list]).to(self._device)
        embeddings = self.visual(pixel_values, grid_thw)
        counts = (grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        return list(zip(embeddings.split(counts), self.get_position_ids(grid_thw)))


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
