"""Lightweight RTP registration for Qwen4Exp's vision worker.

Frontend processes import external model packages during startup. Importing
RTP's multimodal implementation here loads heavy optional libraries in every
frontend; defer that work until the vision worker is actually constructed.
"""

from __future__ import annotations

import json
from pathlib import Path

from rtp_llm.multimodal.multimodal_mixin_register import register_multimodal_mixin


class Qwen4ExpVisionMixin:
    def __new__(cls, *args, **kwargs):
        from atom.plugin.rtpllm.models.qwen4_exp_vision_impl import (
            Qwen4ExpVisionMixinImpl,
        )

        return Qwen4ExpVisionMixinImpl(*args, **kwargs)

    @staticmethod
    def _vision_parameter_count(mm_related_params) -> int:
        from safetensors import safe_open

        checkpoint_path = Path(mm_related_params.config["ckpt_path"])
        with (checkpoint_path / "model.safetensors.index.json").open(
            encoding="utf-8"
        ) as stream:
            weight_map = json.load(stream)["weight_map"]
        vision_weights = {
            name: shard
            for name, shard in weight_map.items()
            if name.startswith("model.visual.")
        }
        if not vision_weights:
            raise ValueError("Qwen4Exp checkpoint has no model.visual weights")
        count = 0
        for shard_name in set(vision_weights.values()):
            with safe_open(
                checkpoint_path / shard_name, framework="pt", device="cpu"
            ) as shard:
                for name, filename in vision_weights.items():
                    if filename != shard_name:
                        continue
                    elements = 1
                    for dim in shard.get_slice(name).get_shape():
                        elements *= dim
                    count += elements
        return count

    @classmethod
    def eval_mm_model_param_count(
        cls, mm_related_params, extra_data_path, local_extra_data_path
    ) -> int:
        return cls._vision_parameter_count(mm_related_params)

    @classmethod
    def eval_mm_model_size(
        cls, mm_related_params, extra_data_path, local_extra_data_path
    ) -> int:
        # RTP's model-size estimate assumes bf16 model weights.
        return cls._vision_parameter_count(mm_related_params) * 2


register_multimodal_mixin(["atom_qwen4_exp", "atom_qwen4_exp_mtp"], Qwen4ExpVisionMixin)
