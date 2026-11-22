# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The encoder-only server (``--encoder-only``): no engine core.

The server builds a model's vision modules alone on one GPU and loads their
weights, skipping the language model's. Each image request is encoded on one
thread, in arrival order, into an ``EcPool`` that a prefill server reads; the
chat handler answers with the pool's ``ec_transfer_params``.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, ClassVar

import torch
from torch import nn

from atom.config import Config
from atom.model_engine.multimodal_runtime import multimodal_cache_seed
from atom.model_loader.loading_core import load_weights_into_model
from atom.model_loader.weight_iterator import safetensors_weights_iterator

logger = logging.getLogger("atom")


def _minimax_m3(config: Config):
    from atom.models.minimax_m3.vl import (
        MiniMaxM3SparseForConditionalGenerationVL,
        build_vision_modules,
    )

    names = ("vision_tower", "multi_modal_projector", "patch_merge_mlp")
    modules = build_vision_modules(config.hf_config)
    return MiniMaxM3SparseForConditionalGenerationVL, dict(zip(names, modules))


def _kimi_k3(config: Config):
    from atom.models.kimi_k3 import KimiK3ForConditionalGeneration
    from atom.models.kimi_k3_vl import build_vision_modules

    names = ("vision_tower", "mm_projector")
    modules = build_vision_modules(config.multimodal_config.vision_config)
    return KimiK3ForConditionalGeneration, dict(zip(names, modules))


# Checkpoint architecture -> (model class, its vision modules by attribute name).
VISION_BUILDERS = {
    "MiniMaxM3SparseForConditionalGeneration": _minimax_m3,
    "KimiK3ForConditionalGeneration": _kimi_k3,
}


class VisionEncoder(nn.Module):
    """A model's vision modules under the model's attribute names, which are
    the checkpoint's, encoding as the model's ``get_vision_embeddings`` does."""

    # Every checkpoint tensor outside the vision modules.
    skip_weight_prefixes: ClassVar[list[str]] = ["language_model."]

    def __init__(self, model_cls: type, modules: dict[str, nn.Module]) -> None:
        super().__init__()
        for name, module in modules.items():
            self.add_module(name, module)
        self.has_vision_tower = True
        self.packed_modules_mapping = model_cls.packed_modules_mapping
        self.weights_mapping = model_cls.weights_mapping
        self._get_vision_embeddings = model_cls.get_vision_embeddings

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor):
        return self._get_vision_embeddings(self, pixel_values, grid_thw)


def _copy_weight(param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
    # One GPU: every tensor is whole.
    param.data.copy_(loaded_weight)


def load_vision_weights(encoder: nn.Module, config: Config) -> None:
    load_weights_into_model(
        encoder,
        config.model,
        config.hf_config,
        load_dummy=config.load_dummy,
        default_weight_loader=_copy_weight,
        fuse_shared_expert=lambda *_: False,
        is_rank0=lambda: True,
        weights_iterator=safetensors_weights_iterator,
    )


class EncoderOnly:
    """Encodes one request's images at a time into the pool."""

    def __init__(
        self,
        config: Config,
        encoder: nn.Module,
        pool: Any,
        device: torch.device | str,
    ) -> None:
        self.config = config
        self._encoder = encoder
        self._pool = pool
        self._device = device
        # One thread, fed by the executor's queue.
        self._thread = ThreadPoolExecutor(1, thread_name_prefix="atom-encoder")

    async def encode(self, data: dict[str, Any]) -> dict[str, Any]:
        """``ec_transfer_params`` for ``data`` as ``_prepare_multimodal_inputs``
        built it. Raises ``EcPoolFull`` when the pool has no room past its
        leases."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._thread, self._encode, data)

    @torch.inference_mode()
    def _encode(self, data: dict[str, Any]) -> dict[str, Any]:
        pixel_values = data["pixel_values"].to(
            device=self._device, dtype=self.config.torch_dtype
        )
        rows = self._encoder(pixel_values, data["image_grid_thw"])
        expected = sum(count for _, count in data["embedding_spans"])
        if rows.shape[0] != expected:
            raise RuntimeError(
                f"Vision output has {rows.shape[0]} rows; the image spans "
                f"hold {expected}"
            )
        return self._pool.put(multimodal_cache_seed(data), rows)

    def close(self) -> None:
        self._thread.shutdown(wait=False, cancel_futures=True)


def create_encoder_only(config: Config) -> EncoderOnly:
    """Build and load the vision modules on GPU 0, then give the pool what
    ``gpu_memory_utilization`` of the GPU leaves."""
    from atom.kv_transfer.disaggregation.mooncake.ec_transport import EcPool

    arch = config.hf_config.architectures[0]
    build = VISION_BUILDERS.get(arch)
    if build is None:
        raise ValueError(
            f"--encoder-only serves {', '.join(VISION_BUILDERS)}, not {arch}"
        )
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(config.torch_dtype)
    try:
        with device:
            encoder = VisionEncoder(*build(config))
    finally:
        torch.set_default_dtype(default_dtype)
    load_vision_weights(encoder, config)
    encoder.eval()
    free, total = torch.cuda.mem_get_info(device)
    pool_bytes = int(total * config.gpu_memory_utilization) - (total - free)
    # Mooncake registers a region this large in slices, and registration fails
    # when a slice starts inside a 2 MiB GPU page: whole GiB keep them aligned.
    pool_bytes = pool_bytes >> 30 << 30
    if pool_bytes <= 0:
        raise ValueError(
            f"--gpu-memory-utilization {config.gpu_memory_utilization} leaves no "
            f"memory for the embedding pool: {(total - free) >> 20} MiB of "
            f"{total >> 20} MiB are in use after the vision modules"
        )
    logger.info("Encoder-only: %d MiB embedding pool", pool_bytes >> 20)
    return EncoderOnly(config, encoder, EcPool(pool_bytes, device), device)
