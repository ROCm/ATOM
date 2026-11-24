# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Make VeOmni's Qwen-Image ``offline_training`` runnable.

A workaround for a VeOmni defect (present at 573848a), to be removed once it is
fixed upstream. Under ``offline_training`` the condition model is built with
``meta_init=True``, and Qwen-Image's ``_load_components`` returns right after
the scheduler, so ``self.vae`` stays None. ``process_condition`` runs every step
and normalises latents from ``self.vae.config``, so the run dies on its first
step with ``'NoneType' object has no attribute 'config'``. (Wan's condition
model keeps its VAE under ``meta_init`` and is unaffected.)

The three values the normalisation needs -- ``latents_mean``, ``latents_std``,
``z_dim`` -- are read from the VAE's ``config.json`` instead. No weights are
loaded and ``self.vae`` is left as it was, so nothing else changes behaviour;
with a VAE present the original method runs.
"""

import importlib
import logging
import types

import torch

logger = logging.getLogger(__name__)

_MODULE = "veomni.models.diffusers.qwen_image.qwen_image_condition.modeling_qwen_image_condition"


def install_qwen_offline_fix():
    """Patch every class in the condition module that defines ``_normalize_latents``."""
    mod = importlib.import_module(_MODULE)
    targets = [
        c
        for c in vars(mod).values()
        if isinstance(c, type)
        and c.__module__ == _MODULE
        and "_normalize_latents" in c.__dict__
    ]
    for cls in targets:
        orig = cls._normalize_latents
        if getattr(orig, "_lumen_offline_fix", False):
            continue

        def _normalize_latents(self, latents, __orig=orig):
            if self.vae is not None:
                return __orig(self, latents)
            cfg = self.__dict__.get("_lumen_vae_cfg")
            if cfg is None:
                from diffusers import AutoencoderKLQwenImage

                raw = AutoencoderKLQwenImage.load_config(
                    self.config.base_model_path, subfolder=self.config.vae_subfolder
                )
                cfg = types.SimpleNamespace(
                    latents_mean=raw["latents_mean"],
                    latents_std=raw["latents_std"],
                    z_dim=raw["z_dim"],
                )
                self.__dict__["_lumen_vae_cfg"] = cfg
            mean = torch.tensor(
                cfg.latents_mean, device=latents.device, dtype=latents.dtype
            ).view(1, cfg.z_dim, 1, 1, 1)
            std = torch.tensor(
                cfg.latents_std, device=latents.device, dtype=latents.dtype
            ).view(1, cfg.z_dim, 1, 1, 1)
            return (latents - mean) / std

        _normalize_latents._lumen_offline_fix = True
        cls._normalize_latents = _normalize_latents
    logger.info(
        "qwen_offline_fix: latent normalisation from the VAE config for %s",
        ", ".join(c.__name__ for c in targets) or "<nothing>",
    )
    return bool(targets)


__all__ = ["install_qwen_offline_fix"]
