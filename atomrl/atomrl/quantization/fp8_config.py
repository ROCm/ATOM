# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""FP8 rollout and training configuration derived from AtomRL quant settings."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

from atomrl.core.config import QuantizationConfig

logger = logging.getLogger(__name__)

RecipeKind = Literal["blockwise", "tensorwise"]


@dataclass(frozen=True)
class FP8Config:
    """Unified FP8 knobs for rollout, KV cache, and training paths."""

    precision: str = "bf16"
    recipe: RecipeKind = "blockwise"
    use_deep_gemm: bool = True
    num_first_layers_in_bf16: int = 0
    num_last_layers_in_bf16: int = 0
    use_weight_pow2_scale: bool = False
    use_activation_pow2_scale: bool = False

    def is_fp8_enabled(self) -> bool:
        """Return True when FP8 numerics are active for rollout or training."""
        p = self.precision.lower().strip()
        if p in {"fp8", "float8", "e4m3", "e5m2", "hybrid_fp8"}:
            return True
        return "fp8" in p

    @classmethod
    def from_config(cls, quant_config: QuantizationConfig) -> FP8Config:
        """Build an :class:`FP8Config` from a structured :class:`QuantizationConfig`."""
        if not isinstance(quant_config, QuantizationConfig):
            raise TypeError(
                f"quant_config must be QuantizationConfig, got {type(quant_config)!r}"
            )

        rollout = quant_config.rollout
        training = quant_config.training

        precision = rollout.precision
        if training.fp8:
            precision = training.fp8

        recipe: RecipeKind = "blockwise"
        if training.fp8_recipe in ("blockwise", "tensorwise"):
            recipe = training.fp8_recipe  # type: ignore[assignment]

        cfg = cls(
            precision=precision,
            recipe=recipe,
            use_deep_gemm=rollout.use_deep_gemm,
            num_first_layers_in_bf16=(
                training.num_first_layers_in_bf16
                if training.num_first_layers_in_bf16 is not None
                else rollout.num_first_layers_in_bf16
            ),
            num_last_layers_in_bf16=(
                training.num_last_layers_in_bf16
                if training.num_last_layers_in_bf16 is not None
                else rollout.num_last_layers_in_bf16
            ),
            use_weight_pow2_scale=bool(
                getattr(rollout, "use_weight_pow2_scale", False)
            ),
            # Rollout *activation* scales are quantised inside the inference engine
            # (vLLM), not here, so this stays off on the AtomRL side.
            use_activation_pow2_scale=False,
        )
        logger.debug("Built FP8Config: %s", cfg)
        return cfg


def megatron_fp8_kwargs(cfg: FP8Config, num_layers: int, pp_size: int = 1) -> dict:
    """Map :class:`FP8Config` onto ``TransformerConfig``'s FP8 fields.

    Returns ``{}`` when FP8 is off, so the caller can splat it unconditionally and
    a BF16 run is bit-for-bit the configuration it was before.

    Kept here rather than in the engine so there is one place that decides what an
    "AMD FP8 recipe" means for Megatron -- the seam B2's numerics spec will govern.

    Blockwise (TE ``Float8BlockScaling``) is the default because it is the format
    the DeepSeek-family checkpoints ship in: ``quant_method=fp8, fmt=e4m3,
    weight_block_size=[128,128]``. Training in the checkpoint's own scaling scheme
    avoids a conversion nobody has specified.

    Deliberately NOT set here:
      * amax history / compute algo -- delayed scaling only; blockwise and
        tensorwise derive scales per block or per tensor each step, and RL
        recalibrates every step anyway because the policy weights move.
      * ``fp8_param`` -- storing FP8 master weights is a separate decision from
        computing in FP8, and Megatron rejects it without ``fp8`` set.
    """
    if not cfg.is_fp8_enabled():
        return {}

    p = cfg.precision.lower().strip()
    fmt = "hybrid" if "hybrid" in p or "e5m2" in p else "e4m3"

    kwargs: dict = {"fp8": fmt, "fp8_recipe": cfg.recipe}

    # Layer keep-outs. The lm_head BF16 keep-out exists because of an AITER
    # blockscale INT32 overflow (vocab ~152k x hidden -> garbage logits -> policy
    # collapse); B4 is the fix that would retire it.
    start = int(cfg.num_first_layers_in_bf16 or 0)
    end = int(cfg.num_last_layers_in_bf16 or 0)
    if start or end:
        if cfg.recipe == "delayed":
            raise ValueError(
                "fp8_recipe='delayed' cannot keep first/last layers in BF16 "
                "(Megatron rejects the combination); use blockwise or tensorwise."
            )
        per_stage = max(1, num_layers // max(1, pp_size))
        for name, v in (
            ("num_first_layers_in_bf16", start),
            ("num_last_layers_in_bf16", end),
        ):
            if v > per_stage:
                raise ValueError(
                    f"{name}={v} exceeds layers per pipeline stage ({per_stage}); "
                    f"num_layers={num_layers} pp={pp_size}"
                )
        kwargs.update(
            first_last_layers_bf16=True,
            num_layers_at_start_in_bf16=start,
            num_layers_at_end_in_bf16=end,
        )
    return kwargs
