# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""vLLM online ``fp8_per_block`` support for AtomRL rollout (verl-free).

This is a self-contained port of the *online per-block FP8* pieces of verl's
``verl/utils/vllm/vllm_fp8_utils.py`` + ``vllm_rollout/utils.py``, with **no
dependency on ``verl``**. It only imports from ``vllm`` / ``torch``.

Strict reproduction of verl's ``rollout.quantization=fp8_per_block``:

* vLLM natively registers the ``fp8_per_block`` *online* quantization method
  (``OnlineQuantizationConfig``); the BF16 training weights are re-quantized to
  per-128-block FP8 inside vLLM on every weight load.
* On ROCm the online per-block weight post-processing needs a guard so it does
  not double-convert an already-FP8 (e4m3fnuz) tensor -> ``apply_vllm_fp8_per_block_patches``.
* Because online quant rebuilds per-layer FP8 scales, each RL weight sync must
  wrap ``model.load_weights`` with vLLM's layerwise-reload lifecycle
  (``prepare_*`` before the first bucket, ``finalize_*`` after the last).

Only what ``fp8_per_block`` needs is ported here (no static-FP8
``load_quanted_weights``, no LoRA, no QAT / NVFP4).
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from unittest.mock import patch

import torch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Module-level idempotency guard: the ROCm per-block patches are process-global
# and must only be started once per worker process.
_PER_BLOCK_PATCHES: list = []
_SOURCE_FINGERPRINTS_ATTR = "_atomrl_reload_source_fingerprints"
_DEFAULT_FINGERPRINT_SAMPLES = 64


class TensorNameClass(str, Enum):
    """How a resident vLLM tensor participates in checkpoint reloads."""

    RELOADABLE = "reloadable"
    CHECKPOINT_STATIC = "checkpoint_static"
    GENERATED = "generated"


def classify_tensor_name(
    name: str,
    *,
    streamed_scales: bool = False,
) -> TensorNameClass:
    """Classify a tensor without assuming an external-to-internal name mapping."""
    leaf = name.rsplit(".", 1)[-1]
    if leaf.endswith("weight_scale_inv"):
        return (
            TensorNameClass.RELOADABLE if streamed_scales else TensorNameClass.GENERATED
        )
    if leaf in {"tid2eid", "e_score_correction_bias"}:
        return TensorNameClass.CHECKPOINT_STATIC
    return TensorNameClass.RELOADABLE


def _reloadable_role(name: str) -> str:
    lowered = name.lower()
    if ".attn." in lowered or ".attention." in lowered or ".self_attn." in lowered:
        return "attention"
    if ".experts." in lowered or ".expert." in lowered or "shared_expert" in lowered:
        return "expert"
    if (
        lowered.endswith((".ffn.gate.weight", ".mlp.gate.weight"))
        or ".router.gate." in lowered
    ):
        return "gate"
    if "lm_head" in lowered:
        return "lm_head"
    return "fallback"


def _canonical_reload_name(name: str) -> str:
    """Canonicalize only checkpoint aliases with a known resident equivalent."""
    parts = name.split(".")
    known_leaves = {
        "weight",
        "bias",
        "weight_scale_inv",
        "tid2eid",
        "e_score_correction_bias",
    }
    if (
        len(parts) >= 5
        and parts[:2] == ["model", "layers"]
        and parts[2].isdigit()
        and parts[-1] in known_leaves
    ):
        if parts[3] == "ffn":
            parts[3] = "mlp"
        elif parts[3] == "attn":
            parts[3] = "self_attn"
    return ".".join(parts)


@dataclass(frozen=True)
class TensorFingerprint:
    """Shape, dtype, and a deterministic checksum of bounded tensor samples."""

    shape: tuple[int, ...]
    dtype: str
    checksum: str
    sample_count: int


def _sample_coordinates(
    shape: tuple[int, ...],
    sample_count: int,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    numel = 1
    for size in shape:
        numel *= size
    linear = torch.linspace(
        0,
        numel - 1,
        steps=sample_count,
        dtype=torch.int64,
        device=device,
    )
    coordinates = []
    for size in reversed(shape):
        coordinates.append(linear.remainder(size))
        linear = torch.div(linear, size, rounding_mode="floor")
    return tuple(reversed(coordinates))


@torch.no_grad()
def fingerprint_tensor(
    tensor: torch.Tensor,
    *,
    max_samples: int = _DEFAULT_FINGERPRINT_SAMPLES,
) -> TensorFingerprint:
    """Fingerprint at most ``max_samples`` elements without copying the tensor."""
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")
    value = tensor.detach()
    sample_count = min(value.numel(), max_samples)
    if sample_count == 0:
        sample = torch.empty(0, dtype=torch.uint8)
    elif value.ndim == 0:
        sample = value.unsqueeze(0)
    else:
        sample = value[
            _sample_coordinates(
                tuple(value.shape),
                sample_count,
                device=value.device,
            )
        ]
    sample_bytes = sample.contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return TensorFingerprint(
        shape=tuple(value.shape),
        dtype=str(value.dtype),
        checksum=hashlib.blake2b(sample_bytes, digest_size=16).hexdigest(),
        sample_count=sample_count,
    )


def snapshot_named_tensors(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
    *,
    max_samples: int = _DEFAULT_FINGERPRINT_SAMPLES,
) -> dict[str, TensorFingerprint]:
    """Snapshot every static tensor and one reloadable per representative role."""
    selected: dict[str, torch.Tensor] = {}
    representatives: dict[str, tuple[str, torch.Tensor]] = {}
    for name, tensor in sorted(named_tensors, key=lambda item: item[0]):
        name_class = classify_tensor_name(name)
        if name_class is TensorNameClass.CHECKPOINT_STATIC:
            selected[name] = tensor
        elif name_class is TensorNameClass.RELOADABLE:
            representatives.setdefault(_reloadable_role(name), (name, tensor))
    selected.update(representatives.values())
    return {
        name: fingerprint_tensor(
            tensor,
            max_samples=(
                max(tensor.numel(), 1)
                if classify_tensor_name(name) is TensorNameClass.CHECKPOINT_STATIC
                else max_samples
            ),
        )
        for name, tensor in selected.items()
    }


def snapshot_model_fingerprints(
    model,
    *,
    max_samples: int = _DEFAULT_FINGERPRINT_SAMPLES,
) -> dict[str, TensorFingerprint]:
    """Snapshot both parameters and buffers, preserving parameter precedence."""
    named = dict(model.named_parameters())
    named.update(
        (name, tensor) for name, tensor in model.named_buffers() if name not in named
    )
    return snapshot_named_tensors(named.items(), max_samples=max_samples)


class ReloadFingerprintTracker:
    """Verify only reload properties provable from resident and source names."""

    def __init__(
        self,
        model,
        *,
        max_samples: int = _DEFAULT_FINGERPRINT_SAMPLES,
        streamed_scales: bool = False,
    ):
        self.model = model
        self.max_samples = max_samples
        self.before = snapshot_model_fingerprints(
            model,
            max_samples=max_samples,
        )
        self.source: dict[str, TensorFingerprint] = {}
        self._source_slots: dict[str, str] = {}
        self._exact_source: dict[str, TensorFingerprint] = {}
        self._canonical_source: dict[str, tuple[str, TensorFingerprint]] = {}
        self.streamed_scales = bool(streamed_scales)
        self._exact_target_names = {
            name
            for name in self.before
            if classify_tensor_name(name) is TensorNameClass.RELOADABLE
        }
        resident = dict(model.named_parameters())
        resident.update(
            (name, tensor)
            for name, tensor in model.named_buffers()
            if name not in resident
        )
        self._checkpoint_static_values = {
            name: (
                tensor.detach().clone(),
                name in dict(model.named_parameters()),
            )
            for name, tensor in resident.items()
            if name.rsplit(".", 1)[-1] == "tid2eid"
        }
        canonical_candidates: dict[
            tuple[str, tuple[int, ...]],
            list[str],
        ] = {}
        for name, tensor in resident.items():
            if classify_tensor_name(name) is not TensorNameClass.RELOADABLE:
                continue
            key = (_canonical_reload_name(name), tuple(tensor.shape))
            canonical_candidates.setdefault(key, []).append(name)
        self._canonical_targets = {
            key: names[0]
            for key, names in canonical_candidates.items()
            if len(names) == 1 and names[0] in self.before
        }

    def classify_name(self, name: str) -> TensorNameClass:
        return classify_tensor_name(
            name,
            streamed_scales=self.streamed_scales,
        )

    def checkpoint_static_changed_names(self) -> list[str]:
        """Return protected resident tensors changed since tracker creation."""
        resident = dict(self.model.named_parameters())
        resident.update(
            (name, tensor)
            for name, tensor in self.model.named_buffers()
            if name not in resident
        )
        changed = []
        for name, before in self.before.items():
            if classify_tensor_name(name) is not TensorNameClass.CHECKPOINT_STATIC:
                continue
            tensor = resident.get(name)
            if tensor is None or tensor.is_meta:
                changed.append(name)
                continue
            current = fingerprint_tensor(
                tensor,
                max_samples=max(tensor.numel(), 1),
            )
            if current != before:
                changed.append(name)
        return sorted(changed)

    @torch.no_grad()
    def restore_checkpoint_static_tensors(self) -> None:
        """Restore protected tensors replaced by the reload lifecycle."""
        parameters = dict(self.model.named_parameters())
        buffers = dict(self.model.named_buffers())
        for name, (saved, was_parameter) in self._checkpoint_static_values.items():
            expected = parameters if was_parameter else buffers
            unexpected = buffers if was_parameter else parameters
            target = expected.get(name)
            if target is None:
                if name in unexpected:
                    raise RuntimeError(
                        "checkpoint-static tensor registration changed during "
                        f"reload: {name}"
                    )
                raise RuntimeError(
                    f"checkpoint-static tensor disappeared during reload: {name}"
                )
            if target.is_meta:
                raise RuntimeError(
                    f"checkpoint-static tensor remains meta after finalize: {name}"
                )
            if target.shape != saved.shape:
                raise RuntimeError(
                    "checkpoint-static tensor shape changed during reload: "
                    f"{name} {tuple(saved.shape)} -> {tuple(target.shape)}"
                )
            if target.dtype != saved.dtype:
                raise RuntimeError(
                    "checkpoint-static tensor dtype changed during reload: "
                    f"{name} {saved.dtype} -> {target.dtype}"
                )
            if target.device != saved.device:
                raise RuntimeError(
                    "checkpoint-static tensor device changed during reload: "
                    f"{name} {saved.device} -> {target.device}"
                )
            target.copy_(saved)

    def observe_source(
        self,
        weights: Iterable[tuple[str, torch.Tensor]],
    ) -> None:
        for name, tensor in weights:
            if self.classify_name(name) is not TensorNameClass.RELOADABLE:
                continue
            role = _reloadable_role(name)
            current_name = self._source_slots.get(role)
            source_shape = tuple(tensor.shape)
            is_exact_target = (
                name in self._exact_target_names
                and self.before[name].shape == source_shape
            )
            canonical_target = self._canonical_targets.get(
                (_canonical_reload_name(name), source_shape)
            )
            is_canonical_alias = (
                canonical_target is not None and canonical_target != name
            )
            is_role_representative = current_name is None or name < current_name
            if (
                not is_exact_target
                and not is_canonical_alias
                and not is_role_representative
            ):
                continue
            fingerprint = fingerprint_tensor(
                tensor,
                max_samples=self.max_samples,
            )
            if is_exact_target:
                self._exact_source[name] = fingerprint
            elif is_canonical_alias:
                self._canonical_source[canonical_target] = (name, fingerprint)
            if not is_role_representative:
                continue
            if current_name is not None:
                self.source.pop(current_name, None)
            self._source_slots[role] = name
            self.source[name] = fingerprint

    def finalize(self) -> dict[str, object]:
        after = snapshot_model_fingerprints(
            self.model,
            max_samples=self.max_samples,
        )
        previous_source = getattr(self.model, _SOURCE_FINGERPRINTS_ATTR, {})
        static_changed = sorted(
            name
            for name, before in self.before.items()
            if classify_tensor_name(name) is TensorNameClass.CHECKPOINT_STATIC
            and after.get(name) != before
        )
        model_names = {name for name, _ in self.model.named_parameters()} | {
            name for name, _ in self.model.named_buffers()
        }
        generated_excluded = sum(
            self.classify_name(name) is TensorNameClass.GENERATED
            for name in model_names
        )
        classified = {
            name_class.value: sum(
                self.classify_name(name) is name_class for name in model_names
            )
            for name_class in TensorNameClass
        }

        source_changed = 0
        source_unchanged = 0
        source_history_baselines = 0
        current_history = {
            f"role:{role}": (name, self.source[name])
            for role, name in self._source_slots.items()
        }
        current_history.update(
            {
                f"exact:{name}": (name, fingerprint)
                for name, fingerprint in self._exact_source.items()
            }
        )
        current_history.update(
            {
                f"canonical:{target_name}": source
                for target_name, source in self._canonical_source.items()
            }
        )
        for role in self._source_slots:
            key = f"role:{role}"
            old_source = previous_source.get(key)
            if old_source is None:
                source_history_baselines += 1
            elif current_history[key] == old_source:
                source_unchanged += 1
            else:
                source_changed += 1

        exact_name_mappings = len(self._exact_source)
        exact_name_change_checks = sum(
            f"exact:{name}" in previous_source for name in self._exact_source
        )
        reload_target_unchanged = []
        for name, source_fingerprint in self._exact_source.items():
            old_source = previous_source.get(f"exact:{name}")
            if old_source is None:
                continue
            _, old_fingerprint = old_source
            if (
                source_fingerprint != old_fingerprint
                and name in self.before
                and name in after
                and self.before[name] == after[name]
            ):
                reload_target_unchanged.append(name)
        for target_name, (_, source_fingerprint) in self._canonical_source.items():
            old_source = previous_source.get(f"canonical:{target_name}")
            if old_source is None:
                continue
            _, old_fingerprint = old_source
            if (
                source_fingerprint != old_fingerprint
                and target_name in self.before
                and target_name in after
                and self.before[target_name] == after[target_name]
            ):
                reload_target_unchanged.append(target_name)

        summary: dict[str, object] = {
            "snapshotted": len(after),
            "source_snapshotted": len(self.source),
            "source_changed": source_changed,
            "source_unchanged": source_unchanged,
            "source_history_baselines": source_history_baselines,
            "first_source_snapshot": not bool(previous_source),
            "exact_name_mappings": exact_name_mappings,
            "exact_name_change_checks": exact_name_change_checks,
            "canonical_shape_mappings": len(self._canonical_source),
            "mapping_verification_scope": (
                "canonical-name+exact-shape change correspondence; "
                "no source/target value-tolerance parity"
            ),
            "classified": classified,
            "checkpoint_static_changed": static_changed,
            "generated_excluded": generated_excluded,
            "reload_target_unchanged_failures": len(reload_target_unchanged),
        }
        failures = []
        if static_changed:
            failures.append(f"checkpoint_static_changed={static_changed}")
        if reload_target_unchanged:
            failures.append(
                f"reload_target_unchanged={sorted(reload_target_unchanged)}"
            )
        if failures:
            raise RuntimeError(
                "Reload fingerprint verification failed: "
                + "; ".join(failures)
                + f"; summary={summary}"
            )

        setattr(self.model, _SOURCE_FINGERPRINTS_ATTR, current_history)
        return summary


def is_online_quant_model(vllm_config) -> bool:
    """Return True if vLLM is using an online quantization (e.g. ``fp8_per_block``)."""
    try:
        from vllm.model_executor.layers.quantization.online.base import (
            OnlineQuantizationConfig,
        )
    except ImportError:
        return False

    quant_config = getattr(vllm_config, "quant_config", None)
    return isinstance(quant_config, OnlineQuantizationConfig)


def _restore_prequantized_fp8_parameter_metadata(
    model,
    *,
    get_layerwise_info,
    block_scale_parameter_cls,
    restore_layer_refs=lambda parameter, layer: parameter,
) -> dict[str, int]:
    """Restore vLLM sharding metadata without replacing resident storage."""
    restored_weights = 0
    restored_scales = 0
    restored_moe_scales = 0
    for layer in model.modules():
        parameters = layer._parameters
        for scale_name in (
            "w13_weight_scale_inv",
            "w2_weight_scale_inv",
        ):
            scale = parameters.get(scale_name)
            if scale is None:
                continue
            changed = False
            weight_name = scale_name.removesuffix("_scale_inv")
            weight = parameters.get(weight_name)
            if not hasattr(scale, "weight_loader"):
                loader = getattr(weight, "weight_loader", None)
                if loader is None:
                    raise RuntimeError(
                        "Cannot restore prequantized FP8 MoE scale loader: "
                        f"{layer.__class__.__name__}.{scale_name} has no "
                        f"{weight_name} loader"
                    )
                scale.weight_loader = loader
                changed = True
            quant_method = getattr(scale, "quant_method", None)
            if quant_method is None:
                scale.quant_method = "block"
                changed = True
            elif quant_method != "block":
                raise RuntimeError(
                    "Prequantized FP8 MoE scale has incompatible quant_method: "
                    f"{layer.__class__.__name__}.{scale_name}={quant_method!r}"
                )
            restored_moe_scales += int(changed)

        weight = parameters.get("weight")
        scale = parameters.get("weight_scale_inv")
        if weight is None:
            continue

        restore_params, _ = get_layerwise_info(layer).restore_metadata
        template = restore_params.get("weight")
        if template is not None and weight.__class__ is not template.__class__:
            resident_loader = getattr(weight, "weight_loader", None)
            template = restore_layer_refs(template, layer)
            weight.__class__ = template.__class__
            weight.__dict__ = template.__dict__.copy()
            if resident_loader is not None:
                weight.weight_loader = resident_loader
            restored_weights += 1

        if scale is None or isinstance(scale, block_scale_parameter_cls):
            continue
        if template is None:
            raise RuntimeError(
                "Cannot restore prequantized FP8 scale sharding metadata: "
                f"{layer.__class__.__name__}.weight has no recorded reload template"
            )
        if weight.ndim != 2 or scale.ndim != 2:
            raise RuntimeError(
                "Unsupported prequantized FP8 linear tensor rank: "
                f"{layer.__class__.__name__}.weight={tuple(weight.shape)} "
                f"weight_scale_inv={tuple(scale.shape)}"
            )
        if getattr(layer, "weight_block_size", None) != [128, 128]:
            raise RuntimeError(
                "Unsupported prequantized FP8 block size for "
                f"{layer.__class__.__name__}: "
                f"{getattr(layer, 'weight_block_size', None)}"
            )

        # Online FP8 creates the scale after the checkpoint parameter metadata
        # is recorded. Give it the weight's exact TP loader metadata and the
        # block-scale type expected by vLLM's linear sharding loaders.
        scale.__class__ = block_scale_parameter_cls
        scale.__dict__ = weight.__dict__.copy()
        restored_scales += 1

    return {
        "weights": restored_weights,
        "block_scales": restored_scales,
        "moe_scales": restored_moe_scales,
    }


def prepare_prequantized_fp8_weights_for_loading(model) -> dict[str, int]:
    """Restore online-FP8 loader metadata for a direct FP8+scale reload."""
    from vllm.model_executor.model_loader.reload.layerwise import (
        get_layerwise_info,
    )
    from vllm.model_executor.model_loader.reload.sanitize import (
        restore_layer_refs,
    )
    from vllm.model_executor.parameter import BlockQuantScaleParameter

    return _restore_prequantized_fp8_parameter_metadata(
        model,
        get_layerwise_info=get_layerwise_info,
        block_scale_parameter_cls=BlockQuantScaleParameter,
        restore_layer_refs=restore_layer_refs,
    )


def rollout_pow2_scales_enabled() -> bool:
    """Whether to round rollout FP8 block scales to powers of two.

    Matches TransformerEngine's force_pow_2_scales (TE's default), so the rollout and
    the trainer quantise the same weights identically instead of only to ~1e-7. TE
    rounds the dequant multiplier UP (it truncates the mantissa of FP8_MAX/amax), so
    both sites here use exp2(ceil(log2(s))) -- the same convention vLLM use.

    Off by default; set ATOMRL_ROLLOUT_POW2_SCALES=1 to enable.
    """
    return os.environ.get("ATOMRL_ROLLOUT_POW2_SCALES", "0").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def _pow2_activation_quant_wrapper(orig):
    """Force use_ue8m0=True on vLLM's per-token-group activation quantiser.

    vLLM gates UE8M0 rounding on is_deep_gemm_e8m0_used(), which is False on ROCm
    (DeepGEMM is CUDA-only), and _triton_per_token_group_quant_fp8_impl hardcodes
    use_ue8m0=False. The rounding itself is platform-independent: the power-of-two
    value goes into the ordinary fp32 scale tensor.
    """

    def wrapper(*args, **kwargs):
        kwargs["use_ue8m0"] = True
        return orig(*args, **kwargs)

    return wrapper


def process_fp8_weight_block_strategy_rocm_safe(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ROCm guard for online ``fp8_per_block`` weight post-processing.

    On AMD, online ``fp8_per_block`` may already emit the platform FP8 dtype
    (e4m3fnuz). Only normalize ``e4m3fn -> e4m3fnuz`` when the tensor is still
    ``e4m3fn`` to avoid a second (corrupting) conversion.
    """
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        _maybe_pad_fp8_weight,
    )
    from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
        normalize_e4m3fn_to_e4m3fnuz,
    )
    from vllm.platforms import current_platform

    if current_platform.is_fp8_fnuz() and weight.dtype == torch.float8_e4m3fn:
        weight, weight_scale, _ = normalize_e4m3fn_to_e4m3fnuz(
            weight=weight, weight_scale=weight_scale
        )

    if rollout_pow2_scales_enabled() and weight_scale.dtype == torch.float32:
        from vllm.model_executor.layers.quantization.utils.fp8_utils import (
            requant_weight_ue8m0_inplace,
        )

        requant_weight_ue8m0_inplace(weight, weight_scale, block_size=(128, 128))

    weight = _maybe_pad_fp8_weight(weight)
    return weight, weight_scale


def apply_vllm_fp8_per_block_patches() -> None:
    """Patch vLLM online ``fp8_per_block`` for ROCm FP8 weight processing.

    Idempotent + resilient: each target is patched independently so a vLLM
    version that lacks one of the symbols does not abort the others.
    """
    if _PER_BLOCK_PATCHES:
        logger.debug("vLLM fp8_per_block patches already applied")
        return

    targets = [
        "vllm.model_executor.layers.quantization.utils.fp8_utils.process_fp8_weight_block_strategy",
        "vllm.model_executor.kernels.linear.scaled_mm.BlockScaledMMLinearKernel.process_fp8_weight_block_strategy",
    ]
    if rollout_pow2_scales_enabled():
        # Activation scales too, or the two GEMM operands disagree.
        try:
            import vllm.model_executor.layers.quantization.utils.fp8_utils as _vf

            patcher = patch.object(
                _vf,
                "per_token_group_quant_fp8",
                _pow2_activation_quant_wrapper(_vf.per_token_group_quant_fp8),
            )
            patcher.start()
            _PER_BLOCK_PATCHES.append(patcher)
            print(
                "[atomrl] rollout pow-2 block scales: ENABLED (weights + activations)",
                flush=True,
            )
        except (AttributeError, ModuleNotFoundError) as exc:
            logger.warning("skip pow-2 activation patch: %s", exc)

    applied = 0
    for target in targets:
        try:
            patcher = patch(target, process_fp8_weight_block_strategy_rocm_safe)
            patcher.start()
            _PER_BLOCK_PATCHES.append(patcher)
            applied += 1
        except (AttributeError, ModuleNotFoundError) as exc:
            logger.warning("skip fp8_per_block patch %s: %s", target, exc)
    print(
        f"[atomrl] applied vLLM fp8_per_block ROCm patches ({applied}/{len(targets)}), "
        f"pow2_scales={rollout_pow2_scales_enabled()}",
        flush=True,
    )


def prepare_online_quantized_weights_for_loading(model) -> None:
    """Set up vLLM per-layer reload state BEFORE the first weight bucket."""
    from vllm.model_executor.model_loader.reload import initialize_layerwise_reload

    initialize_layerwise_reload(model)


def finalize_online_quantized_weights_loading(model, model_config) -> None:
    """Finalize vLLM per-layer reload AFTER the last weight bucket.

    Rebuilds the per-block FP8 weight scales for the freshly loaded BF16 weights.
    """
    from vllm.model_executor.model_loader.reload import finalize_layerwise_reload

    finalize_layerwise_reload(model, model_config)
