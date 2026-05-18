# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import json
import logging
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from torch import nn

logger = logging.getLogger("atom")

_LORA_TENSOR_RE = re.compile(
    r"^(?P<module>.+)\.lora_(?P<side>[AB])(?:\.[^.]+)?\.weight$"
)
_ROUTED_EXPERT_RE = re.compile(
    r"^(?P<prefix>.+\.mlp)\.experts\.(?P<expert_id>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)$"
)


@dataclass(frozen=True)
class LoRAModuleSpec:
    name: str
    path: str


@dataclass(frozen=True)
class LoRATensorPair:
    module_name: str
    lora_a: torch.Tensor
    lora_b: torch.Tensor
    scaling: float


def parse_lora_module_entry(entry: str) -> LoRAModuleSpec:
    """Parse a vLLM-style ``name=path`` LoRA module entry."""
    if "=" in entry:
        name, path = entry.split("=", 1)
        name = name.strip()
        path = path.strip()
    else:
        path = entry.strip()
        name = Path(path).name
    if not name:
        raise ValueError(f"LoRA module entry has an empty name: {entry!r}")
    if not path:
        raise ValueError(f"LoRA module entry has an empty path: {entry!r}")
    return LoRAModuleSpec(name=name, path=path)


def _strip_peft_prefix(module_name: str) -> str:
    while module_name.startswith("base_model.model."):
        module_name = module_name[len("base_model.model.") :]
    module_name = module_name.replace(".mlp.shared_expert.", ".mlp.shared_experts.")
    return module_name


def parse_lora_tensor_name(name: str) -> tuple[str, str] | None:
    match = _LORA_TENSOR_RE.match(name)
    if match is None:
        return None
    return _strip_peft_prefix(match.group("module")), match.group("side")


def _find_adapter_weights(adapter_path: str) -> str:
    weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            "Static LoRA loading currently expects "
            f"{weights_path}; adapter_model.bin is not supported."
        )
    return weights_path


def _load_adapter_config(adapter_path: str) -> dict[str, Any]:
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"LoRA adapter config not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def _pattern_value(
    pattern: dict[str, Any] | None,
    module_name: str,
    default: Any,
) -> Any:
    if not pattern:
        return default
    candidates = [module_name]
    parts = module_name.split(".")
    candidates.extend(".".join(parts[i:]) for i in range(1, len(parts)))
    for candidate in candidates:
        if candidate in pattern:
            return pattern[candidate]
    return default


def _lora_scaling(
    adapter_config: dict[str, Any],
    module_name: str,
    rank: int,
) -> float:
    alpha_value = _pattern_value(adapter_config.get("alpha_pattern"), module_name, None)
    if alpha_value is None:
        alpha_value = adapter_config.get("lora_alpha", rank)
    alpha = float(alpha_value)
    if adapter_config.get("use_rslora", False):
        return alpha / math.sqrt(rank)
    return alpha / rank


def load_lora_tensors(adapter_path: str) -> list[LoRATensorPair]:
    adapter_config = _load_adapter_config(adapter_path)
    weights_path = _find_adapter_weights(adapter_path)

    tensors: dict[str, dict[str, torch.Tensor]] = {}
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for tensor_name in f.keys():
            parsed = parse_lora_tensor_name(tensor_name)
            if parsed is None:
                continue
            module_name, side = parsed
            tensors.setdefault(module_name, {})[side] = f.get_tensor(tensor_name)

    pairs: list[LoRATensorPair] = []
    incomplete = []
    for module_name, sides in sorted(tensors.items()):
        if "A" not in sides or "B" not in sides:
            incomplete.append(module_name)
            continue
        lora_a = sides["A"]
        lora_b = sides["B"]
        if lora_a.dim() != 2 or lora_b.dim() != 2:
            raise ValueError(
                f"LoRA tensors for {module_name} must be 2D, got "
                f"A={tuple(lora_a.shape)} B={tuple(lora_b.shape)}"
            )
        rank = int(lora_a.shape[0])
        if lora_b.shape[1] != rank:
            raise ValueError(
                f"LoRA rank mismatch for {module_name}: "
                f"A={tuple(lora_a.shape)} B={tuple(lora_b.shape)}"
            )
        expected_rank = int(
            _pattern_value(
                adapter_config.get("rank_pattern"),
                module_name,
                adapter_config.get("r", rank),
            )
        )
        if expected_rank != rank:
            raise ValueError(
                f"LoRA rank config mismatch for {module_name}: "
                f"config r={expected_rank}, tensor rank={rank}"
            )
        pairs.append(
            LoRATensorPair(
                module_name=module_name,
                lora_a=lora_a,
                lora_b=lora_b,
                scaling=_lora_scaling(adapter_config, module_name, rank),
            )
        )

    if incomplete:
        raise ValueError(
            "LoRA adapter has incomplete A/B tensor pairs for: "
            + ", ".join(incomplete[:8])
        )
    if not pairs:
        raise ValueError(f"No LoRA tensor pairs found in {weights_path}")
    return pairs


def iter_lora_tensor_module_names(adapter_path: str) -> list[str]:
    weights_path = _find_adapter_weights(adapter_path)
    module_names = set()
    with safe_open(weights_path, framework="pt", device="cpu") as f:
        for tensor_name in f.keys():
            parsed = parse_lora_tensor_name(tensor_name)
            if parsed is not None:
                module_names.add(parsed[0])
    return sorted(module_names)


def validate_lora_adapters_supported(lora_modules: list[str] | None) -> None:
    if not lora_modules:
        return

    for entry in lora_modules:
        spec = parse_lora_module_entry(entry)
        _load_adapter_config(spec.path)
        iter_lora_tensor_module_names(spec.path)


def mark_static_routed_lora_targets(
    model: nn.Module,
    lora_modules: list[str] | None,
) -> None:
    """Mark MoE layers that need unshuffled weights for routed LoRA runtime."""
    if not lora_modules:
        return

    model_modules = dict(model.named_modules())
    marked = 0
    for entry in lora_modules:
        spec = parse_lora_module_entry(entry)
        for module_name in iter_lora_tensor_module_names(spec.path):
            match = _match_routed_expert(module_name)
            if match is None:
                continue
            experts_name = f"{match.group('prefix')}.experts"
            experts = model_modules.get(experts_name)
            if experts is None:
                continue
            if not getattr(experts, "_static_routed_lora_requires_unshuffled", False):
                marked += 1
            experts._static_routed_lora_requires_unshuffled = True

    if marked:
        logger.info(
            "Marked %d MoE layers to preserve unshuffled weights for static routed LoRA",
            marked,
        )


def module_has_static_lora_adapters(module: nn.Module | None) -> bool:
    return bool(getattr(module, "_static_lora_adapters", ()))


def any_module_has_static_lora_adapters(*modules: nn.Module | None) -> bool:
    return any(module_has_static_lora_adapters(module) for module in modules)


def _endswith_module_name(module_name: str, suffix: str) -> bool:
    return module_name == suffix or module_name.endswith(f".{suffix}")


def resolve_lora_target(
    module_name: str,
    model_modules: dict[str, nn.Module],
    packed_modules_mapping: dict[str, Any],
) -> tuple[str, Any | None]:
    if module_name in model_modules:
        return module_name, None

    for checkpoint_name, packed_value in packed_modules_mapping.items():
        if not _endswith_module_name(module_name, checkpoint_name):
            continue
        if isinstance(packed_value, list):
            raise ValueError(
                f"Static LoRA does not support fused checkpoint adapter module "
                f"{module_name!r} mapped through list packed_modules_mapping."
            )
        packed_name, shard_id = packed_value
        target_name = module_name[: -len(checkpoint_name)] + packed_name
        if target_name in model_modules:
            return target_name, shard_id

    if ".mlp.experts." in module_name:
        raise NotImplementedError(
            "Static LoRA does not support routed FusedMoE expert adapter "
            f"modules yet: {module_name}"
        )

    raise KeyError(f"LoRA target module not found in model: {module_name}")


def _slice_or_validate(
    tensor: torch.Tensor,
    dim: int,
    start: int,
    size: int,
    what: str,
) -> torch.Tensor:
    if tensor.size(dim) == size:
        return tensor
    if start + size > tensor.size(dim):
        raise ValueError(
            f"LoRA tensor too small for {what}: shape={tuple(tensor.shape)}, "
            f"dim={dim}, start={start}, size={size}"
        )
    return tensor.narrow(dim, start, size)


def _qkv_output_slice(module: nn.Module, shard_id: str) -> tuple[int, int, int]:
    has_q_gate = len(getattr(module, "output_partition_sizes", [])) == 4
    q_size = module.num_heads * module.head_size
    kv_size = module.num_kv_heads * module.head_size
    if shard_id == "q":
        shard_size = q_size
        shard_offset = q_size if has_q_gate else 0
        shard_rank = module.tp_rank
    elif shard_id == "k":
        shard_size = kv_size
        shard_offset = q_size * 2 if has_q_gate else q_size
        shard_rank = module.tp_rank // module.num_kv_head_replicas
    elif shard_id == "v":
        v_head_size = getattr(module, "v_head_size", module.head_size)
        shard_size = module.num_kv_heads * v_head_size
        shard_offset = q_size * 2 + kv_size if has_q_gate else q_size + kv_size
        shard_rank = module.tp_rank // module.num_kv_head_replicas
    else:
        raise ValueError(f"Unsupported QKV LoRA shard id: {shard_id!r}")
    return shard_offset, shard_rank * shard_size, shard_size


def _slice_qkvg_q_lora_b(module: nn.Module, lora_b: torch.Tensor) -> torch.Tensor:
    q_size = module.num_heads * module.head_size
    shard_size = q_size * 2
    lora_b = _slice_or_validate(
        lora_b,
        dim=0,
        start=module.tp_rank * shard_size,
        size=shard_size,
        what="QKVG q+gate output shard",
    )
    deinterleave = getattr(module, "_deinterleave", None)
    if deinterleave is None:
        raise ValueError(
            "Static LoRA cannot load QKVG q_proj adapters without a "
            "_deinterleave helper on the target module."
        )
    lora_b = deinterleave(lora_b)
    q_part = lora_b.narrow(0, 0, q_size)
    gate_part = lora_b.narrow(0, q_size, q_size)
    return torch.cat([gate_part, q_part], dim=0)


def _validate_lora_output_shape(
    module: nn.Module,
    lora_b: torch.Tensor,
    output_offset: int | None,
) -> None:
    if output_offset is None:
        if lora_b.shape[0] != module.output_size:
            raise ValueError(
                f"LoRA B output mismatch for {module.__class__.__name__}: "
                f"expected {module.output_size}, got {lora_b.shape[0]}"
            )
        return

    if output_offset < 0 or output_offset + lora_b.shape[0] > module.output_size:
        raise ValueError(
            f"LoRA B output slice for {module.__class__.__name__} exceeds the "
            f"module output size: offset={output_offset}, B={tuple(lora_b.shape)}, "
            f"output_size={module.output_size}"
        )


def slice_lora_tensors_for_module(
    module: nn.Module,
    lora_a: torch.Tensor,
    lora_b: torch.Tensor,
    shard_id: Any | None,
) -> tuple[torch.Tensor, torch.Tensor, int | None]:
    tp_dim = getattr(module, "tp_dim", None)
    tp_size = getattr(module, "tp_size", 1)
    tp_rank = getattr(module, "tp_rank", 0)
    output_offset = None

    if tp_dim == 1:
        local_in = module.input_size
        lora_a = _slice_or_validate(
            lora_a, dim=1, start=tp_rank * local_in, size=local_in, what="row input"
        )
    elif tp_dim == 0:
        if shard_id is None:
            local_out = module.output_size
            lora_b = _slice_or_validate(
                lora_b,
                dim=0,
                start=tp_rank * local_out,
                size=local_out,
                what="column output",
            )
        elif isinstance(shard_id, int):
            full_sizes = module.output_sizes
            local_out = full_sizes[shard_id] // tp_size
            output_offset = sum(full_sizes[:shard_id]) // tp_size
            lora_b = _slice_or_validate(
                lora_b,
                dim=0,
                start=tp_rank * local_out,
                size=local_out,
                what=f"packed output shard {shard_id}",
            )
        elif isinstance(shard_id, str) and shard_id in {"q", "k", "v"}:
            has_q_gate = len(getattr(module, "output_partition_sizes", [])) == 4
            if shard_id == "q" and has_q_gate:
                lora_b = _slice_qkvg_q_lora_b(module, lora_b)
                output_offset = 0
            else:
                output_offset, start, shard_size = _qkv_output_slice(module, shard_id)
                lora_b = _slice_or_validate(
                    lora_b,
                    dim=0,
                    start=start,
                    size=shard_size,
                    what=f"QKV output shard {shard_id}",
                )
        else:
            raise ValueError(
                f"Unsupported LoRA shard id {shard_id!r} for {module.__class__}"
            )
    elif shard_id is not None:
        if not isinstance(shard_id, int) or not hasattr(module, "output_sizes"):
            raise ValueError(
                f"Unsupported LoRA shard id {shard_id!r} for replicated module"
            )
        output_offset = sum(module.output_sizes[:shard_id])

    _validate_lora_output_shape(module, lora_b, output_offset)
    return lora_a, lora_b, output_offset


def _lora_dtype_for_module(module: nn.Module) -> torch.dtype:
    weight = getattr(module, "weight", None)
    dtype = getattr(weight, "dtype", None)
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return dtype
    return torch.bfloat16


def _match_routed_expert(module_name: str) -> re.Match[str] | None:
    return _ROUTED_EXPERT_RE.match(module_name)


def _map_routed_expert_id(module: nn.Module, expert_id: int) -> int | None:
    expert_map = getattr(module, "expert_map", None)
    if expert_map is None:
        return expert_id
    if expert_id >= int(expert_map.numel()):
        raise ValueError(
            f"LoRA expert id {expert_id} is outside expert_map with "
            f"{expert_map.numel()} entries"
        )
    local_id = int(expert_map[expert_id].item())
    if local_id == -1:
        return None
    return local_id


def _slice_routed_lora_delta(
    module: nn.Module,
    pair: LoRATensorPair,
    proj: str,
) -> torch.Tensor:
    tp_rank = getattr(module, "tp_rank", 0)
    local_intermediate = module.intermediate_size_per_partition
    compute_dtype = torch.bfloat16
    if proj in {"gate_proj", "up_proj"}:
        lora_b = _slice_or_validate(
            pair.lora_b,
            dim=0,
            start=tp_rank * local_intermediate,
            size=local_intermediate,
            what=f"routed expert {proj} output",
        )
        lora_a = pair.lora_a
    else:
        lora_a = _slice_or_validate(
            pair.lora_a,
            dim=1,
            start=tp_rank * local_intermediate,
            size=local_intermediate,
            what="routed expert down_proj input",
        )
        lora_b = pair.lora_b
    return (
        lora_b.to(dtype=compute_dtype) @ lora_a.to(dtype=compute_dtype)
    ) * pair.scaling


def _dequantize_fp8_blocks(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_n: int,
    block_k: int,
) -> torch.Tensor:
    out_dim, in_dim = weight.shape
    if out_dim % block_n or in_dim % block_k:
        raise ValueError(
            f"FP8 MoE weight shape {tuple(weight.shape)} is not aligned to "
            f"block shape ({block_n}, {block_k})"
        )
    expected_scale = (out_dim // block_n, in_dim // block_k)
    if tuple(scale.shape) != expected_scale:
        raise ValueError(
            f"FP8 MoE scale shape {tuple(scale.shape)} does not match "
            f"{expected_scale} for weight {tuple(weight.shape)}"
        )
    blocks = weight.float().reshape(
        out_dim // block_n,
        block_n,
        in_dim // block_k,
        block_k,
    )
    scale_blocks = scale.float().reshape(
        out_dim // block_n,
        1,
        in_dim // block_k,
        1,
    )
    return (blocks * scale_blocks).reshape(out_dim, in_dim)


def _infer_fp8_block_shape(
    weight: torch.Tensor,
    scale: torch.Tensor,
    block_n: int,
    block_k: int,
) -> tuple[int, int]:
    expected_scale = (weight.shape[0] // block_n, weight.shape[1] // block_k)
    if block_n > 1 and block_k > 1 and tuple(scale.shape) == expected_scale:
        return block_n, block_k
    if weight.shape[0] % scale.shape[0] or weight.shape[1] % scale.shape[1]:
        return block_n, block_k
    return weight.shape[0] // scale.shape[0], weight.shape[1] // scale.shape[1]


def _requantize_fp8_blocks_(
    dst_weight: torch.Tensor,
    dst_scale: torch.Tensor,
    value: torch.Tensor,
    block_n: int,
    block_k: int,
) -> None:
    out_dim, in_dim = value.shape
    blocks = value.float().reshape(
        out_dim // block_n,
        block_n,
        in_dim // block_k,
        block_k,
    )
    fp8_max = torch.finfo(dst_weight.dtype).max
    scale = blocks.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-8) / fp8_max
    qweight = (blocks / scale).clamp(min=-fp8_max, max=fp8_max).to(dst_weight.dtype)
    dst_weight.copy_(qweight.reshape(out_dim, in_dim))
    dst_scale.copy_(scale.squeeze(3).squeeze(1).to(dst_scale.dtype))


def _merge_delta_into_moe_weight_(
    weight: torch.Tensor,
    scale: torch.Tensor | None,
    delta: torch.Tensor,
    block_n: int,
    block_k: int,
) -> None:
    if weight.dtype in (torch.float16, torch.bfloat16, torch.float32):
        weight.add_(delta.to(device=weight.device, dtype=weight.dtype))
        return

    fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
    if weight.dtype not in fp8_dtypes or scale is None:
        raise TypeError(
            "Static routed expert LoRA supports floating-point MoE weights "
            "and FP8 block-quantized MoE weights only"
        )
    block_n, block_k = _infer_fp8_block_shape(weight, scale, block_n, block_k)
    merged = _dequantize_fp8_blocks(weight, scale, block_n, block_k)
    merged.add_(delta.to(device=weight.device, dtype=merged.dtype))
    _requantize_fp8_blocks_(weight, scale, merged, block_n, block_k)


def _merge_routed_expert_lora(
    module_name: str,
    model_modules: dict[str, nn.Module],
    pair: LoRATensorPair,
    adapter_name: str,
) -> tuple[bool, bool]:
    """Return (handled, merged) for routed expert LoRA modules."""
    match = _match_routed_expert(module_name)
    if match is None:
        return False, False

    experts_name = f"{match.group('prefix')}.experts"
    if experts_name not in model_modules:
        raise KeyError(f"LoRA routed expert module not found: {experts_name}")
    experts = model_modules[experts_name]
    local_expert_id = _map_routed_expert_id(experts, int(match.group("expert_id")))
    if local_expert_id is None:
        return True, False

    proj = match.group("proj")
    add_static_routed_lora = getattr(experts, "add_static_routed_lora", None)
    if add_static_routed_lora is not None:
        add_static_routed_lora(
            int(match.group("expert_id")),
            proj,
            pair.lora_a,
            pair.lora_b,
            pair.scaling,
            adapter_name=adapter_name,
        )
        logger.debug(
            "Registered static LoRA adapter %s for routed expert %s",
            adapter_name,
            module_name,
        )
        return True, True

    delta = _slice_routed_lora_delta(experts, pair, proj)
    quant_method = getattr(experts, "quant_method", None)
    block_n = getattr(quant_method, "block_n", 1)
    block_k = getattr(quant_method, "block_k", 1)

    if proj in {"gate_proj", "up_proj"}:
        shard_size = experts.intermediate_size_per_partition
        start = 0 if proj == "gate_proj" else shard_size
        weight = experts.w13_weight.data[local_expert_id, start : start + shard_size, :]
        scale = getattr(experts, "w13_weight_scale", None)
        scale_slice = None
        if scale is not None and scale.dim() == 3:
            if scale.shape[1] % 2:
                raise ValueError(
                    "FP8 MoE w13 scale rows must split evenly between gate "
                    f"and up projections, got shape {tuple(scale.shape)}"
                )
            scale_rows_per_shard = scale.shape[1] // 2
            scale_start = 0 if proj == "gate_proj" else scale_rows_per_shard
            scale_end = scale_start + scale_rows_per_shard
            scale_slice = scale.data[local_expert_id, scale_start:scale_end, :]
        _merge_delta_into_moe_weight_(
            weight,
            scale_slice,
            delta.to(weight.device),
            block_n,
            block_k,
        )
    else:
        weight = experts.w2_weight.data[local_expert_id, :, :]
        scale = getattr(experts, "w2_weight_scale", None)
        scale_slice = None
        if scale is not None and scale.dim() == 3:
            scale_slice = scale.data[local_expert_id]
        _merge_delta_into_moe_weight_(
            weight,
            scale_slice,
            delta.to(weight.device),
            block_n,
            block_k,
        )

    logger.debug(
        "Merged static LoRA adapter %s into routed expert %s",
        adapter_name,
        module_name,
    )
    return True, True


def _looks_like_vocab_parallel_head(module: nn.Module) -> bool:
    return (
        hasattr(module, "weight")
        and hasattr(module, "num_embeddings_per_partition")
        and hasattr(module, "vocab_start_idx")
        and hasattr(module, "vocab_end_idx")
    )


def _slice_vocab_parallel_lora_b(
    module: nn.Module,
    lora_b: torch.Tensor,
) -> torch.Tensor:
    local_vocab = int(module.num_embeddings_per_partition)
    if lora_b.shape[0] == local_vocab:
        return lora_b
    start = int(module.vocab_start_idx)
    return _slice_or_validate(
        lora_b,
        dim=0,
        start=start,
        size=local_vocab,
        what="vocab-parallel lm_head output",
    )


def _merge_vocab_parallel_lora_(
    target_name: str,
    module: nn.Module,
    pair: LoRATensorPair,
    adapter_name: str,
) -> None:
    weight = module.weight.data
    if weight.dim() != 2:
        raise ValueError(
            f"LoRA target {target_name} weight must be 2D, got {tuple(weight.shape)}"
        )
    if pair.lora_a.dim() != 2 or pair.lora_b.dim() != 2:
        raise ValueError(
            f"LoRA tensors for {target_name} must be 2D, got "
            f"A={tuple(pair.lora_a.shape)} B={tuple(pair.lora_b.shape)}"
        )
    if pair.lora_a.shape[0] != pair.lora_b.shape[1]:
        raise ValueError(
            f"LoRA rank mismatch for {target_name}: "
            f"A={tuple(pair.lora_a.shape)} B={tuple(pair.lora_b.shape)}"
        )
    if pair.lora_a.shape[1] != weight.shape[1]:
        raise ValueError(
            f"LoRA A input mismatch for {target_name}: "
            f"expected {weight.shape[1]}, got {pair.lora_a.shape[1]}"
        )

    lora_b = _slice_vocab_parallel_lora_b(module, pair.lora_b)
    if lora_b.shape[0] != weight.shape[0]:
        raise ValueError(
            f"LoRA B output mismatch for {target_name}: "
            f"expected {weight.shape[0]}, got {lora_b.shape[0]}"
        )
    if weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(
            "Static vocab-parallel LoRA supports floating-point lm_head "
            f"weights only, got {weight.dtype}"
        )

    compute_dtype = _lora_dtype_for_module(module)
    delta = (
        lora_b.to(device=weight.device, dtype=compute_dtype)
        @ pair.lora_a.to(device=weight.device, dtype=compute_dtype)
    ) * pair.scaling
    weight.add_(delta.to(dtype=weight.dtype))
    logger.debug(
        "Merged static LoRA adapter %s into vocab-parallel target %s",
        adapter_name,
        target_name,
    )


def _load_vocab_parallel_lora(
    target_name: str,
    module: nn.Module,
    pair: LoRATensorPair,
    adapter_name: str,
) -> None:
    add_lora_adapter = getattr(module, "add_lora_adapter", None)
    if add_lora_adapter is None:
        _merge_vocab_parallel_lora_(target_name, module, pair, adapter_name)
        return

    weight = module.weight
    if weight.dim() != 2:
        raise ValueError(
            f"LoRA target {target_name} weight must be 2D, got {tuple(weight.shape)}"
        )
    if pair.lora_a.shape[1] != weight.shape[1]:
        raise ValueError(
            f"LoRA A input mismatch for {target_name}: "
            f"expected {weight.shape[1]}, got {pair.lora_a.shape[1]}"
        )
    lora_b = _slice_vocab_parallel_lora_b(module, pair.lora_b)
    if lora_b.shape[0] != weight.shape[0]:
        raise ValueError(
            f"LoRA B output mismatch for {target_name}: "
            f"expected {weight.shape[0]}, got {lora_b.shape[0]}"
        )

    dtype = _lora_dtype_for_module(module)
    add_lora_adapter(
        pair.lora_a.to(device=weight.device, dtype=dtype),
        lora_b.to(device=weight.device, dtype=dtype),
        pair.scaling,
        adapter_name=adapter_name,
    )
    logger.debug(
        "Registered LoRA adapter %s for vocab-parallel target %s",
        adapter_name,
        target_name,
    )


def apply_lora_adapters(
    model: nn.Module,
    lora_modules: list[str] | None,
    packed_modules_mapping: dict[str, Any] | None = None,
) -> None:
    if not lora_modules:
        return

    model_modules = dict(model.named_modules())
    packed_modules_mapping = packed_modules_mapping or {}
    loaded_count = 0

    for entry in lora_modules:
        spec = parse_lora_module_entry(entry)
        adapter_pairs = load_lora_tensors(spec.path)
        logger.info(
            "Loading static LoRA adapter %s from %s with %d tensor pairs",
            spec.name,
            spec.path,
            len(adapter_pairs),
        )
        for pair in adapter_pairs:
            routed_handled, routed_merged = _merge_routed_expert_lora(
                pair.module_name,
                model_modules,
                pair,
                spec.name,
            )
            if routed_handled:
                if routed_merged:
                    loaded_count += 1
                continue

            target_name, shard_id = resolve_lora_target(
                pair.module_name,
                model_modules,
                packed_modules_mapping,
            )
            target_module = model_modules[target_name]
            if shard_id is None and _looks_like_vocab_parallel_head(target_module):
                _load_vocab_parallel_lora(
                    target_name,
                    target_module,
                    pair,
                    spec.name,
                )
                loaded_count += 1
                continue

            add_lora_adapter = getattr(target_module, "add_lora_adapter", None)
            if add_lora_adapter is None:
                raise TypeError(
                    f"LoRA target {target_name} does not support static LoRA"
                )

            lora_a, lora_b, output_offset = slice_lora_tensors_for_module(
                target_module,
                pair.lora_a,
                pair.lora_b,
                shard_id,
            )
            device = target_module.weight.device
            dtype = _lora_dtype_for_module(target_module)
            add_lora_adapter(
                lora_a.to(device=device, dtype=dtype),
                lora_b.to(device=device, dtype=dtype),
                pair.scaling,
                output_offset=output_offset,
                adapter_name=spec.name,
            )
            loaded_count += 1

    logger.info("Loaded %d static LoRA tensor pairs", loaded_count)
