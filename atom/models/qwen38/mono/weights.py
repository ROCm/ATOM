# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Reading the Qwen3.8-Flash-Next Quark MXFP4 checkpoint for Mono's layer programs.

``load_layer`` reads one decoder layer (index ``geo.layers``: the MTP drafter's) into a ``QuantizedLayer``, each
linear by its stored dtype (``as_linear``): FP8 rows with an fp32 row scale -> ``FP8RowWeight``, MXFP4 -> ``MXLinear``,
BF16 -> the tensor. ATOM quantizes the target's attention and GDN projections online (PTPC FP8,
``aiter.get_hip_quant``; the indexer projection and ``mtp.*`` stay BF16): ``load_layer`` does the same (``_ptpc_fp8``),
bit for bit. GDN ``in_proj_b`` / ``in_proj_a`` are retained as ``bf16(E4M3 x scale)`` for GDNCore's bf16 b / a path.
The PLE layer is loaded as a delta layer: its PLE block runs Native. ``own_experts`` builds Mono's A8W4 copy of a target
layer's experts (ATOM's TP1 A4W4 tensors keep the intermediate unpadded).
"""

import json
from functools import lru_cache
from pathlib import Path

import torch
from safetensors import safe_open

from atom.models.qwen38.mono.kernels.weights import aiter_moe
from atom.models.qwen38.mono.kernels.weights.fp8row import FP8RowWeight
from atom.models.qwen38.mono.kernels.weights.layer import MXLinear, QuantizedLayer

PREFIX = "model.language_model."
MTP_PREFIX = "mtp."
HC_RETAINED = tuple(
    f"{hc}.{part}"
    for hc in ("attn_hyper_connection", "mlp_hyper_connection")
    for part in (
        "hc_norm",
        "input_mix_weight_down",
        "input_mix_weight_up",
        "block_inject_weight",
    )
)
GDN_LINEARS = (
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
)
GDN_BA = ("linear_attn.in_proj_b", "linear_attn.in_proj_a")
GDN_RETAINED = GDN_BA + (
    "linear_attn.conv1d",
    "linear_attn.A_log",
    "linear_attn.dt_bias",
    "linear_attn.norm",
)
QSA_LINEARS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.indexer.index_qk_proj",
)
QSA_RETAINED = (
    "self_attn.q_norm",
    "self_attn.k_norm",
    "self_attn.indexer.q_layernorm",
    "self_attn.indexer.k_layernorm",
)
MOE_LINEARS = (
    "mlp.shared_expert.gate_proj",
    "mlp.shared_expert.up_proj",
    "mlp.shared_expert.down_proj",
)
MOE_RETAINED = ("mlp.gate", "mlp.shared_expert_gate")
EXPERT_PROJS = ("gate_proj", "up_proj", "down_proj")
# ATOM's online PTPC FP8 targets (``--online_quant_config``: ``*indexer*`` and ``mtp.*`` excluded)
ONLINE_PTPC = GDN_LINEARS + QSA_LINEARS[:4]
_BARE = ("A_log", "dt_bias")


def layer_prefix(index: int, geo) -> str:
    """Checkpoint prefix of decoder layer ``index``; ``geo.layers`` is the MTP layer."""
    return (
        f"{MTP_PREFIX}layers.0." if index == geo.layers else f"{PREFIX}layers.{index}."
    )


def _key(prefix: str, name: str) -> str:
    return f"{prefix}{name}" if name.endswith(_BARE) else f"{prefix}{name}.weight"


@lru_cache(maxsize=4)
def _weight_map(checkpoint: str) -> dict[str, str]:
    return json.loads((Path(checkpoint) / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]


def read_tensors(keys: list[str], checkpoint, device) -> dict[str, torch.Tensor]:
    weight_map = _weight_map(str(checkpoint))
    by_file: dict[str, list[str]] = {}
    for key in keys:
        by_file.setdefault(weight_map[key], []).append(key)
    out = {}
    for file, file_keys in by_file.items():
        with safe_open(
            str(Path(checkpoint) / file), framework="pt", device=str(device)
        ) as f:
            for key in file_keys:
                out[key] = f.get_tensor(key)
    return out


def _ptpc_fp8(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """ATOM's online PTPC weight quantization (``aiter.get_hip_quant(per_Token)``): (E4M3 [N, K], fp32 [N])."""
    from aiter import QuantType, dtypes, get_hip_quant

    q, s = get_hip_quant(QuantType.per_Token)(
        weight.contiguous(), quant_dtype=dtypes.fp8
    )
    return q.view(torch.float8_e4m3fn), s.reshape(-1).to(torch.float32)


def as_linear(name: str, weight: torch.Tensor, scale: torch.Tensor | None):
    """One stored linear by dtype: FP8 rows, MXFP4 or BF16."""
    if weight.dtype == torch.float8_e4m3fn:
        if (
            scale is None
            or scale.dtype != torch.float32
            or scale.numel() != weight.shape[0]
        ):
            raise ValueError(f"{name}: FP8 weights need an fp32 per-row scale, got "
                             f"{None if scale is None else (scale.dtype, tuple(scale.shape))}")  # fmt: skip
        return FP8RowWeight.from_checkpoint(weight, scale)
    if weight.dtype == torch.uint8:
        if scale is None or scale.dtype != torch.uint8:
            raise ValueError(f"{name}: packed E2M1 weights need E8M0 codes")
        return MXLinear(weight, scale)
    if weight.dtype == torch.bfloat16 and scale is None:
        return weight
    got = None if scale is None else scale.dtype
    raise ValueError(f"{name}: unsupported stored linear {weight.dtype} (scale {got})")


def dequant_ba(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """GDN b / a: an FP8 per-channel weight -> ``bf16(E4M3 x scale)`` [N, K] (GDNCore's ``ba_w``)."""
    return FP8RowWeight.from_checkpoint(weight, scale).dequantize().to(torch.bfloat16)


def load_layer(
    index: int, checkpoint, geo, kind: str, experts: bool, device="cuda"
) -> QuantizedLayer:
    """Layer ``index`` as stored, as a ``kind`` layer ("delta" | "qsa" | "mtp"); ``experts=False`` leaves the routed
    experts out (the target binds Mono's own copies, ``own_experts``)."""
    prefix = layer_prefix(index, geo)
    retained_names = HC_RETAINED + MOE_RETAINED
    if kind == "delta":
        linear_names, retained_names = (
            GDN_LINEARS + MOE_LINEARS,
            GDN_RETAINED + retained_names,
        )
    else:
        linear_names, retained_names = (
            QSA_LINEARS + MOE_LINEARS,
            QSA_RETAINED + retained_names,
        )
    weight_map = _weight_map(str(checkpoint))
    keys = [_key(prefix, n) for n in retained_names]
    keys += [f"{prefix}{n}.weight_scale" for n in retained_names if n in GDN_BA]
    for n in linear_names:
        keys.append(f"{prefix}{n}.weight")
        if f"{prefix}{n}.weight_scale" in weight_map:
            keys.append(f"{prefix}{n}.weight_scale")
    for e in range(geo.experts if experts else 0):
        for p in EXPERT_PROJS:
            keys += [
                f"{prefix}mlp.experts.{e}.{p}.weight",
                f"{prefix}mlp.experts.{e}.{p}.weight_scale",
            ]
    keys = [k for k in keys if k in weight_map or not k.endswith(".weight_scale")]
    raw = read_tensors(keys, checkpoint, device)
    linears = {}
    for n in linear_names:
        w, s = raw[f"{prefix}{n}.weight"], raw.get(f"{prefix}{n}.weight_scale")
        if (
            n in ONLINE_PTPC
            and w.dtype == torch.bfloat16
            and s is None
            and index != geo.layers
        ):
            w, s = _ptpc_fp8(w)
        linears[n] = as_linear(n, w, s)
    for p in EXPERT_PROJS if experts else ():
        parts = [f"{prefix}mlp.experts.{e}.{p}" for e in range(geo.experts)]
        if (
            raw[parts[0] + ".weight"].dtype == torch.bfloat16
        ):  # (the BF16 drafter's experts: [E x N, K] bf16)
            linears[f"mlp.experts.{p}"] = torch.cat([raw[k + ".weight"] for k in parts])
            continue
        linears[f"mlp.experts.{p}"] = MXLinear(
            torch.cat([raw[k + ".weight"] for k in parts]),
            torch.cat([raw[k + ".weight_scale"] for k in parts]),
        )
    retained = {n: raw[_key(prefix, n)] for n in retained_names}
    for n in GDN_BA:
        if n in retained:
            scale = raw.get(f"{prefix}{n}.weight_scale")
            if scale is None and retained[n].dtype == torch.bfloat16:
                # ATOM quantizes b / a online too
                retained[n], scale = _ptpc_fp8(retained[n])
            retained[n] = dequant_ba(retained[n], scale)
    if "linear_attn.conv1d" in retained:
        conv = retained["linear_attn.conv1d"]
        # [C, 1, K] -> [C, K]
        retained["linear_attn.conv1d"] = conv.reshape(conv.shape[0], -1)
    return QuantizedLayer(index, kind, linears, retained)


def own_experts(
    index: int, checkpoint, geo, args, device="cuda"
) -> dict[str, torch.Tensor]:
    """Mono's A8W4 expert tensors of target layer ``index`` under the MoE blocks' arg names ``args`` (``MoEArgs``):
    the intermediate padded to 256, gate / up interleaved, the fused shared expert last. A copy, ~1.5 GB a layer.
    """
    prefix = layer_prefix(index, geo)
    keys = []
    for e in range(geo.experts):
        for p in EXPERT_PROJS:
            keys += [
                f"{prefix}mlp.experts.{e}.{p}.weight",
                f"{prefix}mlp.experts.{e}.{p}.weight_scale",
            ]
    for p in EXPERT_PROJS:
        keys += [
            f"{prefix}mlp.shared_expert.{p}.weight",
            f"{prefix}mlp.shared_expert.{p}.weight_scale",
        ]
    raw = read_tensors(keys, checkpoint, device)
    routed = []
    for p in EXPERT_PROJS:
        parts = [f"{prefix}mlp.experts.{e}.{p}" for e in range(geo.experts)]
        routed.append(MXLinear(torch.cat([raw[k + ".weight"] for k in parts]),
                               torch.cat([raw[k + ".weight_scale"] for k in parts])))  # fmt: skip
    shared = tuple(
        MXLinear(
            raw[f"{prefix}mlp.shared_expert.{p}.weight"],
            raw[f"{prefix}mlp.shared_expert.{p}.weight_scale"],
        )
        for p in EXPERT_PROJS
    )
    del raw
    t = aiter_moe.atom_experts(*routed, geo.experts, shared)
    for name in ("w13_scale", "w2_scale"):
        t[name].swizzle = "none"
    return {
        args.w13: t["w13"],
        args.s13: t["w13_scale"],
        args.w2: t["w2"],
        args.s2: t["w2_scale"],
    }
