"""Quantize Qwen/Qwen3.8-Flash-Next (BF16) into the MXFP4 checkpoint ATOM loads.

MXFP4 here is OCP microscaling: E2M1 values with one E8M0 scale per 32 values
along K, in the Quark layout ATOM reads (``<name>.weight`` uint8 ``[N, K/2]``,
element ``2i`` in the low nibble, and ``<name>.weight_scale`` uint8
``[N, K/32]``).

Rounding, per block of 32 values with largest magnitude ``amax`` (float32):

- the scale is ``2^ceil(log2(max(amax / 6, 1e-12)))``, clamped to the E8M0
  range ``[2^-126, 2^127]``;
- each value is ``x / scale`` rounded to the nearest E2M1 value, ties toward
  zero, saturating at 6; a value that rounds to zero is stored as +0.

Per tensor:

- Linears ATOM runs as MXFP4 (``MXFP4``) are quantized. Routed experts are split
  into ``experts.<e>.{gate,up,down}_proj``.
- ``lm_head`` and the hyper-connection mix weights (``ROUNDED``) are rounded to
  MXFP4 values the same way and stored as BF16, since ATOM runs them in BF16.
  Every MXFP4 value is exact in BF16.
- Everything else is copied unchanged.

``config.json`` declares MXFP8 activations, so ATOM runs its W4A8 linears.

Usage::

    python convert_mxfp4.py --hf HF_DIR --out OUT_DIR [--workers 16]
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# Linears ATOM's qwen4 runs as MXFP4 linears or MoE.
MXFP4 = re.compile(
    r"\.(mlp\.experts\.(gate_up|down)_proj"
    r"|mlp\.shared_expert\.(gate|up|down)_proj"
    r"|self_attn\.[qkvo]_proj"
    r"|self_attn\.indexer\.index_qk_proj"
    r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
    r"|ple\.(key|value)_proj)$"
)

# Weights rounded to MXFP4 values but stored as BF16.
ROUNDED = re.compile(r"(^lm_head|\.input_mix_weight_(down|up))$")

# Modules that stay BF16 in the converted checkpoint. GDN B/A, the router and the
# hyper-connections have no quantized path in ATOM's qwen4; the PLE n-gram table
# is a BF16 embedding; the MTP fc projections stay BF16.
EXCLUDE = [
    "*.linear_attn.in_proj_b",
    "*.linear_attn.in_proj_a",
    "*.ple_embedding*",
    "*.mlp.gate",
    "*.shared_expert_gate",
    "*hyper_connection*",
    "*fc_embedding",
    "*fc_hidden",
    "lm_head",
    "*.embed_tokens",
    "re:^(model\\.)?visual\\..*",
]


def quantization_config():
    mx = {"qscheme": "per_group", "group_size": 32, "scale_format": "e8m0"}
    return {
        "quant_method": "quark",
        "global_quant_config": {
            "weight": {**mx, "dtype": "fp4_e2m1", "is_dynamic": False},
            # E4M3 with group-32 E8M0 scales is MXFP8
            "input_tensors": {**mx, "dtype": "fp8_e4m3", "is_dynamic": True},
            "output_tensors": None,
        },
        "layer_quant_config": {},
        "exclude": EXCLUDE,
    }


def tensor_index(directory):
    """Tensor name -> safetensors file, over every shard in ``directory``."""
    out = {}
    for path in sorted(glob.glob(os.path.join(directory, "*.safetensors"))):
        with safe_open(path, "pt") as f:
            # `safe_open` has no `__iter__`; `.keys()` is required.
            for name in f.keys():  # noqa: SIM118
                out[name] = path
    return out


def read(index, name):
    with safe_open(index[name], "pt") as f:
        return f.get_tensor(name)


# Rounding thresholds between consecutive E2M1 magnitudes 0, 0.5, 1, 1.5, 2, 3,
# 4, 6: a magnitude above the i-th threshold gets at least code i + 1, so a tie
# keeps the smaller code.
FP4_THRESHOLDS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
# Floor of amax / 6: an all-but-zero block gets scale 2^-39 and all-zero codes.
MIN_RATIO = 1e-12


def quantize(weight):
    """BF16 ``[N, K]`` -> (uint8 ``[N, K/2]``, E8M0 uint8 ``[N, K/32]``)."""
    N, K = weight.shape
    if K % 32:
        raise ValueError(f"K = {K} is not a multiple of 32")
    x = weight.float().view(N, K // 32, 32)
    bits = (x.abs().amax(-1) / 6).clamp_min(MIN_RATIO).view(torch.int32)
    # E8M0 exponent of the ratio, rounded up when it is not a power of two.
    exponent = ((bits >> 23) & 0xFF) + ((bits & 0x7FFFFF) != 0).to(torch.int32)
    exponent = exponent.clamp(1, 254)
    scale = (exponent << 23).view(torch.float32)
    magnitude = (x / scale.unsqueeze(-1)).abs()
    codes = torch.zeros_like(magnitude, dtype=torch.uint8)
    for threshold in FP4_THRESHOLDS:
        codes += (magnitude > threshold).to(torch.uint8)
    negative = (x.view(torch.int32) < 0) & (codes > 0)
    codes = (codes | (negative.to(torch.uint8) << 3)).view(N, K)
    packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return packed.contiguous(), exponent.to(torch.uint8).contiguous()


FP4_VALUES = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)


def dequantize(packed, scales):
    """uint8 ``[N, K/2]`` + E8M0 ``[N, K/32]`` -> fp32 ``[N, K]``."""
    codes = torch.stack([packed & 15, packed >> 4], dim=-1).reshape(packed.shape[0], -1)
    values = FP4_VALUES[codes.long()].view(packed.shape[0], -1, 32)
    scale = (scales.to(torch.int32) << 23).view(torch.float32)
    return (values * scale.unsqueeze(-1)).view(packed.shape[0], -1)


def round_to_mxfp4(weight):
    """BF16 ``[N, K]`` -> the same tensor rounded to MXFP4 values, in BF16."""
    values = dequantize(*quantize(weight))
    out = values.to(torch.bfloat16)
    if not torch.equal(out.float(), values):
        raise ValueError("an MXFP4 value is not exact in BF16")
    return out


def group_of(name):
    """Output shard of a tensor: one per decoder / MTP layer, plus visual and rest."""
    m = re.match(r"(model\.language_model|mtp)\.layers\.(\d+)\.", name)
    if m:
        return f"{'mtp' if m.group(1) == 'mtp' else 'layer'}{int(m.group(2)):03d}"
    if name.startswith("model.visual."):
        return "visual"
    return "rest"


def convert_group(args):
    """Write one output shard. Returns ``{name: (dtype, shape)}`` and counters."""
    group, names, hf_dir, out_dir = args
    hf = tensor_index(hf_dir)
    out, counts = {}, {"mxfp4": 0, "rounded_bf16": 0, "copied": 0}
    for name in names:
        base = name.removesuffix(".weight")
        if ROUNDED.search(base):
            out[name] = round_to_mxfp4(read(hf, name))
            counts["rounded_bf16"] += 1
            continue
        if not MXFP4.search(base):
            out[name] = read(hf, name)
            counts["copied"] += 1
            continue
        weight = read(hf, name)
        if weight.dim() == 3:
            # Stacked routed experts [E, N, K]; gate_up holds [gate; up].
            stem = base.rsplit(".", 1)[0]  # ...mlp.experts
            for e in range(weight.shape[0]):
                if base.endswith("gate_up_proj"):
                    gate, up = weight[e].chunk(2)
                    parts = {"up_proj": up, "gate_proj": gate}
                else:
                    parts = {"down_proj": weight[e]}
                for proj, w in parts.items():
                    packed, scales = quantize(w)
                    out[f"{stem}.{e}.{proj}.weight"] = packed
                    out[f"{stem}.{e}.{proj}.weight_scale"] = scales
                    counts["mxfp4"] += 1
            continue
        out[name], out[base + ".weight_scale"] = quantize(weight)
        counts["mxfp4"] += 1
    path = os.path.join(out_dir, f"model-{group}.safetensors")
    save_file(out, path, metadata={"format": "pt"})
    meta = {k: (str(v.dtype), tuple(v.shape)) for k, v in out.items()}
    return group, os.path.basename(path), meta, counts


def write_config(hf_dir, out_dir):
    with open(os.path.join(hf_dir, "config.json")) as f:
        config = json.load(f)
    config["quantization_config"] = quantization_config()
    # ATOM's PLE asks for the n-gram table's dtype once the model is quantized.
    config["text_config"]["ple_embedding_dtype"] = "bfloat16"
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    for path in glob.glob(os.path.join(hf_dir, "*")):
        name = os.path.basename(path)
        if (
            os.path.isfile(path)
            and not name.endswith(".safetensors")
            and name not in ("config.json", "model.safetensors.index.json")
        ):
            shutil.copy2(path, os.path.join(out_dir, name))


def convert(hf_dir, out_dir, workers):
    os.makedirs(out_dir, exist_ok=True)
    groups = {}
    for name in tensor_index(hf_dir):
        groups.setdefault(group_of(name), []).append(name)
    jobs = [(g, names, hf_dir, out_dir) for g, names in sorted(groups.items())]
    weight_map, totals, start = {}, {}, time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for group, shard, meta, counts in pool.map(convert_group, jobs):
            for name in meta:
                weight_map[name] = shard
            for k, v in counts.items():
                totals[k] = totals.get(k, 0) + v
            print(f"[{time.time() - start:7.1f}s] {group}: {counts}", flush=True)
    size = sum(
        os.path.getsize(os.path.join(out_dir, s)) for s in set(weight_map.values())
    )
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(
            {"metadata": {"total_size": size}, "weight_map": weight_map}, f, indent=2
        )
    write_config(hf_dir, out_dir)
    print(f"done: {len(weight_map)} tensors, {size / 2**30:.1f} GiB, {totals}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hf", required=True, help="Qwen/Qwen3.8-Flash-Next (BF16)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(max(1, (os.cpu_count() or 1) // args.workers))
    convert(args.hf, args.out, args.workers)


if __name__ == "__main__":
    sys.exit(main())
