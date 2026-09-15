# SPDX-License-Identifier: MIT
"""Replay saved Engram and V4 attention inputs without loading the target model."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open

from atom.model_ops.v4_kernels.paged_decode import sparse_attn_v4_paged_decode


def difference(actual, expected):
    error = (actual.float() - expected.float()).abs()
    return {
        "unequal": int((actual != expected).sum()),
        "elements": actual.numel(),
        "max_error": float(error.max()),
        "relative_l2": float(error.norm() / expected.float().norm().clamp_min(1e-30)),
    }


def engram_gate(residual, projected, weight):
    hc, dim = residual.shape[-2:]
    key = projected[..., : hc * dim].reshape_as(residual).float()
    value = projected[..., hc * dim :].float()
    residual = residual.float()
    residual_mean = residual.square().mean(-1)
    key_mean = key.square().mean(-1)
    rstd = torch.rsqrt(residual_mean + 1e-20) * torch.rsqrt(key_mean + 1e-20)
    dot_sum = (residual * weight * key).sum(-1)
    dot = dot_sum * rstd * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    return {
        "residual_mean": residual_mean,
        "key_mean": key_mean,
        "dot_sum": dot_sum,
        "gate": gate,
        "output": (residual + gate.unsqueeze(-1) * value.unsqueeze(-2)).bfloat16(),
    }


def probe_engram(data):
    values = data["values"]
    label = "layers.1.engram"
    residual = values[label + ".input"]["block"].cuda()
    projected = values[label + ".wkv.output"]["block"].cuda()
    weight = data["engram_weights"][1].cuda()
    block = engram_gate(residual, projected, weight)
    serial = [
        engram_gate(r[None], p[None], weight) for r, p in zip(residual, projected)
    ]
    serial = {name: torch.cat([row[name] for row in serial]) for name in block}
    saved = values[label + ".output"]
    return {
        "frozen_input_block_vs_serial": {
            name: difference(block[name], serial[name]) for name in block
        },
        "reproduces_saved_block": difference(block["output"], saved["block"].cuda()),
        "reproduces_saved_serial": difference(
            serial["output"],
            torch.stack([saved["serial"][i] for i in range(len(residual))]).cuda(),
        ),
    }


def probe_attention(data, model, rank):
    weight_map = json.loads((model / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    result = []
    for layer in (0, 1, 2, 3):
        prefix = f"layers.{layer}.sparse_attn_v4_paged_decode.0"
        values = data["values"]
        query = values[prefix + ".input"]["block"].cuda()
        keys = values[prefix + ".kv"]["block"].cuda()
        indices = values[prefix + ".indices"]["block"]
        lengths = (indices >= 0).sum(-1).tolist()
        rows, heads, dim = query.shape
        pool = keys.flatten(0, 1)
        physical = torch.cat(
            [
                torch.arange(
                    i * keys.shape[1],
                    i * keys.shape[1] + length,
                    device="cuda",
                    dtype=torch.int32,
                )
                for i, length in enumerate(lengths)
            ]
        )
        indptr = (
            torch.tensor([0] + lengths, device="cuda", dtype=torch.int32)
            .cumsum(0)
            .int()
        )
        key = f"layers.{layer}.attn.attn_sink"
        with safe_open(model / weight_map[key], framework="pt") as handle:
            sink = handle.get_tensor(key)[rank * heads : (rank + 1) * heads].cuda()
        block = sparse_attn_v4_paged_decode(
            query, pool, physical, indptr, sink, dim**-0.5
        )
        serial, fixed_shape = [], []
        for i, length in enumerate(lengths):
            ids = physical[sum(lengths[:i]) : sum(lengths[: i + 1])]
            pointer = torch.tensor([0, length], device="cuda", dtype=torch.int32)
            serial.append(
                sparse_attn_v4_paged_decode(
                    query[i : i + 1], pool, ids, pointer, sink, dim**-0.5
                )
            )
            # Duplicate this SAME query and SAME KV list to isolate launch shape.
            repeated_query = query[i : i + 1].expand_as(query).contiguous()
            repeated_pointer = (
                torch.arange(rows + 1, device="cuda", dtype=torch.int32) * length
            )
            fixed_shape.append(
                sparse_attn_v4_paged_decode(
                    repeated_query,
                    pool,
                    ids.repeat(rows),
                    repeated_pointer,
                    sink,
                    dim**-0.5,
                )[i : i + 1]
            )
        result.append(
            {
                "layer": layer,
                "lengths": lengths,
                "frozen_input_block_vs_serial": difference(block, torch.cat(serial)),
                "frozen_input_block_vs_fixed_launch_shape": difference(
                    block, torch.cat(fixed_shape)
                ),
                "reproduces_saved_block": difference(
                    block, values[prefix + ".output"]["block"].cuda()
                ),
            }
        )
    return result


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--model", type=Path, default=Path("/mnt/DeepSeek-V4.1-Flash"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in sorted(args.traces.glob("trace_*_rank*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=True)
        rank = int(path.stem.rsplit("rank", 1)[1])
        records.append(
            {
                "trace": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "engram": probe_engram(data),
                "attention": probe_attention(data, args.model, rank),
            }
        )
    assert records, "No trace inputs found"
    args.output.write_text(
        json.dumps({"completed": True, "records": records}, indent=2) + "\n"
    )
    print(json.dumps(records), flush=True)


if __name__ == "__main__":
    main()
