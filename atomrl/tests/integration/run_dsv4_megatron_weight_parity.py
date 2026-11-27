# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Compare selected Megatron-exported weights with the source checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


def _index(root: Path) -> dict[str, str]:
    return json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]


def _load(root: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(root / weight_map[name], framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def compare_exports(source_dir: Path, exported_dir: Path) -> dict[str, Any]:
    source_map = _index(source_dir)
    exported_map = _index(exported_dir)
    reports = []
    first_mismatch = None
    matched = 0
    for name in exported_map:
        item: dict[str, Any] = {"name": name}
        if name not in source_map:
            item["status"] = "missing_from_source"
        else:
            source = _load(source_dir, source_map, name)
            exported = _load(exported_dir, exported_map, name)
            item["source_shape"] = list(source.shape)
            item["exported_shape"] = list(exported.shape)
            if source.shape != exported.shape:
                item["status"] = "shape_mismatch"
            elif torch.equal(source, exported):
                item["status"] = "exact"
                matched += 1
            else:
                diff = source.float() - exported.float()
                item.update(
                    status="value_mismatch",
                    max_abs_diff=float(diff.abs().max().item()),
                    mean_abs_diff=float(diff.abs().mean().item()),
                )
        reports.append(item)
        if item["status"] != "exact" and first_mismatch is None:
            first_mismatch = item
    return {
        "compared": len(reports),
        "matched": matched,
        "mismatched": len(reports) - matched,
        "first_mismatch": first_mismatch,
        "weights": reports,
    }


def _default_names(source_dir: Path) -> list[str]:
    available = _index(source_dir)
    suffixes = (
        "hc_attn_base",
        "hc_attn_fn",
        "hc_attn_scale",
        "hc_ffn_base",
        "hc_ffn_fn",
        "hc_ffn_scale",
        "attn.attn_sink",
        "attn_norm.weight",
        "ffn_norm.weight",
        "attn.q_norm.weight",
        "attn.kv_norm.weight",
        "attn.wq_a.weight",
        "attn.wq_b.weight",
        "attn.wkv.weight",
        "attn.wo_a.weight",
        "attn.wo_b.weight",
        "ffn.gate.weight",
        "ffn.gate.bias",
        "ffn.gate.tid2eid",
        "ffn.experts.0.w1.weight",
        "ffn.experts.0.w2.weight",
        "ffn.experts.0.w3.weight",
        "ffn.experts.255.w1.weight",
        "ffn.experts.255.w2.weight",
        "ffn.experts.255.w3.weight",
        "attn.compressor.ape",
        "attn.compressor.norm.weight",
        "attn.compressor.wgate.weight",
        "attn.compressor.wkv.weight",
        "attn.indexer.compressor.ape",
        "attn.indexer.compressor.norm.weight",
        "attn.indexer.compressor.wgate.weight",
        "attn.indexer.compressor.wkv.weight",
        "attn.indexer.weights_proj.weight",
        "attn.indexer.wq_b.weight",
    )
    names = [
        name
        for name in (
            "embed.weight",
            "norm.weight",
            "head.weight",
            "hc_head_base",
            "hc_head_fn",
            "hc_head_scale",
        )
        if name in available
    ]
    for layer in (0, 10, 22, 42):
        for suffix in suffixes:
            name = f"layers.{layer}.{suffix}"
            if name in available:
                names.append(name)
    return names


def _download_export(source_url: str, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    index_url = source_url.rstrip("/") + "/model.safetensors.index.json"
    urllib.request.urlretrieve(index_url, destination / "model.safetensors.index.json")
    weight_map = _index(destination)
    for filename in sorted(set(weight_map.values())):
        urllib.request.urlretrieve(
            source_url.rstrip("/") + "/" + filename,
            destination / filename,
        )


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", required=True)
    parser.add_argument("--export-dir", default="/dev/shm/dsv4_megatron_weight_parity")
    parser.add_argument(
        "--download-dir", default="/runtime/dsv4_megatron_weight_parity"
    )
    return parser.parse_known_args()


def main() -> None:
    from atomrl.core.config import AtomRLConfig
    from atomrl.trainer.main import _setup_logging
    from atomrl.trainer.rl_trainer import RLTrainer

    args, config_args = _parse_args()
    sys.argv = [sys.argv[0], *config_args]
    os.environ["ATOMRL_FORWARD_ONLY_INIT"] = "1"
    os.environ["ATOMRL_SKIP_ROLLOUT_INIT"] = "1"
    _setup_logging()
    config = AtomRLConfig.from_cli()
    config.checkpointing.resume = False
    config.eval.enabled = False
    config.logger.wandb_enabled = False
    config.moe.r3.enabled = False
    config.weight_sync.enabled = False
    config.controller.ray.rollout.num_workers = 0
    config.controller.ray.rollout.process_on_nodes = []

    source_dir = Path(config.policy.model_name)
    names = _default_names(source_dir)
    trainer = RLTrainer(config)
    try:
        trainer.setup()
        if trainer._actor_wg is None:
            raise RuntimeError("Megatron actor worker group is unavailable")
        results = trainer._actor_wg.execute_all_sync(
            "export_state_dict_safetensors",
            sync_dir=args.export_dir,
            include_names=names,
        )
        writer = next(result for result in results if result["writer"])
        exported_dir = Path(args.download_dir)
        _download_export(writer["weight_url"], exported_dir)
        report = compare_exports(source_dir, exported_dir)
        report["requested"] = len(names)
        report["export"] = writer
        Path(args.output).write_text(json.dumps(report, indent=2))
        print(
            "MEGATRON_WEIGHT_PARITY_JSON="
            + json.dumps(
                {
                    k: report[k]
                    for k in (
                        "requested",
                        "compared",
                        "matched",
                        "mismatched",
                        "first_mismatch",
                    )
                }
            ),
            flush=True,
        )
    finally:
        trainer.cleanup()


if __name__ == "__main__":
    main()
