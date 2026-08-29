#!/usr/bin/env python3
"""Apply the AITER-side pieces of the DSV4-0731 MI308X recipe."""

import argparse
import csv
import subprocess
from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    if new in text:
        return
    if old not in text:
        raise RuntimeError(f"expected source anchor not found in {path}")
    path.write_text(text.replace(old, new, 1))


def merge_csv(destination: Path, additions: list[Path], keys: tuple[str, ...]) -> None:
    with destination.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        if fields is None:
            raise RuntimeError(f"missing CSV header in {destination}")
        rows = list(reader)

    merged = {tuple(row[key] for key in keys): row for row in rows}
    for addition in additions:
        with addition.open(newline="") as stream:
            for row in csv.DictReader(stream):
                normalized = {field: row.get(field, "") for field in fields}
                merged[tuple(normalized[key] for key in keys)] = normalized

    with destination.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(merged.values())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--aiter-root", default="/app/aiter-test")
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    root = Path(args.aiter_root)
    assets = args.assets

    patch = assets / "hca-plan1024-nw4.patch"
    check = subprocess.run(
        ["patch", "--dry-run", "-p1", "-i", str(patch)], cwd=root, capture_output=True
    )
    if check.returncode == 0:
        subprocess.run(["patch", "-p1", "-i", str(patch)], cwd=root, check=True)
    elif b"Reversed (or previously applied)" not in check.stdout + check.stderr:
        raise RuntimeError((check.stdout + check.stderr).decode(errors="replace"))

    mqa = root / "aiter/ops/flydsl/kernels/mqa_logits/fp8_mqa_logits.py"
    replace_once(
        mqa,
        "    if tag not in _VARIANT_BUILDERS:\n",
        "    if tag == \"mfma_r4_adaptive\":\n"
        "        tag = \"mfma_r4_w2\" if seq_len_kv <= 8192 else \"mfma_r4_w4\"\n"
        "    if tag not in _VARIANT_BUILDERS:\n",
    )

    mhc = root / "aiter/ops/mhc.py"
    replace_once(
        mhc,
        "def _mhc_fused_config_gfx942_80(m, hidden_size, num_cu):\n"
        "    tile_k = 32 if (hidden_size <= 4096 and m <= 128) else 64\n",
        "def _mhc_fused_config_gfx942_80(m, hidden_size, num_cu):\n"
        "    # DSV4-0731 MI308X long-prefill optimum.\n"
        "    if hidden_size == 16384 and m >= 8192:\n"
        "        return 8, 16, 32, 64\n"
        "    tile_k = 32 if (hidden_size <= 4096 and m <= 128) else 64\n",
    )

    sparse = root / "aiter/ops/triton/_triton_kernels/attention/sparse_attention_dsv4.py"
    replace_once(sparse, "        for BLOCK_H in [32, 64]\n", "        for BLOCK_H in [16, 32, 64]\n")

    configs = root / "aiter/configs"
    merge_csv(
        configs / "a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
        [
            assets / "qkv_pad2176.csv",
            assets / "dense8k16k.csv",
            assets / "blockscale_m64000.csv",
            assets / "blockscale_m131072.csv",
            assets / "blockscale_m54784.csv",
        ],
        ("gfx", "cu_num", "M", "N", "K"),
    )
    merge_csv(
        configs / "model_configs/dsv4_bf16_tuned_gemm.csv",
        [assets / "bf16_longchunk.csv"],
        ("gfx", "cu_num", "M", "N", "K", "bias", "dtype", "outdtype", "scaleAB", "bpreshuffle"),
    )
    merge_csv(
        configs / "tuned_fmoe.csv",
        [assets / "fmoe_m131072.csv"],
        ("cu_num", "token", "model_dim", "inter_dim", "expert", "topk", "act_type", "dtype", "q_dtype_a", "q_dtype_w", "q_type", "use_g1u1", "doweight_stage1"),
    )


if __name__ == "__main__":
    main()
