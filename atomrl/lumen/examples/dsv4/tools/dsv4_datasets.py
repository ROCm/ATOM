# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Dataset helpers for DSV4 GRPO finetune."""

from __future__ import annotations

import subprocess
from pathlib import Path


def ensure_gsm8k(data_dir: str | Path) -> Path:
    """Ensure ``{data_dir}/gsm8k/train.parquet`` exists (hf download if missing)."""
    root = Path(data_dir)
    train_parquet = root / "gsm8k" / "train.parquet"
    if train_parquet.is_file():
        return train_parquet
    root.mkdir(parents=True, exist_ok=True)
    print(f"[dataset] downloading zhuzilin/gsm8k -> {root / 'gsm8k'}")
    subprocess.run(
        ["hf", "download", "zhuzilin/gsm8k", "--local-dir", str(root / "gsm8k")],
        check=True,
    )
    if not train_parquet.is_file():
        raise FileNotFoundError(
            f"gsm8k train.parquet missing after download: {train_parquet}"
        )
    return train_parquet
