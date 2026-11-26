# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Generate fake_rollout.pt for native DSV4 GRPO finetune (debug-train-only).

Wraps Miles ``scripts.gen_fake_rollout_data`` when available; falls back to a
minimal torch.save of legacy random samples.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _default_output() -> str:
    return os.environ.get(
        "FAKE_ROLLOUT_DATA",
        f"{os.environ.get('MODEL_DIR', '/root/models')}/fake_rollout.pt",
    )


def prepare(
    *,
    output_path: str | None = None,
    model_dir: str | None = None,
    model_name: str | None = None,
    data_dir: str | None = None,
    miles_dir: str | None = None,
) -> str:
    out = output_path or _default_output()
    if Path(out).is_file():
        print(f"[rollout] already present — skipping: {out}")
        return out

    model_dir = model_dir or os.environ.get("MODEL_DIR", "/root/models")
    model_name = model_name or os.environ.get(
        "MODEL_NAME", "DeepSeek-V4-Flash-FP8-4layer"
    )
    data_dir = data_dir or os.environ.get("DATA_DIR", "/root/datasets")
    miles_dir = miles_dir or os.environ.get("MILES_DIR", "/workspace/miles")

    miles_path = Path(miles_dir)
    if miles_path.is_dir():
        try:
            sys.path.insert(0, str(miles_path))
            from scripts.gen_fake_rollout_data import (  # noqa: WPS433
                make_realistic_samples,
                make_samples,
                save_rollout_data,
            )
        except ImportError as exc:
            print(
                f"[rollout] Miles rollout helper unavailable ({exc}) — using minimal legacy rollout"
            )
        else:
            bf16_path = f"{model_dir}/{model_name}-bf16"
            data_path = f"{data_dir}/gsm8k/train.parquet"
            use_realistic = os.environ.get("SMOKE_LEGACY_FAKE_ROLLOUT", "0") != "1"
            if use_realistic and Path(bf16_path).is_dir() and Path(data_path).is_file():
                print(f"[rollout] generating realistic rollout from {data_path}")
                samples = make_realistic_samples(
                    model_path=bf16_path,
                    data_path=data_path,
                    n_prompts=int(os.environ.get("ROLLOUT_N_PROMPTS", "32")),
                    n_per_prompt=int(os.environ.get("ROLLOUT_N_PER_PROMPT", "8")),
                    response_len=int(os.environ.get("ROLLOUT_RESPONSE_LEN", "64")),
                )
            else:
                print("[rollout] generating legacy random rollout samples")
                samples = make_samples(
                    n_prompts=int(os.environ.get("ROLLOUT_N_PROMPTS", "32")),
                    n_per_prompt=int(os.environ.get("ROLLOUT_N_PER_PROMPT", "8")),
                )
            save_rollout_data(samples, out)
            return out

    import torch

    print("[rollout] Miles not mounted — writing minimal legacy rollout")
    n_prompts = int(os.environ.get("ROLLOUT_N_PROMPTS", "32"))
    n_per_prompt = int(os.environ.get("ROLLOUT_N_PER_PROMPT", "8"))
    prompt_len = 20
    response_len = int(os.environ.get("ROLLOUT_RESPONSE_LEN", "64"))
    samples = []
    for g in range(n_prompts):
        for s in range(n_per_prompt):
            reward = float(s) / n_per_prompt - 0.5
            prompt_tokens = [(100 + g * 7 + j) % 128000 for j in range(prompt_len)]
            resp_tokens = [(200 + s * 13 + j) % 128000 for j in range(response_len)]
            samples.append(
                {
                    "group_index": g,
                    "tokens": prompt_tokens + resp_tokens,
                    "response_length": response_len,
                    "reward": reward,
                    "rollout_log_probs": [-0.5 - 0.01 * j for j in range(response_len)],
                }
            )
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"rollout_id": 1, "metadata": {}, "samples": samples}, out)
    print(f"[rollout] saved {len(samples)} samples → {out}")
    return out


if __name__ == "__main__":
    prepare()
