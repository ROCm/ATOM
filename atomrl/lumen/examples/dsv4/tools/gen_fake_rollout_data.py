# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Generate fake_rollout.pt for native DSV4 GRPO finetune."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


@dataclass
class RolloutSampleDict:
    group_index: int
    index: int
    prompt: str
    tokens: list[int]
    response_length: int
    reward: float
    rollout_log_probs: list[float]
    status: str = "completed"
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_samples(
    n_prompts: int = 32,
    n_per_prompt: int = 8,
    prompt_len: int = 20,
    response_len: int = 64,
) -> list[RolloutSampleDict]:
    samples: list[RolloutSampleDict] = []
    for g in range(n_prompts):
        for s in range(n_per_prompt):
            reward = float(s) / n_per_prompt - 0.5
            prompt_tokens = [(100 + g * 7 + j) % 128000 for j in range(prompt_len)]
            resp_tokens = [(200 + s * 13 + j) % 128000 for j in range(response_len)]
            samples.append(
                RolloutSampleDict(
                    group_index=g,
                    index=s,
                    prompt="fake",
                    tokens=prompt_tokens + resp_tokens,
                    response_length=response_len,
                    reward=reward,
                    rollout_log_probs=[-0.5 - 0.01 * j for j in range(response_len)],
                    label="fake",
                )
            )
    return samples


def _read_rows(data_path: Path, *, n_prompts: int) -> list[dict]:
    suffix = data_path.suffix.lower()
    rows: list[dict] = []
    if suffix == ".parquet":
        import pyarrow.parquet as pq

        table = pq.read_table(data_path)
        for i in range(min(n_prompts, table.num_rows)):
            rows.append({col: table[col][i].as_py() for col in table.column_names})
        return rows
    if suffix in (".jsonl", ".json"):
        with data_path.open() as f:
            for i, line in enumerate(f):
                if i >= n_prompts:
                    break
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    raise ValueError(f"unsupported dataset format: {data_path}")


def _row_to_messages(row: dict) -> list[dict]:
    if "messages" in row:
        return row["messages"]
    prompt = row.get("prompt")
    if isinstance(prompt, list):
        return prompt
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    question = row.get("question")
    if isinstance(question, str):
        return [{"role": "user", "content": question}]
    raise KeyError(
        f"row must contain messages/prompt/question, got keys={list(row.keys())}"
    )


def _tokenize_prompt(
    tokenizer,
    messages: list[dict],
    *,
    apply_chat_template_kwargs: dict | None = None,
) -> tuple[list[int], str]:
    kwargs = apply_chat_template_kwargs or {}
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    return prompt_ids, prompt_text


def _response_variants(
    tokenizer,
    answer_text: str,
    *,
    n_per_prompt: int,
    response_len: int,
    group_seed: int,
) -> list[tuple[list[int], float]]:
    answer_ids = tokenizer(str(answer_text), add_special_tokens=False)["input_ids"]
    if not answer_ids:
        answer_ids = [tokenizer.eos_token_id or 0]

    variants: list[tuple[list[int], float]] = []
    vocab = max(getattr(tokenizer, "vocab_size", 128000) - 16, 256)
    for s in range(n_per_prompt):
        rank = float(s) / max(n_per_prompt - 1, 1)
        if s == 0:
            resp_ids = [
                (group_seed * 997 + s * 131 + j) % vocab for j in range(response_len)
            ]
            reward = 0.0
        else:
            take = max(1, len(answer_ids) * s // n_per_prompt)
            resp_ids = list(answer_ids[:take])
            if len(resp_ids) < response_len:
                resp_ids = resp_ids + [answer_ids[-1]] * (response_len - len(resp_ids))
            else:
                resp_ids = resp_ids[:response_len]
            reward = rank
        variants.append((resp_ids, reward))
    return variants


def _estimate_rollout_log_probs(
    resp_ids: list[int], *, sample_index: int, group_index: int
) -> list[float]:
    base = 2.2 + 0.06 * sample_index + 0.01 * (group_index % 8)
    return [-(base + 0.012 * (tid % 977) + 0.004 * j) for j, tid in enumerate(resp_ids)]


def make_realistic_samples(
    *,
    model_path: str,
    data_path: str,
    n_prompts: int = 32,
    n_per_prompt: int = 8,
    response_len: int = 64,
    apply_chat_template_kwargs: dict | None = None,
) -> list[RolloutSampleDict]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"[rollout] tokenizer loaded from {model_path}")

    rows = _read_rows(Path(data_path), n_prompts=n_prompts)
    if len(rows) < n_prompts:
        raise RuntimeError(
            f"need {n_prompts} prompts, found {len(rows)} in {data_path}"
        )

    samples: list[RolloutSampleDict] = []
    for g, row in enumerate(rows):
        messages = _row_to_messages(row)
        label = row.get("label") or row.get("answer") or ""
        prompt_ids, prompt_text = _tokenize_prompt(
            tokenizer, messages, apply_chat_template_kwargs=apply_chat_template_kwargs
        )
        for s, (resp_ids, reward) in enumerate(
            _response_variants(
                tokenizer,
                label,
                n_per_prompt=n_per_prompt,
                response_len=response_len,
                group_seed=g,
            )
        ):
            samples.append(
                RolloutSampleDict(
                    group_index=g,
                    index=s,
                    prompt=prompt_text,
                    tokens=prompt_ids + resp_ids,
                    response_length=len(resp_ids),
                    reward=reward,
                    rollout_log_probs=_estimate_rollout_log_probs(
                        resp_ids, sample_index=s, group_index=g
                    ),
                    label=str(label),
                )
            )
        if (g + 1) % 8 == 0 or g + 1 == len(rows):
            print(
                f"[rollout] built {g + 1}/{len(rows)} prompt groups ({len(samples)} samples)"
            )
    return samples


def save_rollout_data(
    samples: list[RolloutSampleDict], out_path: str | Path, *, rollout_id: int = 1
) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "rollout_id": rollout_id,
            "metadata": {},
            "samples": [s.to_dict() for s in samples],
        },
        out,
    )
    print(f"[rollout] saved {len(samples)} samples → {out}")


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
) -> str:
    """Generate fake_rollout.pt for debug-train-only GRPO finetune."""
    out = output_path or _default_output()
    if Path(out).is_file():
        print(f"[rollout] already present — skipping: {out}")
        return out

    model_dir = model_dir or os.environ.get("MODEL_DIR", "/root/models")
    model_name = model_name or os.environ.get(
        "MODEL_NAME", "DeepSeek-V4-Flash-FP8-4layer"
    )
    data_dir = data_dir or os.environ.get("DATA_DIR", "/root/datasets")

    from dsv4_datasets import ensure_gsm8k  # noqa: WPS433

    n_prompts = int(os.environ.get("ROLLOUT_N_PROMPTS", "32"))
    n_per_prompt = int(os.environ.get("ROLLOUT_N_PER_PROMPT", "8"))
    response_len = int(os.environ.get("ROLLOUT_RESPONSE_LEN", "64"))
    bf16_path = Path(model_dir) / f"{model_name}-bf16"
    use_legacy = os.environ.get("SMOKE_LEGACY_FAKE_ROLLOUT", "0") == "1"

    if use_legacy or not bf16_path.is_dir():
        if not bf16_path.is_dir():
            print(
                f"[rollout] BF16 checkpoint missing ({bf16_path}) — using legacy random rollout"
            )
        samples = make_samples(
            n_prompts=n_prompts,
            n_per_prompt=n_per_prompt,
            response_len=response_len,
        )
    else:
        data_path = ensure_gsm8k(data_dir)
        print(f"[rollout] generating realistic rollout from {data_path}")
        samples = make_realistic_samples(
            model_path=str(bf16_path),
            data_path=str(data_path),
            n_prompts=n_prompts,
            n_per_prompt=n_per_prompt,
            response_len=response_len,
        )

    save_rollout_data(samples, out)
    return out


if __name__ == "__main__":
    prepare()
