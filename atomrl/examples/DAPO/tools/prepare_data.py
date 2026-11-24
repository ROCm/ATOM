# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Download the models and datasets of the DAPO and GRPO examples and filter the prompts.

Runs inside the example image, which has huggingface_hub, datasets and transformers:

    python3 examples/DAPO/tools/prepare_data.py --data-root "$DATA_ROOT" \
        --models Qwen3-8B-Base Qwen3-30B-A3B-Base

Produces, under the data root:

    models/<name>/                                                  one per --models entry
    data_cached/qwen3-8b-maxprompt1024/dapo-math-17k.filtered.parquet   train
    data_cached/qwen3-8b-maxprompt1024/aime-2024.filtered.parquet       validation

Prompts longer than 1024 tokens are dropped; without that the trainer starts with a
slow scan for overlong prompts. Qwen3-8B-Base and the Qwen3-30B-A3B models share a
byte-identical tokenizer, so one filtered set serves every example. Anything already
present at the configured immutable revision is kept; legacy or mismatched model
directories are refreshed. ``--source modelscope`` uses ModelScope with the same IDs
and source-specific immutable revisions.
"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    hf_revision: str
    modelscope_revision: str

    def revision(self, source: str) -> str:
        return self.hf_revision if source == "hf" else self.modelscope_revision


MODELS = {
    "Qwen3-8B-Base": ModelSpec(
        "Qwen/Qwen3-8B-Base",
        "49e3418fbbbca6ecbdf9608b4d22e5a407081db4",
        "932bc907a0f908fd665867dec24af47c2f57e719",
    ),
    "Qwen3-8B": ModelSpec(
        "Qwen/Qwen3-8B",
        "b968826d9c46dd6066d109eabc6255188de91218",
        "26028140be3ee69b82b1d1450179ab71bb1121b9",
    ),
    "Qwen3-30B-A3B-Base": ModelSpec(
        "Qwen/Qwen3-30B-A3B-Base",
        "1b75feb79f60b8dc6c5bc769a898c206a1c6a4f9",
        "b4a03a639a3d9d5ece1ff5f6a5a4678669e70e05",
    ),
    "Qwen3-30B-A3B": ModelSpec(
        "Qwen/Qwen3-30B-A3B",
        "ad44e777bcd18fa416d9da3bd8f70d33ebb85d39",
        "8143878278a202cc0ff9ef1bbe00b2abb1adf86d",
    ),
}
DATASETS = {
    "DAPO-Math-17k": "BytedTsinghua-SIA/DAPO-Math-17k",
    "AIME-2024": "BytedTsinghua-SIA/AIME-2024",
}
MODEL_FILES = [
    "*.json",
    "*.txt",
    "*.safetensors",
    "*.model",
    "tokenizer*",
    "*.py",
    "*.tiktoken",
]
DATA_FILES = ["*.parquet", "*.json", "*.jsonl", "*.md", "*.txt"]
TOKENIZER_MODEL = "Qwen3-8B-Base"
MAX_PROMPT_LENGTH = 1024
PROMPT_KEY = "prompt"
FILTERED = "data_cached/qwen3-8b-maxprompt1024"
DONE = ".download_complete"


def present(dest: Path, dataset: bool, expected_marker: str | None = None) -> bool:
    marker = dest / DONE
    if dataset and marker.exists():
        return True
    if dataset:
        return bool(glob.glob(f"{dest}/**/*.parquet", recursive=True))
    complete = (dest / "config.json").exists() and bool(
        glob.glob(f"{dest}/*.safetensors")
    )
    return (
        complete
        and marker.exists()
        and marker.read_text(encoding="utf-8").strip() == expected_marker
    )


def snapshot(
    source: str,
    repo: str,
    dest: Path,
    patterns: list[str],
    dataset: bool,
    revision: str | None = None,
) -> None:
    expected_marker = f"{source}@{revision}" if revision else None
    if present(dest, dataset, expected_marker):
        suffix = f" at {revision}" if revision else ""
        print(f"[skip] {repo}{suffix}: {dest} is present")
        return
    suffix = f" at {revision}" if revision else ""
    print(f"[download] {repo}{suffix} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    kwargs = {"repo_type": "dataset"} if dataset else {}
    if revision:
        kwargs["revision"] = revision
    if source == "modelscope":
        from modelscope.hub.snapshot_download import snapshot_download

        snapshot_download(
            repo_id=repo, local_dir=str(dest), allow_patterns=patterns, **kwargs
        )
    else:
        from huggingface_hub import snapshot_download

        snapshot_download(repo, local_dir=str(dest), allow_patterns=patterns, **kwargs)
    (dest / DONE).write_text(expected_marker or "complete", encoding="utf-8")


def first_parquet(pattern: str) -> str:
    hits = sorted(glob.glob(pattern, recursive=True))
    if not hits:
        raise FileNotFoundError(f"no parquet under {pattern}")
    return hits[0]


def filter_prompts(root: Path) -> None:
    import datasets
    from transformers import AutoTokenizer

    out_dir = root / FILTERED
    jobs = [
        (root / "raw/DAPO-Math-17k", out_dir / "dapo-math-17k.filtered.parquet"),
        (root / "raw/AIME-2024", out_dir / "aime-2024.filtered.parquet"),
    ]
    if all(dst.exists() for _, dst in jobs):
        print(f"[skip] {out_dir} is complete")
        return
    tokenizer = AutoTokenizer.from_pretrained(root / "models" / TOKENIZER_MODEL)

    def prompt_len(doc) -> int:
        # transformers 4.x returns a flat list of ids; 5.x returns a BatchEncoding,
        # whose len() counts its keys instead of the tokens.
        enc = tokenizer.apply_chat_template(
            doc[PROMPT_KEY], add_generation_prompt=True, tokenize=True
        )
        ids = enc["input_ids"] if hasattr(enc, "keys") else enc
        if ids and isinstance(ids[0], (list, tuple)):
            ids = ids[0]
        return len(ids)

    out_dir.mkdir(parents=True, exist_ok=True)
    nproc = max(1, min(64, (os.cpu_count() or 8) // 4))
    for src_dir, dst in jobs:
        if dst.exists():
            continue
        src = first_parquet(f"{src_dir}/**/*.parquet")
        ds = datasets.Dataset.from_parquet(src)
        before = len(ds)
        ds = ds.filter(lambda d: prompt_len(d) <= MAX_PROMPT_LENGTH, num_proc=nproc)
        tmp = dst.with_suffix(".tmp.parquet")
        ds.to_parquet(str(tmp))
        tmp.rename(dst)
        print(f"[filter] {src} -> {dst}: {before} -> {len(ds)}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    p.add_argument(
        "--models", nargs="*", default=[TOKENIZER_MODEL], choices=sorted(MODELS)
    )
    p.add_argument("--source", choices=("hf", "modelscope"), default="hf")
    p.add_argument(
        "--models-only",
        action="store_true",
        help="prepare the selected pinned models without datasets or prompt filtering",
    )
    args = p.parse_args()
    if not args.data_root:
        p.error("--data-root (or DATA_ROOT) is required")
    root = Path(args.data_root)
    for name in dict.fromkeys([TOKENIZER_MODEL, *args.models]):
        model = MODELS[name]
        snapshot(
            args.source,
            model.repo,
            root / "models" / name,
            MODEL_FILES,
            dataset=False,
            revision=model.revision(args.source),
        )
    if args.models_only:
        return
    for name, repo in DATASETS.items():
        snapshot(args.source, repo, root / "raw" / name, DATA_FILES, dataset=True)
    filter_prompts(root)


if __name__ == "__main__":
    main()
