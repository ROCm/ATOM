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
present is kept. ``--source modelscope`` uses ModelScope with the same IDs.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

MODELS = {
    "Qwen3-8B-Base": "Qwen/Qwen3-8B-Base",
    "Qwen3-8B": "Qwen/Qwen3-8B",
    "Qwen3-30B-A3B-Base": "Qwen/Qwen3-30B-A3B-Base",
    "Qwen3-30B-A3B": "Qwen/Qwen3-30B-A3B",
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


def present(dest: Path, dataset: bool) -> bool:
    if (dest / DONE).exists():
        return True
    if dataset:
        return bool(glob.glob(f"{dest}/**/*.parquet", recursive=True))
    return (dest / "config.json").exists() and bool(glob.glob(f"{dest}/*.safetensors"))


def snapshot(
    source: str, repo: str, dest: Path, patterns: list[str], dataset: bool
) -> None:
    if present(dest, dataset):
        print(f"[skip] {repo}: {dest} is present")
        return
    print(f"[download] {repo} -> {dest}")
    kwargs = {"repo_type": "dataset"} if dataset else {}
    if source == "modelscope":
        from modelscope.hub.snapshot_download import snapshot_download

        snapshot_download(
            repo_id=repo, local_dir=str(dest), allow_patterns=patterns, **kwargs
        )
    else:
        from huggingface_hub import snapshot_download

        snapshot_download(repo, local_dir=str(dest), allow_patterns=patterns, **kwargs)
    (dest / DONE).touch()


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
    args = p.parse_args()
    if not args.data_root:
        p.error("--data-root (or DATA_ROOT) is required")
    root = Path(args.data_root)
    for name in dict.fromkeys([TOKENIZER_MODEL, *args.models]):
        snapshot(
            args.source,
            MODELS[name],
            root / "models" / name,
            MODEL_FILES,
            dataset=False,
        )
    for name, repo in DATASETS.items():
        snapshot(args.source, repo, root / "raw" / name, DATA_FILES, dataset=True)
    filter_prompts(root)


if __name__ == "__main__":
    main()
