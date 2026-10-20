#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Prepare the pinned GQA draft on shared NFS with the Hugging Face Python SDK."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path

REPO = "Inferact/MiniMax-M3-EAGLE3-GQA"
REVISION = "96692486b5fd38ebf8fd2a5f6bb53427d30819a8"
RECEIPT = ".atom-gqa-ready.json"


def validate_config(directory):
    cfg = json.loads((directory / "config.json").read_text())
    expected = {
        "num_attention_heads": 64,
        "num_key_value_heads": 4,
        "num_hidden_layers": 1,
        "head_dim": 128,
    }
    if cfg.get("architectures") != ["LlamaForCausalLMEagle3"]:
        raise ValueError("Unexpected GQA draft architecture")
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f"GQA draft {key}: expected {value}, got {cfg.get(key)}")


def file_hash(path, lfs):
    digest = hashlib.sha256() if lfs else hashlib.sha1()
    if not lfs:
        digest.update(f"blob {path.stat().st_size}\0".encode())
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_ready(directory):
    receipt = json.loads((directory / RECEIPT).read_text())
    if receipt["repo"] != REPO or receipt["revision"] != REVISION:
        raise ValueError("Draft receipt has a different model/revision")
    validate_config(directory)
    if "config.json" not in receipt["files"] or not any(
        name.endswith(".safetensors") for name in receipt["files"]
    ):
        raise ValueError("Draft receipt is missing config or weights")
    for name, entry in receipt["files"].items():
        stat = (directory / name).stat()
        if stat.st_size != entry["size"] or stat.st_mtime_ns != entry["mtime_ns"]:
            raise ValueError(f"Draft file changed or is incomplete: {name}")
        if file_hash(directory / name, entry["lfs"]) != entry["hash"]:
            raise ValueError(f"Draft content hash mismatch: {name}")
    return receipt


def prepare(directory, api=None, download=None):
    directory.parent.mkdir(parents=True, exist_ok=True)
    # Two CI submissions may prepare the same directory simultaneously.
    with (directory.parent / f".{directory.name}.prepare.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            return check_ready(directory)
        except (OSError, ValueError, KeyError):
            pass
        if api is None or download is None:
            from huggingface_hub import HfApi, snapshot_download

            api, download = HfApi(), snapshot_download
        info = api.model_info(REPO, revision=REVISION, files_metadata=True)
        if info.sha != REVISION:
            raise ValueError("Hugging Face returned a different revision")
        expected = {}
        for item in info.siblings:
            # Config and all safetensors shards/index constitute the draft;
            # tokenizer comes from the target model, not this draft repository.
            if item.rfilename == "config.json" or item.rfilename.endswith(
                (".safetensors", ".safetensors.index.json")
            ):
                expected[item.rfilename] = {
                    "size": item.size,
                    "lfs": bool(item.lfs),
                    "hash": item.lfs.sha256 if item.lfs else item.blob_id,
                }
        if "config.json" not in expected or not any(
            n.endswith(".safetensors") for n in expected
        ):
            raise ValueError("Pinned repository has no config/weights")

        def complete():
            for name, entry in expected.items():
                path = directory / name
                if (
                    not path.is_file()
                    or path.stat().st_size != entry["size"]
                    or file_hash(path, entry["lfs"]) != entry["hash"]
                ):
                    return False
            return True

        if not complete():
            print(
                f"[gqa-draft] SDK snapshot_download {REPO}@{REVISION} -> {directory}",
                flush=True,
            )
            # Force replaces incomplete/corrupt local files even if HF's local
            # cache metadata incorrectly still calls them complete.
            download(
                repo_id=REPO,
                revision=REVISION,
                local_dir=str(directory),
                allow_patterns=list(expected),
                force_download=True,
            )
        if not complete():
            raise ValueError("Draft content/size check failed after SDK download")
        validate_config(directory)
        index = directory / "model.safetensors.index.json"
        if index.exists():
            shards = set(json.loads(index.read_text())["weight_map"].values())
            if not shards <= expected.keys():
                raise ValueError("Draft weight index references missing shards")
        receipt = {
            "repo": REPO,
            "revision": REVISION,
            "files": {
                name: {**entry, "mtime_ns": (directory / name).stat().st_mtime_ns}
                for name, entry in expected.items()
            },
        }
        tmp = directory / (RECEIPT + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(receipt, indent=2) + "\n")
        tmp.replace(directory / RECEIPT)
        return check_ready(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    receipt = (
        check_ready(args.directory) if args.check_only else prepare(args.directory)
    )
    print(
        f"[gqa-draft] READY {args.directory} revision={receipt['revision']}; "
        f"bytes={sum(f['size'] for f in receipt['files'].values())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
