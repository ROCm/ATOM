#!/usr/bin/env python3
"""Resolve a draft shipped inside the target's Hugging Face repository."""

import argparse
from pathlib import Path, PurePosixPath


def resolve_draft_model(model: str, draft: str) -> str:
    if not draft or Path(draft).is_dir() or Path(model).is_dir():
        return draft
    # Independent draft repositories and absolute paths remain unchanged.
    target = model.rstrip("/")
    if target.startswith("/") or len(target.split("/")) != 2:
        return draft
    prefix = target + "/"
    if not draft.startswith(prefix):
        return draft
    subfolder = draft[len(prefix) :].rstrip("/")
    if not subfolder or ".." in PurePosixPath(subfolder).parts:
        raise ValueError(f"Invalid draft subfolder: {draft}")

    from huggingface_hub import snapshot_download

    snapshot = snapshot_download(repo_id=target, allow_patterns=[subfolder + "/*"])
    resolved = Path(snapshot) / subfolder
    if not (resolved / "config.json").is_file():
        raise FileNotFoundError(f"Draft config missing after download: {resolved}")
    return str(resolved)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", required=True)
    args = parser.parse_args()
    print(resolve_draft_model(args.model, args.draft))
