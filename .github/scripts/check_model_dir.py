#!/usr/bin/env python3
"""Exit 0 if a local model directory is complete enough to serve, else print why and exit 1.

Shared by the SGLang and vLLM accuracy workflows (host-side cache root selection,
in-container local resolution and the download step) so they agree on one rule:

- config.json must exist;
- a model.safetensors.index.json, when present, must list no missing shards;
- with a completion marker (.download-complete / .atom_download_complete) that is enough;
- without one, also require an index (or a single model.safetensors) and a tokenizer file.

An interrupted download can leave config.json and some shards but no index or
tokenizer, so it fails here and falls through to the resumable download, while a
hand-copied complete model without a marker (e.g. MI308 local roots) still passes.

Usage: check_model_dir.py <model_dir>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COMPLETION_MARKERS = (".download-complete", ".atom_download_complete")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer.model",
    "tiktoken.model",
    "tokenizer_config.json",
)


def incomplete_reason(model_dir: Path) -> str | None:
    if not (model_dir / "config.json").is_file():
        return "no config.json"

    index_path = model_dir / "model.safetensors.index.json"
    if index_path.is_file():
        try:
            weight_map = json.loads(index_path.read_text()).get("weight_map", {})
        except (OSError, ValueError, AttributeError) as exc:
            return f"invalid safetensors index ({exc})"
        expected = {Path(name).name for name in weight_map.values()}
        present = {shard.name for shard in model_dir.glob("*.safetensors")}
        missing = sorted(expected - present)
        if missing:
            shown = ", ".join(missing[:10])
            more = "" if len(missing) <= 10 else f" ... and {len(missing) - 10} more"
            return f"missing safetensors shards: {shown}{more}"

    if any((model_dir / name).is_file() for name in COMPLETION_MARKERS):
        return None
    if not index_path.is_file() and not (model_dir / "model.safetensors").is_file():
        return "no download completion marker or safetensors index"
    if not any((model_dir / name).is_file() for name in TOKENIZER_FILES):
        return "no download completion marker or tokenizer files"
    return None


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.strip().splitlines()[-1], file=sys.stderr)
        return 2
    model_dir = Path(sys.argv[1])
    reason = incomplete_reason(model_dir)
    if reason is not None:
        print(f"Incomplete local model under {model_dir}: {reason}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
