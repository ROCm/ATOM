# SPDX-License-Identifier: MIT
"""Strict JSON and bounded, relocatable artifact paths."""

import gzip
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

from . import AGGREGATION_VERSION


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f, parse_constant=_invalid_constant)


def _invalid_constant(value):
    raise ValueError(f"Non-finite JSON number: {value}")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
            f.write("\n")
            os.fchmod(f.fileno(), 0o644)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def write_view(path, value, summary):
    write_json(
        path,
        {
            **value,
            "derivation": {
                "requests_sha256": summary["requests_sha256"],
                "aggregation_version": AGGREGATION_VERSION,
            },
        },
    )


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError(f"Invalid artifact path: {relative!r}")
    root = Path(root).resolve()
    candidate = root / relative
    if ".." in Path(relative).parts or not candidate.resolve().is_relative_to(root):
        raise ValueError(f"Artifact path escapes bundle: {relative!r}")
    # Even internal symlinks make the immutable file manifest ambiguous.
    if any(p.is_symlink() for p in [candidate, *candidate.parents] if p != root):
        raise ValueError(f"Symlink in artifact path: {relative!r}")
    return candidate


def iter_jsonl(path):
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=_invalid_constant)
                if not isinstance(row, dict):
                    raise ValueError(  # noqa: TRY004 - invalid file format
                        "record must be an object"
                    )
                yield row
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path.name}:{line_number}: {exc}") from exc


class JsonlWriter:
    """Bounded buffering; flush during writes once per second, and on close."""

    def __init__(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = open(  # noqa: SIM115 - closed by recorder lifecycle
            path, "w", encoding="utf-8", buffering=64 * 1024
        )
        self.next_flush = time.monotonic() + 1

    def write(self, row):
        self.file.write(json.dumps(row, allow_nan=False, separators=(",", ":")) + "\n")
        now = time.monotonic()
        if now >= self.next_flush:
            self.file.flush()
            self.next_flush = now + 1

    def close(self):
        self.file.close()
