"""Hash and transfer the frozen K31 inputs; never execute their callers."""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def checked_files(root, spec):
    expected = spec["files"]
    for name in expected:
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts, "unsafe member")
    paths = list(root.rglob("*"))
    require(not any(p.is_symlink() for p in paths), "seed links are forbidden")
    require(all(p.is_file() or p.is_dir() for p in paths), "special seed member")
    require(
        {p.relative_to(root).as_posix() for p in paths if p.is_file()} == set(expected),
        "seed member set differs",
    )
    result = {}
    for name, wanted in expected.items():
        path = root / name
        require(path.stat().st_size <= 2 * 1024**2, "oversize seed member")
        data = path.read_bytes()
        require(digest(data) == wanted, f"seed hash differs: {name}")
        result[name] = data
    return result


def transfer(root, spec_path, out, source_ref, actual_ref):
    require(
        re.fullmatch(r"[0-9a-f]{40}", source_ref) is not None,
        "seed ref must be an immutable SHA",
    )
    require(source_ref == actual_ref, "checkout differs from seed ref")
    spec = json.loads(spec_path.read_text())
    files = checked_files(root, spec)
    out.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        target = out / "seed" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    record = {
        "seed_ref": source_ref,
        "spec_sha256": digest(spec_path.read_bytes()),
        "files": spec["files"],
        "model_execution_authorized": False,
    }
    data = (json.dumps(record, indent=2) + "\n").encode()
    (out / "input-manifest.json").write_bytes(data)
    return digest(data)


def verify(root, spec_path, record_path, wanted):
    raw = record_path.read_bytes()
    require(digest(raw) == wanted, "handoff manifest differs")
    record, spec = json.loads(raw), json.loads(spec_path.read_text())
    require(
        record["spec_sha256"] == digest(spec_path.read_bytes()),
        "handoff specification differs",
    )
    require(
        record["files"] == spec["files"]
        and record["model_execution_authorized"] is False,
        "invalid handoff",
    )
    checked_files(root, spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("pack", "verify"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--ref")
    parser.add_argument("--actual-ref")
    parser.add_argument("--record", type=Path)
    parser.add_argument("--record-sha")
    args = parser.parse_args()
    if args.mode == "pack":
        value = transfer(args.root, args.spec, args.out, args.ref, args.actual_ref)
        if os.environ.get("GITHUB_OUTPUT"):
            with open(os.environ["GITHUB_OUTPUT"], "a") as output:
                output.write(f"manifest_sha256={value}\n")
    else:
        verify(args.root, args.spec, args.record, args.record_sha)
