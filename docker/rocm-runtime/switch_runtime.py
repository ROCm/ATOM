#!/usr/bin/env python3
"""Select the image's HIP/HSA pair. Stop GPU applications before switching."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

FAMILIES = ("libamdhip64.so", "libhsa-runtime64.so")
SONAMES = ("libamdhip64.so.7", "libhsa-runtime64.so.1")


def libraries(directory):
    return sorted(p for p in directory.iterdir()
                  if any(p.name == f or p.name.startswith(f + ".") for f in FAMILIES))


def copy_aliases(sources, destination):
    """Materialize each inode once and hard-link aliases, including absolute symlinks."""
    inodes = {}
    for name, source in sources.items():
        stat = source.stat()
        key = (stat.st_dev, stat.st_ino)
        target = destination / name
        if key in inodes:
            os.link(inodes[key], target)
        else:
            shutil.copy2(source, target, follow_symlinks=True)
            inodes[key] = target


def select_runtime(root, rocm, mode):
    info = json.loads((root / "build-info.json").read_text())
    stock = root / "stock"
    if not info["enabled"]:
        if not info.get("restore_stock", False) or not stock.exists():
            print(json.dumps(info))
            return
        # A custom base may already contain a patched runtime. Opting out must
        # restore its saved original pair instead of leaving that patch active.
        if mode != "status":
            mode = "stock"
    lib = (rocm / "lib").resolve()
    if mode == "status":
        print(json.dumps(dict(build=info, active=(root / "active").read_text().strip())))
        return
    patched = root / "patched/lib"
    # Check both components before touching either library.
    if mode == "patched":
        for soname in SONAMES:
            if not (patched / soname).is_file():
                raise RuntimeError(f"Missing patched runtime: {soname}")
    if not stock.exists():
        if mode != "patched":
            raise RuntimeError("No stock runtime backup exists")
        for soname in SONAMES:
            if not (lib / soname).is_file():
                raise RuntimeError(f"Missing original runtime: {soname}")
        originals = {p.name: p for p in libraries(lib)}
        with tempfile.TemporaryDirectory(dir=root, prefix=".backup-") as temporary:
            backup = Path(temporary) / "stock"
            (backup / "lib").mkdir(parents=True)
            copy_aliases(originals, backup / "lib")
            hashes = {name: hashlib.sha256(p.read_bytes()).hexdigest()
                      for name, p in originals.items()}
            (backup / "sha256.json").write_text(json.dumps(hashes, indent=2))
            backup.rename(stock)
    hashes = json.loads((stock / "sha256.json").read_text())
    # Refuse to use a corrupt backup, even when selecting the patched pair.
    for name, digest in hashes.items():
        if hashlib.sha256((stock / "lib" / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Stock backup checksum mismatch: {name}")
    if mode == "stock":
        sources = {name: stock / "lib" / name for name in hashes}
    else:
        names = set(hashes) | {p.name for p in libraries(patched)}
        sources = {}
        for name in sorted(names):
            family = next(i for i, prefix in enumerate(FAMILIES) if name.startswith(prefix))
            sources[name] = patched / SONAMES[family]
    # Stage complete copies before replacing the live names. Hard links also cover
    # consumers using an absolute path to an old versioned filename (RTLD_NOLOAD).
    with tempfile.TemporaryDirectory(dir=lib, prefix=".atom-runtime-") as temporary:
        staging = Path(temporary)
        copy_aliases(sources, staging)
        for p in libraries(lib):
            if p.name not in sources:
                p.unlink()
        for p in staging.iterdir():
            os.replace(p, lib / p.name)
    (root / "active").write_text(mode + "\n")
    print(f"Selected {mode} HIP/HSA; restart GPU applications to load it.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("patched", "stock", "status"))
    parser.add_argument("--root", type=Path, default=Path("/opt/atom-rocm-runtime"))
    parser.add_argument("--rocm-path", type=Path, default=Path("/opt/rocm"))
    args = parser.parse_args()
    select_runtime(args.root, args.rocm_path, args.mode)


if __name__ == "__main__":
    main()
