"""Prepare a source-verified Mooncake overlay on the pinned serving image."""
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    vendor = Path(__file__).parent / "mooncake-c48"
    provenance = json.loads((vendor / "provenance.json").read_text())
    version = importlib.metadata.version("vllm")
    assert "81198e97b" in version, version
    package = Path(importlib.util.find_spec("vllm").origin).parent
    before = {}
    for relative, expected in provenance["expected_image_source_hashes"].items():
        before[relative] = digest(package / relative)
        assert before[relative] == expected, (relative, before[relative], expected)
    site = Path("/tmp/mooncake-c48-site")
    site.mkdir(exist_ok=True)
    target = site / "vllm"
    # Reuse every image file, including compiled extensions. Replace one Python
    # file with the already validated stats-only child compatibility fix.
    shutil.copytree(package, target, copy_function=os.symlink)
    multi = target / "distributed/kv_transfer/kv_connector/v1/multi_connector.py"
    multi.unlink()
    shutil.copyfile(vendor / "multi_connector.py", multi)
    assert digest(multi) == provenance["multi_connector_after_sha256"]
    versions = {}
    for name in ("vllm", "torch", "triton", "aiter", "mori", "mooncake-transfer-engine-rocm"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    record = {
        "python": sys.executable, "image_package": str(package),
        "overlay_package": str(target), "versions": versions,
        "verified_image_source_hashes": before,
        "multi_connector_after_sha256": digest(multi),
        "provenance": provenance,
        "compiled_extensions": [
            {"path": str(p), "resolved": str(p.resolve()), "bytes": p.stat().st_size}
            for p in target.glob("*.so")
        ],
    }
    Path(sys.argv[1]).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
