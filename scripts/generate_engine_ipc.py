"""Regenerate the checked-in Python IPC bindings with protoc 27.2."""

import argparse
import subprocess
import tempfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protoc", default="protoc")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    version = subprocess.check_output([args.protoc, "--version"], text=True).strip()
    if version != "libprotoc 27.2":
        parser.error(f"expected libprotoc 27.2, found {version}")
    root = Path(__file__).resolve().parents[1]
    proto_dir = Path("atom/model_engine/ipc/proto")
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) if args.check else root
        subprocess.run(
            [args.protoc, f"-I{root}", f"--python_out={output}"]
            + [
                str(proto_dir / f"{name}.proto")
                for name in ("common", "request", "control", "result")
            ],
            cwd=root,
            check=True,
        )
        if args.check:
            for name in ("common", "request", "control", "result"):
                relative = proto_dir / f"{name}_pb2.py"
                if (root / relative).read_bytes() != (output / relative).read_bytes():
                    parser.error(f"generated binding is stale: {relative}")


if __name__ == "__main__":
    main()
