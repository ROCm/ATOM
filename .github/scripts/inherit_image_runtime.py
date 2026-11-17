"""Give a writable overlay venv the image's existing Python package runtime."""

import json
import os
import site
import subprocess
import sys
from pathlib import Path

import torch


def inherit_runtime(destination: Path, receipt: Path):
    parent = {
        "python": sys.executable,
        "prefix": sys.prefix,
        "site_paths": [path for path in site.getsitepackages() if Path(path).is_dir()],
        "torch": torch.__version__,
        "torch_file": str(Path(torch.__file__).resolve()),
        "hip": torch.version.hip,
    }
    assert not destination.exists(), destination
    subprocess.run([sys.executable, "-m", "venv", str(destination)], check=True)
    python = destination / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    target_site = Path(
        subprocess.check_output(
            [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
            text=True,
        ).strip()
    )
    # --system-site-packages inherits the base interpreter, not a parent venv
    # such as /opt/venv. Process its .pth files too (FlyDSL may be editable).
    additions = "; ".join(
        "site.addsitedir(" + repr(path) + ")" for path in parent["site_paths"]
    )
    (target_site / "00_image_runtime.pth").write_text(
        "import site; " + additions + "\n"
    )
    child = json.loads(
        subprocess.check_output(
            [
                str(python),
                "-c",
                "import json, pathlib, torch; print(json.dumps({'torch': torch.__version__, 'torch_file': str(pathlib.Path(torch.__file__).resolve()), 'hip': torch.version.hip}))",
            ],
            text=True,
        )
    )
    assert child == {key: parent[key] for key in child}, (parent, child)
    receipt.write_text(json.dumps({"image": parent, "overlay": child}, indent=2) + "\n")
    return python


if __name__ == "__main__":
    inherit_runtime(Path(sys.argv[1]), Path(sys.argv[2]))
