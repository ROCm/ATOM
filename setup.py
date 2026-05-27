from __future__ import annotations

import os
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


class install_atom_mesh(_build_py):
    def run(self) -> None:
        if os.environ.get("ATOM_MESH_BUILD") == "1":
            root = Path(__file__).resolve().parent
            mesh_dir = root / "atom" / "mesh"
            print(f"Building atom-mesh from {mesh_dir}...")
            subprocess.run(
                ["cargo", "build", "--release"],
                cwd=mesh_dir,
                check=True,
                text=True,
            )

        super().run()


setup(
    use_scm_version=True,
    cmdclass={"build_py": install_atom_mesh},
)
