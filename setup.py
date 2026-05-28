from __future__ import annotations

import runpy
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


def get_build_env(name: str):
    """Read build-time envs without importing the atom package."""
    envs_path = Path(__file__).resolve().parent / "atom" / "utils" / "envs.py"
    return runpy.run_path(str(envs_path))["environment_variables"][name]()


class install_atom_mesh(_build_py):
    def run(self) -> None:
        if get_build_env("ATOM_MESH_BUILD"):
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
