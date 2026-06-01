from __future__ import annotations

from importlib import import_module
import runpy
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

_editable_wheel = import_module("setuptools.command.editable_wheel").editable_wheel
_ATOM_MESH_BUILT = False


def get_build_env(name: str):
    """Read build-time envs without importing the atom package."""
    envs_path = Path(__file__).resolve().parent / "atom" / "utils" / "envs.py"
    return runpy.run_path(str(envs_path))["environment_variables"][name]()


def build_atom_mesh() -> None:
    global _ATOM_MESH_BUILT

    if not get_build_env("ATOM_MESH_BUILD"):
        return
    if _ATOM_MESH_BUILT:
        return

    root = Path(__file__).resolve().parent
    mesh_dir = root / "atom" / "mesh"
    print(f"Building atom-mesh from {mesh_dir}...", flush=True)
    subprocess.run(
        ["cargo", "build", "--release"],
        cwd=mesh_dir,
        check=True,
        text=True,
    )
    _ATOM_MESH_BUILT = True


class install_atom_mesh(_build_py):
    def run(self) -> None:
        build_atom_mesh()
        super().run()


class editable_install_atom_mesh(_editable_wheel):
    def run(self) -> None:
        build_atom_mesh()
        super().run()


setup(
    use_scm_version=True,
    cmdclass={
        "build_py": install_atom_mesh,
        "editable_wheel": editable_install_atom_mesh,
    },
)
