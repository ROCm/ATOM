from __future__ import annotations

from importlib import import_module
import runpy
import subprocess
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py

_editable_wheel = import_module("setuptools.command.editable_wheel").editable_wheel
_ATOMESH_BUILT = False
_ROOT = Path(__file__).resolve().parent


def get_build_env(name: str):
    """Read build-time envs without importing the atom package."""
    envs_path = Path(__file__).resolve().parent / "atom" / "utils" / "envs.py"
    return runpy.run_path(str(envs_path))["environment_variables"][name]()


def build_atomesh() -> None:
    global _ATOMESH_BUILT

    if not get_build_env("ATOM_MESH_BUILD"):
        return
    if _ATOMESH_BUILT:
        return

    root = Path(__file__).resolve().parent
    mesh_dir = root / "atom" / "mesh"
    command = ["cargo", "build", "--release"]
    features = get_build_env("ATOM_MESH_FEATURES")
    if features:
        command.extend(["--features", features])
    print(f"Building atomesh from {mesh_dir}...", flush=True)
    subprocess.run(
        command,
        cwd=mesh_dir,
        check=True,
        text=True,
    )
    _ATOMESH_BUILT = True


def generate_engine_protocol() -> None:
    """Compile engine_v1.proto into the binding the package imports."""
    # Run by path: the package itself is not importable yet at build time.
    script = _ROOT / "atom" / "model_engine" / "protocol" / "generate.py"
    binding = runpy.run_path(str(script))["generate"](_ROOT)
    print(f"Generated {binding.relative_to(_ROOT)}", flush=True)


class install_atomesh(_build_py):
    def run(self) -> None:
        build_atomesh()
        # Before the copy: build_py collects the package's .py files as it runs.
        generate_engine_protocol()
        super().run()


class editable_install_atomesh(_editable_wheel):
    def run(self) -> None:
        build_atomesh()
        generate_engine_protocol()
        super().run()


setup(
    use_scm_version=True,
    cmdclass={
        "build_py": install_atomesh,
        "editable_wheel": editable_install_atomesh,
    },
)
