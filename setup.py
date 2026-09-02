from __future__ import annotations

from importlib import import_module
import runpy
import subprocess
from pathlib import Path

from setuptools import Command, setup
from setuptools.command.build_py import build_py as _build_py

_editable_wheel = import_module("setuptools.command.editable_wheel").editable_wheel
_ATOMESH_BUILT = False
_PROTOS_BUILT = False


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
    print(f"Building atomesh from {mesh_dir}...", flush=True)
    subprocess.run(
        ["cargo", "build", "--release"],
        cwd=mesh_dir,
        check=True,
        text=True,
    )
    _ATOMESH_BUILT = True


def build_protos() -> None:
    global _PROTOS_BUILT

    if _PROTOS_BUILT:
        return
    import grpc_tools
    from grpc_tools import protoc

    root = Path(__file__).resolve().parent
    proto_root = root / "atom" / "proto"
    source_root = proto_root
    output_root = proto_root
    protos = sorted(source_root.rglob("*.proto"))
    if not protos:
        return

    output_root.mkdir(parents=True, exist_ok=True)
    result = protoc.main(
        [
            "grpc_tools.protoc",
            f"--proto_path={proto_root}",
            f"--proto_path={Path(grpc_tools.__file__).parent / '_proto'}",
            f"--python_out={output_root}",
            *map(str, protos),
        ]
    )
    if result:
        raise RuntimeError(f"protobuf code generation failed with exit code {result}")
    for generated in output_root.rglob("*_pb2.py"):
        generated.replace(generated.with_name(generated.name.replace("_pb2.py", "_proto.py")))
    _PROTOS_BUILT = True


class build_proto(Command):
    description = "generate Python protobuf modules"
    user_options: list[tuple[str, str, str]] = []

    def initialize_options(self) -> None:
        pass

    def finalize_options(self) -> None:
        pass

    def run(self) -> None:
        build_protos()


class install_atomesh(_build_py):
    def run(self) -> None:
        build_protos()
        build_atomesh()
        super().run()


class editable_install_atomesh(_editable_wheel):
    def run(self) -> None:
        build_protos()
        build_atomesh()
        super().run()


setup(
    use_scm_version=True,
    cmdclass={
        "build_py": install_atomesh,
        "build_proto": build_proto,
        "editable_wheel": editable_install_atomesh,
    },
)
