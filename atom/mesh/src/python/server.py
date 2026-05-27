"""Python entrypoint for running ATOM Mesh with a Python-owned engine.

This module intentionally mirrors the engine/tokenizer initialization used by
``atom.entrypoints.openai.api_server``. The Rust side receives the already
constructed Python objects and uses them in ``AtomStandaloneRouter``.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
from pathlib import Path
import sys
from typing import Any

logger = logging.getLogger("atom")

engine: Any | None = None
tokenizer: Any | None = None


def ensure_atom_source_on_path() -> None:
    atom_source_root = Path(__file__).resolve().parents[4]
    atom_source_root_str = str(atom_source_root)
    if atom_source_root_str in sys.path:
        sys.path.remove(atom_source_root_str)
    sys.path.insert(0, atom_source_root_str)

    loaded_atom = sys.modules.get("atom")
    if loaded_atom is not None and not module_loaded_from(
        loaded_atom, atom_source_root
    ):
        for module_name in list(sys.modules):
            if module_name == "atom" or module_name.startswith("atom."):
                del sys.modules[module_name]
    importlib.invalidate_caches()


def module_loaded_from(module: Any, root: Path) -> bool:
    paths: list[str] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        paths.append(module_file)
    module_path = getattr(module, "__path__", None)
    if module_path:
        paths.extend(str(path) for path in module_path)
    spec = getattr(module, "__spec__", None)
    locations = getattr(spec, "submodule_search_locations", None)
    if locations:
        paths.extend(str(path) for path in locations)
    if not paths:
        return False
    return any(is_relative_to(Path(path).resolve(), root) for path in paths)


def is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def import_atomesh_runner() -> Any:
    # Provided by the Rust PyO3 module in atom/mesh/src/python/mod.rs.
    try:
        import atomesh_runner

        return atomesh_runner
    except ModuleNotFoundError as exc:
        if exc.name != "atomesh_runner":
            raise ModuleNotFoundError(f"Module named 'atomesh_runner' not found: {exc}")

    mesh_root = Path(__file__).resolve().parents[2]
    candidates = [
        mesh_root / "target" / "debug" / "libmesh.so",
        mesh_root / "target" / "release" / "libmesh.so",
    ]

    for library_path in candidates:
        if not library_path.exists():
            continue
        spec = importlib.util.spec_from_file_location("atomesh_runner", library_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules["atomesh_runner"] = module
        spec.loader.exec_module(module)
        return module

    searched = "\n".join(f"  - {path}" for path in candidates)
    raise ModuleNotFoundError(
        "No module named 'atomesh_runner' and no loadable libmesh.so was found. "
        f"Searched:\n{searched}"
    )


def print_version(verbose: bool = False) -> None:
    try:
        atomesh_runner = import_atomesh_runner()
        version_fn = (
            atomesh_runner.version_verbose_string
            if verbose
            else atomesh_runner.version_string
        )
        print(version_fn())
    except Exception:
        print("ATOM Mesh Python interface")


def initialize_engine(args: argparse.Namespace) -> tuple[Any, Any]:
    from atom.model_engine.arg_utils import EngineArgs
    from atom.model_engine.llm_engine import _load_tokenizer

    global engine, tokenizer

    logger.info("Loading tokenizer from %s...", args.model)
    tokenizer = _load_tokenizer(args.model, args.trust_remote_code)

    logger.info("Initializing engine with model %s...", args.model)
    engine_args = EngineArgs.from_cli_args(args)
    engine = engine_args.create_engine(tokenizer=tokenizer)
    return engine, tokenizer


def initialize_standalone_service(args: argparse.Namespace) -> Any:
    from atom_standalone_service import AtomStandaloneService

    global engine, tokenizer
    return AtomStandaloneService(
        engine=engine,
        tokenizer=tokenizer,
        model_name=args.model,
    )


def launch_atom_standalone(atomesh_runner: Any, raw_args: list[str]) -> None:
    from atom.model_engine.arg_utils import EngineArgs

    parser = argparse.ArgumentParser(description="ATOM Mesh Python interface")
    EngineArgs.add_cli_args(parser)
    engine_args, mesh_args = parser.parse_known_args(raw_args)
    parsed_args = atomesh_runner.parse_from(mesh_args)
    cli_args = parsed_args["cli_args"]
    initialize_engine(engine_args)
    standalone_service = initialize_standalone_service(engine_args)

    print("\033[32mATOM starting...\033[0m")
    print(f"\033[32mHost: {cli_args['host']}:{cli_args['port']}\033[0m")
    atomesh_runner.launch_mesh(
        server_config=parsed_args["server_config"],
        standalone_service=standalone_service,
    )


def launch_atomesh(atomesh_runner: Any, raw_args: list[str]) -> None:
    parsed_args = atomesh_runner.parse_from(
        [arg for arg in raw_args if arg != "mesh-only"]
    )
    cli_args = parsed_args["cli_args"]
    prefill_urls = parsed_args["prefill_urls"]
    decode_urls = parsed_args["decode_urls"]

    print("\033[32mAtomesh starting...\033[0m")
    print(f"\033[32mHost: {cli_args['host']}:{cli_args['port']}\033[0m")
    mode = (
        "PD Disaggregated"
        if cli_args["pd_disaggregation"]
        else f"Regular ({cli_args['backend']})"
    )
    print(f"Mode: {mode}")
    print(f"Policy: {cli_args['policy']}")

    if cli_args["pd_disaggregation"] and prefill_urls:
        print(f"Prefill nodes: {prefill_urls}")
    if cli_args["pd_disaggregation"] and decode_urls:
        print(f"Decode nodes: {decode_urls}")

    atomesh_runner.launch_mesh(
        server_config=parsed_args["server_config"],
        standalone_service=None,
    )


def main() -> None:
    ensure_atom_source_on_path()
    raw_args = sys.argv[1:]
    for arg in raw_args:
        if arg in ("--version", "-V"):
            print_version(verbose=False)
            return
        if arg == "--version-verbose":
            print_version(verbose=True)
            return

    # `python xxx mesh-only ...` starts mesh routing;
    # other invocations default to ATOM standalone.
    use_atom_standalone = "mesh-only" not in raw_args
    # Import the mesh_python module.
    atomesh_runner = import_atomesh_runner()

    if use_atom_standalone:
        launch_atom_standalone(atomesh_runner, raw_args)
    else:
        launch_atomesh(atomesh_runner, raw_args)


if __name__ == "__main__":
    main()
