"""Python entrypoint for running Atomesh with a Python-owned engine.

This module intentionally mirrors the engine/tokenizer initialization used by
``atom.entrypoints.openai.api_server``. The Rust side receives the already
constructed Python objects and uses them in ``AtomStandaloneRouter``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import importlib.util
import json
import logging
from pathlib import Path
import sys
from typing import Any

logger = logging.getLogger("atom")

engine: Any | None = None
tokenizer: Any | None = None


@dataclass(frozen=True)
class StandaloneArgs:
    """Parsed standalone launch args split by their owning layer."""

    engine_args: argparse.Namespace
    mesh_args: list[str]
    default_chat_template_kwargs: dict[str, Any]


def import_atomesh_runner() -> Any:
    # Provided by the Rust PyO3 module in atom/mesh/src/python.rs.
    try:
        import atomesh_runner

        return atomesh_runner
    except ModuleNotFoundError as exc:
        if exc.name != "atomesh_runner":
            raise ModuleNotFoundError(f"Module named 'atomesh_runner' not found: {exc}")

    atom_source_root = Path(__file__).resolve().parents[3]
    mesh_root = atom_source_root / "atom" / "mesh"
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
        print("Atomesh Python interface")


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


def initialize_standalone_service(
    args: argparse.Namespace,
    default_chat_template_kwargs: dict[str, Any],
) -> Any:
    from atom.entrypoints.atomesh.atom_standalone_service import AtomStandaloneService

    global engine, tokenizer
    return AtomStandaloneService(
        engine=engine,
        tokenizer=tokenizer,
        model_name=args.model,
        default_chat_template_kwargs=default_chat_template_kwargs,
    )


def split_standalone_mesh_args(raw_args: list[str]) -> tuple[list[str], list[str]]:
    """Keep mesh-owned network args from being consumed by Python parsers.

    EngineArgs also defines --port for internal engine communication. In
    Atomesh standalone mode, the user-facing --port should configure the mesh
    HTTP router, matching the Rust CLI behavior. --server-port is accepted for
    compatibility with the classic OpenAI entrypoint and translated to --port.
    """
    mesh_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    mesh_parser.add_argument("--host")
    mesh_parser.add_argument("--port")
    mesh_parser.add_argument("--server-port")
    mesh_namespace, python_args = mesh_parser.parse_known_args(raw_args)

    mesh_args: list[str] = []
    if mesh_namespace.host is not None:
        mesh_args.extend(["--host", mesh_namespace.host])
    port = mesh_namespace.port or mesh_namespace.server_port
    if port is not None:
        mesh_args.extend(["--port", port])
    return python_args, mesh_args


def json_object_arg(raw_value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(
            f"--default-chat-template-kwargs must be valid JSON: {exc.msg}"
        ) from exc
    if not isinstance(parsed, dict):
        raise argparse.ArgumentTypeError(
            "--default-chat-template-kwargs must decode to a JSON object"
        )
    return parsed


def parse_standalone_args(raw_args: list[str]) -> StandaloneArgs:
    from atom.model_engine.arg_utils import EngineArgs

    parser = argparse.ArgumentParser(
        description="Atomesh Python interface",
        allow_abbrev=False,
    )
    EngineArgs.add_cli_args(parser)
    parser.add_argument(
        "--default-chat-template-kwargs",
        type=json_object_arg,
        default=None,
        help=(
            "Default kwargs for chat template rendering (JSON string). "
            "Merged with per-request chat_template_kwargs (request wins). "
        ),
    )

    python_raw_args, mesh_network_args = split_standalone_mesh_args(raw_args)
    engine_args, mesh_args = parser.parse_known_args(python_raw_args)

    return StandaloneArgs(
        engine_args=engine_args,
        mesh_args=mesh_args + mesh_network_args,
        default_chat_template_kwargs=engine_args.default_chat_template_kwargs or {},
    )


def launch_atom_standalone(atomesh_runner: Any, raw_args: list[str]) -> None:
    standalone_args = parse_standalone_args(raw_args)
    parsed_args = atomesh_runner.parse_from(standalone_args.mesh_args)
    cli_args = parsed_args["cli_args"]
    initialize_engine(standalone_args.engine_args)
    standalone_service = initialize_standalone_service(
        standalone_args.engine_args,
        standalone_args.default_chat_template_kwargs,
    )

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
