"""Python bootstrap for Atomesh with Rust-owned EngineCore transport.

Python parses EngineArgs, loads model/tokenizer state and spawns EngineCore
processes. Rust owns HTTP preparation plus the EngineCore input, control and
output sockets after startup handoff.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atom.entrypoints.openai.tool_parser.registry import (
    TOOL_CALL_PARSER_HELP,
    validate_tool_call_parser,
)
from atom.utils.gc_utils import (
    freeze_gc_heap,
    maybe_attach_gc_debug_callback,
    tune_gc,
)

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
    except Exception:  # noqa: BLE001 - a version banner must not fail the CLI
        print("Atomesh Python interface")


def initialize_engine(
    args: argparse.Namespace,
    external_transport_factory: Any | None = None,
) -> tuple[Any, Any]:
    from atom.model_engine.arg_utils import EngineArgs

    global engine, tokenizer

    logger.info("Initializing engine with model %s...", args.model)
    engine_args = EngineArgs.from_cli_args(args)
    engine = engine_args.create_engine(
        external_transport_factory=external_transport_factory,
    )
    tokenizer = engine.tokenizer
    return engine, tokenizer


def initialize_standalone_service(
    args: argparse.Namespace,
    default_chat_template_kwargs: dict[str, Any],
) -> Any:
    from atom.entrypoints.atomesh.atom_standalone_service import AtomStandaloneService

    return AtomStandaloneService(
        engine=engine,
        tokenizer=tokenizer,
        model_name=args.model,
        default_chat_template_kwargs=default_chat_template_kwargs,
        tool_call_parser=getattr(args, "tool_call_parser", None),
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


def _is_atom_tool_call_parser(value: str | None) -> bool:
    """Whether this name is one ATOM's own resolver understands.

    A question, not an assertion: the mesh router answers to a different set
    of names for the same flag, and neither side owns the other's.
    """
    if value is None:
        return False
    try:
        validate_tool_call_parser(value)
    except ValueError:
        return False
    return True


def _forward_tool_call_parser(value: str | None) -> list[str]:
    """Forward a Rust-supported parser name consumed by the Python CLI layer."""
    return [] if value is None else ["--tool-call-parser", value]


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
    from atom.utils.arg_parser import FlexibleArgumentParser

    parser = FlexibleArgumentParser(
        description="Atomesh Python interface",
        allow_abbrev=False,
    )
    EngineArgs.add_cli_args(parser)
    parser.add_argument(
        "--tool-call-parser",
        type=str,
        # Not "auto": that is what *unset* resolves to anyway, and telling the
        # two apart is what decides whether the mesh router is told as well.
        default=None,
        help=TOOL_CALL_PARSER_HELP,
    )
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
    forwarded = engine_args.tool_call_parser
    if forwarded is not None and _is_atom_tool_call_parser(forwarded):
        parser.error(
            f"--tool-call-parser={forwarded!r} requires the legacy Python "
            "AtomStandaloneService and is not supported by the Rust EngineCore path"
        )

    # Rust parser names are forwarded to mesh and removed from EngineArgs.
    # ATOM-only names were consumed by AtomStandaloneService, which is not on
    # the Rust EngineCore request path, so accepting them would silently change
    # tool-call behavior.
    if engine_args.tool_call_parser is not None and not _is_atom_tool_call_parser(
        engine_args.tool_call_parser
    ):
        logger.info(
            "--tool-call-parser=%r is not one of ATOM's formats, so it is "
            "forwarded to the mesh router and ATOM reads its own format from "
            "the chat template.",
            engine_args.tool_call_parser,
        )
        engine_args.tool_call_parser = None

    return StandaloneArgs(
        engine_args=engine_args,
        mesh_args=(
            mesh_args + mesh_network_args + _forward_tool_call_parser(forwarded)
        ),
        default_chat_template_kwargs=engine_args.default_chat_template_kwargs or {},
    )


def launch_atom_standalone(atomesh_runner: Any, raw_args: list[str]) -> None:
    standalone_args = parse_standalone_args(raw_args)
    mesh_args = list(standalone_args.mesh_args)
    if not any(
        arg == "--model-path" or arg.startswith("--model-path=") for arg in mesh_args
    ):
        mesh_args.extend(["--model-path", standalone_args.engine_args.model])
    parsed_args = atomesh_runner.parse_from(mesh_args)
    cli_args = parsed_args["cli_args"]
    engine_core_ipc = None
    standalone_engine = None
    try:
        standalone_engine, _ = initialize_engine(
            standalone_args.engine_args,
            external_transport_factory=atomesh_runner.bind_engine_core_ipc,
        )
        engine_core_ipc = standalone_engine.core_mgr.external_transport_owner
        if engine_core_ipc is None:
            raise RuntimeError(
                "Rust-owned EngineCore transport was not initialized by CoreManager"
            )
    except Exception:
        if standalone_engine is not None:
            standalone_engine.close()
        raise

    # This frontend is the api_server's counterpart -- same tokenizer, same
    # per-request accumulators -- but it builds its engine itself and never
    # runs that FastAPI lifespan, so it has to apply the GC policy here.
    tune_gc()
    maybe_attach_gc_debug_callback("atomesh")
    freeze_gc_heap("atomesh")

    print("\033[32mATOM starting...\033[0m")
    print(f"\033[32mHost: {cli_args['host']}:{cli_args['port']}\033[0m")
    try:
        atomesh_runner.launch_mesh(
            server_config=parsed_args["server_config"],
            engine_core_ipc=engine_core_ipc,
            default_chat_template_kwargs=(
                standalone_args.default_chat_template_kwargs
            ),
        )
    finally:
        if standalone_engine is not None:
            standalone_engine.close()


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
