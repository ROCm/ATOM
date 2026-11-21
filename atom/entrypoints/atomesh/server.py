"""Python entrypoint for Atomesh.

ATOM standalone: Python parses the args, launches the EngineCores and reaps
them; the Rust runtime (``atom/mesh/src/atom_engine``) is their only client
and serves HTTP. ``mesh-only``: a plain mesh router.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atom.entrypoints.atomesh.engine_spec import (
    apply_chat_template,
    build_engine_spec,
    check_standalone_checkpoint,
    check_standalone_model,
    default_reasoning_parser,
    synthetic_token_text,
    tool_call_format,
)
from atom.entrypoints.openai.tool_parser.registry import TOOL_CALL_PARSER_HELP
from atom.utils.arg_parser import FlexibleArgumentParser
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


def load_tokenizer(args: argparse.Namespace) -> Any:
    from atom.model_engine.llm_engine import load_tokenizer as load_model_tokenizer

    global tokenizer

    logger.info("Loading tokenizer from %s...", args.model)
    tokenizer = load_model_tokenizer(args.model, args.trust_remote_code)
    return tokenizer


def initialize_engine(args: argparse.Namespace) -> Any:
    from atom.model_engine.arg_utils import EngineArgs

    global engine

    logger.info("Initializing engine with model %s...", args.model)
    engine_args = EngineArgs.from_cli_args(args)
    # The Rust runtime connects to the EngineCores; CoreManager only spawns them.
    engine = engine_args.create_engine(tokenizer=tokenizer, external_engine_client=True)
    return engine


def split_standalone_mesh_args(raw_args: list[str]) -> tuple[list[str], list[str]]:
    """Keep mesh-owned network args from being consumed by Python parsers.

    EngineArgs also defines --port for internal engine communication. In
    Atomesh standalone mode, the user-facing --port should configure the mesh
    HTTP router, matching the Rust CLI behavior. --server-port is accepted for
    compatibility with the classic OpenAI entrypoint and translated to --port.
    """
    mesh_parser = FlexibleArgumentParser(add_help=False, allow_abbrev=False)
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

    parser = FlexibleArgumentParser(
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

    # Decided here against ATOM's own resolvers and handed to the router
    # decided (`launch_atom_standalone`).
    parser.add_argument(
        "--tool-call-parser", type=str, default=None, help=TOOL_CALL_PARSER_HELP
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        default=None,
        help="A chat template file, or the template itself, as the Python server takes.",
    )
    parser.add_argument(
        "--reasoning-parser",
        type=str,
        default=None,
        help="The router's reasoning parser; read off the chat template when unset.",
    )

    python_raw_args, mesh_network_args = split_standalone_mesh_args(raw_args)
    # What Python does not know -- the rest of the router's flags -- is the
    # mesh router's.
    engine_args, mesh_args = parser.parse_known_args(python_raw_args)
    return StandaloneArgs(
        engine_args=engine_args,
        mesh_args=mesh_args + mesh_network_args,
        default_chat_template_kwargs=engine_args.default_chat_template_kwargs or {},
    )


def launch_atom_standalone(raw_args: list[str]) -> None:
    standalone_args = parse_standalone_args(raw_args)
    engine_args = standalone_args.engine_args
    # Refused before the Rust runtime is even imported, as before any weights.
    check_standalone_model(engine_args)
    atomesh_runner = import_atomesh_runner()
    tokenizer = load_tokenizer(engine_args)
    check_standalone_checkpoint(engine_args.model)
    # The template, reasoning parser and tool-call format are decided here,
    # against ATOM's own resolvers, and handed to the router decided.
    mesh_args = list(standalone_args.mesh_args)
    with tempfile.TemporaryDirectory(prefix="atomesh-") as workdir:
        chat_template = apply_chat_template(
            tokenizer, engine_args.chat_template, workdir
        )
        reasoning_parser = engine_args.reasoning_parser or default_reasoning_parser(
            tokenizer
        )
        mesh_args += ["--reasoning-parser", reasoning_parser]
        tool_format = tool_call_format(
            engine_args.tool_call_parser,
            tokenizer,
            engine_args.model,
            set(atomesh_runner.tool_parser_names()),
        )
        if tool_format.mesh_name is not None:
            mesh_args += ["--tool-call-parser", tool_format.mesh_name]
        serve_standalone(
            atomesh_runner,
            standalone_args,
            tokenizer,
            mesh_args,
            tool_format.parser,
            chat_template,
        )


def serve_standalone(
    atomesh_runner: Any,
    standalone_args: StandaloneArgs,
    tokenizer: Any,
    mesh_args: list[str],
    tool_parser: Any,
    chat_template: str | None,
) -> None:
    engine_args = standalone_args.engine_args
    parsed_args = atomesh_runner.parse_from(mesh_args)
    cli_args = parsed_args["cli_args"]
    initialize_engine(engine_args)
    try:
        engine_spec = build_engine_spec(
            engine,
            tokenizer,
            engine_args.served_model_name or engine_args.model,
            standalone_args.default_chat_template_kwargs,
            tool_parser,
            chat_template,
            synthetic_text=synthetic_token_text(engine_args),
        )
        # This process keeps the engine and reaps its Cores while Rust serves.
        tune_gc()
        maybe_attach_gc_debug_callback("atomesh")
        freeze_gc_heap("atomesh")
        print("\033[32mATOM starting...\033[0m")
        print(f"\033[32mHost: {cli_args['host']}:{cli_args['port']}\033[0m")
        atomesh_runner.launch_mesh(
            server_config=parsed_args["server_config"], engine_spec=engine_spec
        )
    finally:
        # Rust has stopped the EngineCores by now, or died trying; reap them.
        engine.close()


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

    atomesh_runner.launch_mesh(server_config=parsed_args["server_config"])


def main() -> None:
    raw_args = sys.argv[1:]
    for arg in raw_args:
        if arg in ("--version", "-V"):
            print_version(verbose=False)
            return
        if arg == "--version-verbose":
            print_version(verbose=True)
            return
    # `mesh-only` starts a plain mesh router; anything else, ATOM standalone.
    if "mesh-only" in raw_args:
        launch_atomesh(import_atomesh_runner(), raw_args)
    else:
        launch_atom_standalone(raw_args)


if __name__ == "__main__":
    main()
