"""Python entrypoint for running Atomesh with a Python-owned engine.

This module intentionally mirrors the engine/tokenizer initialization used by
``atom.entrypoints.openai.api_server``. The Rust side serves the OpenAI API
(``AtomBridgeRouter``): it renders, tokenizes, detokenizes and parses, and
hands token ids to this engine through ``TokenBridge``
(docs/atomesh_rust_frontend.md).
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


def initialize_engine(args: argparse.Namespace) -> tuple[Any, Any]:
    from atom.model_engine.arg_utils import EngineArgs
    from atom.model_engine.llm_engine import load_tokenizer

    global engine, tokenizer

    logger.info("Loading tokenizer from %s...", args.model)
    tokenizer = load_tokenizer(args.model, args.trust_remote_code)

    logger.info("Initializing engine with model %s...", args.model)
    engine_args = EngineArgs.from_cli_args(args)
    engine = engine_args.create_engine(tokenizer=tokenizer)
    return engine, tokenizer


def initialize_token_bridge(atomesh_runner: Any, args: argparse.Namespace) -> Any:
    from atom.entrypoints.atomesh.token_bridge import TokenBridge

    return TokenBridge(
        engine=engine,
        sink=atomesh_runner.StepSink(),
        tokenizer=tokenizer,
        model=args.model,
        tool_call_parser=getattr(args, "tool_call_parser", None),
    )


def served_model_name(args: argparse.Namespace) -> str:
    return getattr(args, "served_model_name", None) or args.model


def rust_frontend_mesh_args(
    args: argparse.Namespace, mesh_args: list[str]
) -> list[str]:
    """The Rust frontend tokenizes, so the router loads the model's tokenizer
    unless it was told where one is."""
    if any(
        arg.split("=", 1)[0] in ("--model-path", "--tokenizer-path")
        for arg in mesh_args
    ):
        return mesh_args
    return [*mesh_args, "--model-path", args.model]


# ATOM's tool-call formats by the mesh router's name for the same syntax.
# ATOM's `qwen` is the XML format (Qwen3.5, Qwen3-Coder), which the router
# calls `qwen_xml`; its `qwen` is Qwen3's JSON-in-tag. GLM is left to the
# router's model-name mapping: its 4.5 and 4.7 dialects are one ATOM parser.
MESH_TOOL_CALL_PARSERS = {
    "dsml": "dsml",
    "dsml_v41": "dsml_v41",
    "kimi": "kimik2",
    "kimi_k3": "kimi_k3",
    "minimax": "minimax",
    "qwen": "qwen_xml",
}


def _mesh_reasoning_parser(dialect: Any) -> str | None:
    """The router's parser for ATOM's reasoning dialect, or None to leave it
    to the router. The `<think>` one starts outside reasoning: whether a
    request's prompt opened the block is decided per request."""
    from atom.entrypoints.openai import kimi_k3_tokens
    from atom.entrypoints.openai.reasoning_dialects import THINK_END_MARKER

    return {
        kimi_k3_tokens.THINK_END: "kimi_k3",
        THINK_END_MARKER: "qwen3",
    }.get(dialect.think_end_marker)


def rust_frontend_parser_args(
    args: argparse.Namespace, mesh_args: list[str]
) -> list[str]:
    """`--tool-call-parser` / `--reasoning-parser` for the router, read off the
    model the way ATOM's own server reads them, unless the launch named one.

    The router's own choice goes by the served name, which misses a model
    served under a path or a name of its own."""
    from atom.entrypoints.openai.chat_encoders import (
        chat_template_source,
        load_custom_message_encoder,
        render_probe_prompt,
    )
    from atom.entrypoints.openai.reasoning_dialects import resolve_dialect
    from atom.entrypoints.openai.tool_parser.registry import (
        resolve_tool_call_parser,
    )

    named = {arg.split("=", 1)[0] for arg in mesh_args}
    encoder = load_custom_message_encoder(args.model)
    parser_args = []
    if "--tool-call-parser" not in named:
        parser = resolve_tool_call_parser(None, tokenizer, encoder, model=args.model)
        name = MESH_TOOL_CALL_PARSERS.get(parser.NAME) if parser else None
        if name:
            parser_args += ["--tool-call-parser", name]
    if "--reasoning-parser" not in named:
        dialect, _ = resolve_dialect(
            chat_template_source(tokenizer, encoder),
            render_probe_prompt(tokenizer, encoder, tools=False) or "",
        )
        name = _mesh_reasoning_parser(dialect)
        if name:
            parser_args += ["--reasoning-parser", name]
    logger.info("Rust frontend parsers: %s", parser_args or "the router's own")
    return parser_args


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
    """Give the mesh router back the flag the Python parser just consumed.

    `--tool-call-parser` has two consumers with two vocabularies: the Rust
    router declares its own (`cliargs.rs`, sglang spelling) and reads it in
    `app_context`, and the Python service resolves it to a `ToolCallParser`.
    Until this entrypoint registered the flag it reached the router as an
    unrecognised arg and was passed straight through; registering it made
    `parse_known_args` swallow it, so the router silently lost a setting that
    had been reaching it. Both need it, so both get it.
    """
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

    # Not validated against ATOM's vocabulary here, and that is the point.
    # This flag has two consumers with disjoint vocabularies: the Rust router
    # declares its own (`json`, `python`, `xml`, `hermes`) and ATOM's
    # resolver takes `dsml`, `glm`, `kimi`, `kimi_k3`, `minimax`, `qwen`.
    # Raising on a name ATOM does not know would kill any existing standalone
    # deployment launched with a router name -- before this entrypoint
    # registered the flag at all, those passed straight through.
    #
    # So: a name ATOM knows binds ATOM's parser, anything else is the
    # router's and ATOM falls back to reading the chat template. Either way
    # the value reaches the router, and the choice is logged.
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

    # The router parses the output, so an ATOM format name becomes its own.
    forwarded = MESH_TOOL_CALL_PARSERS.get(forwarded, forwarded)
    mesh_args = mesh_args + mesh_network_args + _forward_tool_call_parser(forwarded)
    return StandaloneArgs(
        engine_args=engine_args,
        mesh_args=rust_frontend_mesh_args(engine_args, mesh_args),
        default_chat_template_kwargs=engine_args.default_chat_template_kwargs or {},
    )


def launch_atom_standalone(atomesh_runner: Any, raw_args: list[str]) -> None:
    standalone_args = parse_standalone_args(raw_args)
    parsed_args = atomesh_runner.parse_from(standalone_args.mesh_args)
    cli_args = parsed_args["cli_args"]
    initialize_engine(standalone_args.engine_args)
    parser_args = rust_frontend_parser_args(
        standalone_args.engine_args, standalone_args.mesh_args
    )
    if parser_args:
        parsed_args = atomesh_runner.parse_from(standalone_args.mesh_args + parser_args)
    token_bridge = initialize_token_bridge(atomesh_runner, standalone_args.engine_args)

    # This frontend is the api_server's counterpart -- same tokenizer, same
    # per-request accumulators -- but it builds its engine itself and never
    # runs that FastAPI lifespan, so it has to apply the GC policy here.
    tune_gc()
    maybe_attach_gc_debug_callback("atomesh")
    freeze_gc_heap("atomesh")

    print("\033[32mATOM starting...\033[0m")
    print(f"\033[32mHost: {cli_args['host']}:{cli_args['port']}\033[0m")
    atomesh_runner.launch_mesh(
        server_config=parsed_args["server_config"],
        token_bridge=token_bridge,
        served_model_name=served_model_name(standalone_args.engine_args),
        default_chat_template_kwargs=json.dumps(
            standalone_args.default_chat_template_kwargs
        ),
    )


def _with_atom_backend(mesh_args: list[str]) -> list[str]:
    """Route to ATOM workers unless the command line names a backend.

    The router defaults to `--backend sglang`, which keys an image chat by
    SGLang's prefix cache rather than ATOM's and discovers workers through
    SGLang's endpoints. Appended rather than prepended, so it lands inside a
    `launch` subcommand's arguments when one is given.
    """
    names_backend = any(
        arg in ("--backend", "--runtime")
        or arg.startswith(("--backend=", "--runtime="))
        for arg in mesh_args
    )
    return mesh_args if names_backend else mesh_args + ["--backend", "atom"]


def launch_atomesh(atomesh_runner: Any, raw_args: list[str]) -> None:
    parsed_args = atomesh_runner.parse_from(
        _with_atom_backend([arg for arg in raw_args if arg != "mesh-only"])
    )
    cli_args = parsed_args["cli_args"]
    prefill_urls = parsed_args["prefill_urls"]
    decode_urls = parsed_args["decode_urls"]
    encode_urls = parsed_args["encode_urls"]

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
    if cli_args["pd_disaggregation"] and encode_urls:
        print(f"Encode nodes: {encode_urls}")

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
