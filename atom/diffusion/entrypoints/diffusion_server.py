# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""ATOM diffusion server.

Usage:
    python -m atom.diffusion.entrypoints.diffusion_server \\
        --model /path/to/MiniMax-H3/FL2VA --num-gpus 4 --port 30010

One process serves one replica. Model *variants* that are separate checkpoint
partitions -- MiniMax-H3 ``fl2va`` and ``ref2va`` are ~135 GiB each -- are
separate replicas on separate ports, not two branches of one load.
"""

import argparse
import contextlib
import logging
import os
import signal
import sys

from atom.diffusion.config import DiffusionConfig
from atom.diffusion.engine.diffusion_engine import DiffusionEngine
from atom.diffusion.registry import (
    MODEL_INDEX_FILENAME,
    checkpoint_architecture,
    pipeline_class_for_checkpoint,
    supported_architectures,
)
from atom.utils import envs, set_atom_log_level

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atom-diffusion-server")
    parser.add_argument("--model", required=True, help="checkpoint root")
    parser.add_argument(
        "--pipeline",
        default=None,
        help="dotted path to a ComposedPipeline subclass; by default it is "
        "looked up from the checkpoint's model_index.json",
    )
    parser.add_argument(
        "--model-variant",
        default=None,
        help="checkpoint partition under --model (e.g. FL2VA, Ref2VA); "
        "omit if --model already points at the partition",
    )
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument(
        "--ulysses-degree",
        type=int,
        default=None,
        help="sequence-parallel degree; defaults to --num-gpus",
    )
    parser.add_argument("--num-inference-steps", type=int, default=50)
    parser.add_argument("--max-queued-jobs", type=int, default=32)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--attn-backend",
        choices=["asm", "triton", "sdpa"],
        default=None,
        help="packed varlen attention kernel; triton reproduces the sglang "
        "reference bit-for-bit, asm is faster",
    )
    parser.add_argument(
        "--no-warmup",
        dest="warmup",
        action="store_false",
        help="skip the throwaway denoise step at load; the first request then "
        "pays the first-forward cost (~11 s on gfx950 at Ulysses-8)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30010)
    parser.add_argument(
        "--log-level",
        default=None,
        help="log level for this server, its workers and the atom loggers; "
        "defaults to $ATOM_LOG_LEVEL, else info",
    )
    return parser


def pipeline_class_for_model(model_path: str) -> str:
    """Pick the pipeline for a checkpoint, or explain what would have worked."""
    dotted = pipeline_class_for_checkpoint(model_path)
    if dotted is not None:
        return dotted
    architecture = checkpoint_architecture(model_path)
    found = (
        f"declares _class_name {architecture!r}"
        if architecture
        else f"has no readable {MODEL_INDEX_FILENAME}"
    )
    raise ValueError(
        f"cannot pick a diffusion pipeline for {model_path!r}: it {found}. "
        f"Supported architectures: {', '.join(supported_architectures())}. "
        "Pass --pipeline with a dotted path to use another."
    )


def config_from_args(args: argparse.Namespace) -> DiffusionConfig:
    # Partitions are separate replicas, so the variant is a path, not a runtime
    # branch. Accept --model /root --model-variant FL2VA or --model /root/FL2VA.
    model_path = args.model
    if args.model_variant:
        candidate = os.path.join(args.model, args.model_variant)
        if os.path.basename(os.path.normpath(args.model)) != args.model_variant:
            model_path = candidate
    return DiffusionConfig(
        model_path=model_path,
        pipeline_class=args.pipeline or pipeline_class_for_model(model_path),
        model_variant=args.model_variant,
        num_gpus=args.num_gpus,
        ulysses_degree=args.ulysses_degree or args.num_gpus,
        num_inference_steps=args.num_inference_steps,
        max_queued_jobs=args.max_queued_jobs,
        output_dir=args.output_dir,
        seed=args.seed,
        warmup=args.warmup,
    )


def build_app(engine: DiffusionEngine):
    from fastapi import FastAPI

    from atom.diffusion.entrypoints.video_api import router

    app = FastAPI(title="ATOM diffusion server")
    app.state.engine = engine
    app.include_router(router)
    return app


def apply_atom_log_level(log_level: str | None) -> str:
    """Give the atom loggers of this process and of its workers one level.

    basicConfig() only sets the root logger's level, and atom.* records never
    consult it: they are filtered by the atom logger's own level, then printed
    by the root handler main() installs. The workers configure their atom
    logger from ATOM_LOG_LEVEL when they import atom.utils after spawn, so an
    explicit --log-level is exported there; this process imported atom.utils
    before parsing argv, so its atom logger is set directly. Without
    --log-level, ATOM_LOG_LEVEL (or INFO) stays in charge. Returns the
    canonical name of the level applied.
    """
    if log_level is not None:
        os.environ["ATOM_LOG_LEVEL"] = log_level
    return set_atom_log_level(envs.ATOM_LOG_LEVEL)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log_level = apply_atom_log_level(args.log_level)
    logging.basicConfig(
        level=log_level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    if args.attn_backend:
        from atom.diffusion.attention import ATTENTION_BACKEND_ENV

        # Set before the workers are spawned so they inherit it: the backend is
        # frozen when each model is constructed, and a replica split across two
        # kernels would be silently inconsistent.
        os.environ[ATTENTION_BACKEND_ENV] = args.attn_backend

    engine = DiffusionEngine(config_from_args(args))
    logger.info("loading %s on %d GPU(s) ...", args.model, args.num_gpus)
    engine.start()

    import uvicorn

    app = build_app(engine)

    # Workers ignore SIGINT and are torn down by close(); handling it here
    # rather than letting uvicorn race the engine is what keeps ~100 GB of VRAM
    # per rank from being stranded in a zombie on shutdown.
    def _shutdown(*_args) -> None:
        logger.info("shutting down")
        engine.close()

    with contextlib.suppress(ValueError):
        signal.signal(signal.SIGTERM, _shutdown)

    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            # An explicit value goes to uvicorn untouched (it also knows
            # "trace"); otherwise the canonical name applied above.
            log_level=args.log_level or log_level.lower(),
        )
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
