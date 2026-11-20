"""Start independent exact ATOM encoders with a fixed total CPU/HTTP budget."""

import argparse
import json
import os
from pathlib import Path

CONFIG_ENV = "ATOM_EXACT_FACADE_PROCESS_CONFIG"


def config_for_args(args):
    if (
        args.processes < 1
        or args.workers < 1
        or args.queued < 1
        or args.max_connections < 1
    ):
        raise ValueError(
            "processes, total encoder workers, and total queued slots must be positive"
        )
    if (
        args.workers % args.processes
        or args.queued % args.processes
        or args.max_connections % args.processes
        or args.max_keepalive_connections % args.processes
    ):
        raise ValueError(
            "total workers, slots, and HTTP budgets must divide evenly across processes"
        )
    if not 0 <= args.max_keepalive_connections <= args.max_connections:
        raise ValueError(
            "keepalive connections must be between zero and total max connections"
        )
    if args.timeout_keep_alive < 0:
        raise ValueError("timeout keep alive must be nonnegative")
    return {
        "model": args.model,
        "backend": args.backend,
        "workers": args.workers // args.processes,
        "queued": args.queued // args.processes,
        "max_connections": args.max_connections // args.processes,
        "max_keepalive_connections": args.max_keepalive_connections // args.processes,
        "encode_backend": args.encode_backend,
    }


def build_app():
    # Uvicorn's spawn workers call this independently. No parent executor,
    # AsyncClient, asyncio semaphore, encoder, or request state crosses a process.
    config = json.loads(os.environ[CONFIG_ENV])
    from exact_atom_chat import ExactAtomChatEncoder
    from exact_tokenizer_ingress import create_app
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tok = AutoTokenizer.from_pretrained(
        config["model"], use_fast=True, trust_remote_code=True
    )
    probe = "Hello world 你好"
    if tok.decode(tok.encode(probe), skip_special_tokens=True) != probe:
        tok = PreTrainedTokenizerFast.from_pretrained(config["model"])
    if config["encode_backend"] == "fastokens":
        from fastokens_final_encode import install_fastokens_final_encode

        metadata = install_fastokens_final_encode(tok, bpe_threads=1)
        print("FASTOKENS_ADAPTER_READY " + json.dumps(metadata), flush=True)
    encoder = ExactAtomChatEncoder(config["model"], tok, defaults={})
    return create_app(
        encoder,
        config["backend"],
        workers=config["workers"],
        queued=config["queued"],
        max_connections=config["max_connections"],
        max_keepalive_connections=config["max_keepalive_connections"],
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--backend", required=True)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=24671)
    p.add_argument(
        "--encode-backend",
        choices=["fastokens", "huggingface"],
        default="fastokens",
        help="only the final text encode backend; ATOM chat rendering is unchanged",
    )
    p.add_argument("--processes", type=int, default=4)
    p.add_argument(
        "--workers",
        type=int,
        default=16,
        help="total encoder threads across all HTTP processes",
    )
    p.add_argument(
        "--queued",
        type=int,
        default=1280,
        help="total native capacity slots across all HTTP processes",
    )
    p.add_argument(
        "--max-connections",
        type=int,
        default=2048,
        help="total upstream HTTP connection budget",
    )
    p.add_argument(
        "--max-keepalive-connections",
        type=int,
        default=512,
        help="total upstream idle keepalive budget",
    )
    p.add_argument(
        "--segment-cache-mib",
        type=int,
        choices=[0],
        default=0,
        help="only disabled segment caching is supported",
    )
    p.add_argument("--timeout-keep-alive", type=int, default=5)
    args = p.parse_args()
    try:
        config = config_for_args(args)
    except ValueError as exc:
        p.error(str(exc))
    os.environ[CONFIG_ENV] = json.dumps(config)
    import uvicorn

    uvicorn.run(
        f"{Path(__file__).stem}:build_app",
        factory=True,
        host=args.host,
        port=args.port,
        workers=args.processes,
        access_log=False,
        timeout_keep_alive=args.timeout_keep_alive,
    )


if __name__ == "__main__":
    main()
