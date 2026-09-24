# SPDX-License-Identifier: MIT
import argparse
import json
import sys
from pathlib import Path

from .io import read_json, write_json


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Versioned ATOM benchmark artifacts (CPU-only)"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("capture-launch")
    launch.add_argument("--output", required=True)
    launch.add_argument("args", nargs=argparse.REMAINDER)
    dataset = commands.add_parser("capture-hf-dataset")
    dataset.add_argument("--cache-dir", required=True)
    dataset.add_argument("--export", required=True)
    dataset.add_argument("--output", required=True)
    capture = commands.add_parser("capture-config")
    capture.add_argument("--output", required=True)
    capture.add_argument("--launch")
    capture.add_argument("--model", required=True)
    capture.add_argument("--kind", choices=("random", "agentic"), required=True)
    capture.add_argument("--concurrency", type=int, required=True)
    capture.add_argument("args", nargs=argparse.REMAINDER)
    build = commands.add_parser("build")
    build.add_argument("--config", required=True)
    build.add_argument("--records")
    build.add_argument("--record-format", choices=("atom", "aiperf"), default="atom")
    build.add_argument("--raw-dir")
    build.add_argument("--telemetry-dir")
    build.add_argument("--client-log")
    build.add_argument("--server-log")
    build.add_argument("--collector-log")
    build.add_argument("--client-launch")
    build.add_argument("--harness-summary")
    build.add_argument("--output", required=True)
    build.add_argument("--exit-code", type=int, default=0)
    build.add_argument("--require-full", action="store_true")
    check = commands.add_parser("verify")
    check.add_argument("bundle")
    check.add_argument("--require-full", action="store_true")
    index = commands.add_parser("index")
    index.add_argument("directory")
    index.add_argument("--output", required=True)
    index.add_argument("--summary-only", action="store_true")
    index.add_argument("--plan")
    for command in ("summary-package", "rebuild"):
        sub = commands.add_parser(command)
        sub.add_argument("bundle")
        sub.add_argument("--output", required=True)
    telemetry = commands.add_parser("collect")
    telemetry.add_argument("--config", required=True)
    telemetry.add_argument("--output", required=True)
    telemetry.add_argument("--interval", type=float, default=1.0)
    telemetry.add_argument("--server-url")
    args = parser.parse_args(argv)
    try:
        if args.command == "capture-launch":
            from .metadata import git_sha, performance_env

            write_json(
                args.output,
                {
                    "argv": args.args[1:] if args.args[:1] == ["--"] else args.args,
                    "environment": performance_env(),
                    "atom_sha": git_sha(Path.cwd()),
                },
            )
        elif args.command == "capture-hf-dataset":
            from .metadata import capture_hf_dataset

            write_json(args.output, capture_hf_dataset(args.cache_dir, args.export))
        elif args.command == "capture-config":
            from .metadata import capture_config

            launch = (
                read_json(args.launch)
                if args.launch and Path(args.launch).is_file()
                else None
            )
            config = capture_config(
                args.model,
                args.args[1:] if args.args[:1] == ["--"] else args.args,
                args.kind,
                args.concurrency,
                launch,
            )
            write_json(args.output, config)
        elif args.command == "build":
            from .bundle import build_bundle

            logs = {
                k: v
                for k, v in {
                    "client": args.client_log,
                    "server": args.server_log,
                    "collector": args.collector_log,
                }.items()
                if v
            }
            result = build_bundle(
                read_json(args.config),
                args.records,
                args.output,
                record_format=args.record_format,
                raw_dir=args.raw_dir,
                telemetry_dir=args.telemetry_dir,
                logs=logs,
                exit_code=args.exit_code,
                require_full=args.require_full,
                client_launch=args.client_launch,
                harness_summary=args.harness_summary,
            )
            validation = read_json(Path(args.output) / "validation.json")
            print(
                json.dumps(
                    {
                        "output": args.output,
                        "status": result["status"],
                        "point_id": result["point_id"],
                        "missing": validation["missing"],
                    }
                )
            )
            for reason in validation["missing"]:
                print(f"Bundle incomplete: {reason}", file=sys.stderr)
            if result["status"] == "failed":
                return 1
            if args.require_full and result["status"] != "complete":
                return 2
        elif args.command == "verify":
            from .bundle import verify_bundle

            result = verify_bundle(args.bundle)
            print(
                json.dumps({"status": result["status"], "files": len(result["files"])})
            )
            if args.require_full and result["status"] != "complete":
                return 2
        elif args.command == "index":
            from .bundle import run_index

            run_index(Path(args.directory), args.output, args.summary_only, args.plan)
        elif args.command == "summary-package":
            from .bundle import summary_package

            summary_package(args.bundle, args.output)
        elif args.command == "rebuild":
            from .bundle import rebuild_bundle

            result = rebuild_bundle(args.bundle, args.output)
            print(
                json.dumps(
                    {
                        "output": args.output,
                        "status": result["status"],
                        "point_id": result["point_id"],
                    }
                )
            )
        elif args.command == "collect":
            from .telemetry import collect

            collect(read_json(args.config), args.output, args.interval, args.server_url)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"{args.command}: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
