"""Execute launcher argument parsing/config with uvicorn.run intercepted only.

No process launch or model encoding is claimed by this CLI probe.
"""

import argparse
import hashlib
import importlib.util
import io
import json
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.launcher
    spec = importlib.util.spec_from_file_location("review_actual_launcher", source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import uvicorn

    records = []

    def invoke(name, flags, expected_exit=0, expected_processes=4):
        captured = []
        stderr = io.StringIO()
        stdout = io.StringIO()
        with (
            patch.object(
                sys,
                "argv",
                [
                    str(source),
                    "--model",
                    "real-checkpoint",
                    "--backend",
                    "http://router",
                    *flags,
                ],
            ),
            patch.object(
                uvicorn, "run", side_effect=lambda *a, **kw: captured.append((a, kw))
            ),
            redirect_stderr(stderr),
            redirect_stdout(stdout),
        ):
            exit_code = 0
            try:
                mod.main()
            except SystemExit as exc:
                exit_code = exc.code
        assert exit_code == expected_exit, (name, exit_code, stderr.getvalue())
        record = {"id": name, "flags": flags, "exit_code": exit_code, "status": "PASS"}
        if expected_exit == 0:
            assert len(captured) == 1
            uv_args, kw = captured[0]
            assert (
                uv_args == (source.stem + ":build_app",)
                and kw["factory"] is True
                and kw["workers"] == expected_processes
            )
            config = json.loads(os.environ[mod.CONFIG_ENV])
            record.update({"per_process_config": config, "uvicorn": kw})
            assert (
                config["workers"] * expected_processes == 16
                and config["queued"] * expected_processes == 1280
            )
            assert (
                config["max_connections"] * expected_processes == 2048
                and config["max_keepalive_connections"] * expected_processes == 512
            )
        else:
            assert not captured
            record["error"] = stderr.getvalue().strip()
        records.append(record)

    invoke("default-four-process-total-budget", [])
    invoke(
        "historical-recipe-total-budget",
        [
            "--processes",
            "4",
            "--workers",
            "16",
            "--queued",
            "1280",
            "--max-connections",
            "2048",
            "--max-keepalive-connections",
            "512",
            "--timeout-keep-alive",
            "600",
            "--segment-cache-mib",
            "0",
        ],
    )
    invoke("huggingface-same-budget", ["--encode-backend", "huggingface"])
    for name, flags in [
        ("zero-processes", ["--processes", "0"]),
        ("negative-threads", ["--workers", "-1"]),
        ("zero-capacity", ["--queued", "0"]),
        ("indivisible-threads", ["--workers", "15"]),
        ("indivisible-capacity", ["--queued", "1281"]),
        ("indivisible-http-budget", ["--max-connections", "2047"]),
        ("invalid-idle-limit", ["--max-keepalive-connections", "2052"]),
        ("negative-keepalive-time", ["--timeout-keep-alive", "-1"]),
        ("nonzero-segment-cache", ["--segment-cache-mib", "1"]),
        ("unknown-encode-backend", ["--encode-backend", "unreviewed"]),
    ]:
        invoke(name, flags, expected_exit=2)
    data = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "python": sys.version,
        "uvicorn_version": uvicorn.__version__,
        "scope": "Actual launcher main/argument parsing/config; uvicorn.run intercepted, no real processes/model loaded",
        "results": records,
    }
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": len(records),
                "source_sha256": data["source_sha256"],
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
