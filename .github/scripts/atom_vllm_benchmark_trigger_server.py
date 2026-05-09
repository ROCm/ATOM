#!/usr/bin/env python3
"""Local web trigger for ATOM vLLM benchmark workflow."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
STATIC_DIR = REPO_ROOT / ".github/pages/atom-vllm-benchmark-trigger"
CATALOG_PATH = REPO_ROOT / ".github/benchmark/oot_benchmark_models.json"
WORKFLOW_FILE = "atom-vllm-benchmark.yaml"
MAX_MODEL_SLOTS = 8
MANUAL_SELECTION_GUIDE_DEFAULT = (
    "Reference only - open dropdown for model -> TP sizes"
)


def load_catalog(path: str | Path = CATALOG_PATH) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "families" not in payload:
        raise ValueError("Expected a family-based benchmark model catalog.")
    return payload


def _family_map(catalog: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {family["choice_label"]: family for family in catalog["families"]}


def _normalize_tp_list(
    family_name: str,
    tp_sizes: list[Any],
    supported_tp_sizes: list[int],
) -> list[int]:
    normalized = []
    for tp_size in tp_sizes:
        value = int(tp_size)
        if value not in supported_tp_sizes:
            unsupported_text = ",".join(str(item) for item in tp_sizes)
            supported_text = ",".join(str(item) for item in supported_tp_sizes)
            raise ValueError(
                f"{family_name} does not support TP sizes {unsupported_text}; "
                f"supported TP sizes: {supported_text}"
            )
        if value not in normalized:
            normalized.append(value)
    return normalized


def _serialize_tp_sizes(tp_sizes: list[int]) -> str:
    return ",".join(str(tp_size) for tp_size in tp_sizes)


def _serialize_isl_osl_pairs(pairs: list[list[int] | tuple[int, int]]) -> str:
    serialized = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"Invalid ISL/OSL pair: {pair!r}")
        input_length = int(pair[0])
        output_length = int(pair[1])
        serialized.append(f"{input_length},{output_length}")
    if not serialized:
        raise ValueError("At least one ISL/OSL pair is required.")
    return ";".join(serialized)


def _serialize_csv(values: list[Any], *, field_name: str) -> str:
    if not values:
        raise ValueError(f"At least one {field_name} value is required.")
    return ",".join(str(value) for value in values)


def build_workflow_inputs(
    payload: dict[str, Any],
    catalog: dict[str, Any],
) -> dict[str, str]:
    model_selections = payload.get("model_selections", [])
    if len(model_selections) > MAX_MODEL_SLOTS:
        raise ValueError(
            f"The local trigger supports at most {MAX_MODEL_SLOTS} model selections."
        )

    family_map = _family_map(catalog)
    inputs: dict[str, str] = {
        "manual_selection_guide": MANUAL_SELECTION_GUIDE_DEFAULT,
        "benchmark_client": str(
            payload.get("benchmark_client", "InferenceMax bench")
        ),
        "isl_osl_pairs": _serialize_isl_osl_pairs(payload.get("isl_osl_pairs", [])),
        "concurrency_values": _serialize_csv(
            payload.get("concurrency_values", []),
            field_name="concurrency",
        ),
        "random_range_ratios": _serialize_csv(
            payload.get("random_range_ratios", ["0.8"]),
            field_name="random range ratio",
        ),
        "oot_image": str(payload.get("oot_image", "")),
        "publish_to_dashboard": str(
            bool(payload.get("publish_to_dashboard", False))
        ).lower(),
        "upload_to_custom_dashboard": str(
            bool(payload.get("upload_to_custom_dashboard", True))
        ).lower(),
    }

    for slot_idx in range(1, MAX_MODEL_SLOTS + 1):
        inputs[f"model_slot_{slot_idx}"] = "none"
        inputs[f"tp_sizes_slot_{slot_idx}"] = ""

    for slot_idx, selection in enumerate(model_selections, start=1):
        family_name = str(selection["family"])
        family = family_map.get(family_name)
        if family is None:
            raise ValueError(f"Unknown model family: {family_name}")

        supported_tp_sizes = [int(variant["tp_size"]) for variant in family["variants"]]
        requested_tp_sizes = selection.get("tp_sizes", [])
        if not requested_tp_sizes:
            if len(supported_tp_sizes) == 1:
                normalized_tp_sizes = supported_tp_sizes
            else:
                raise ValueError(
                    f"TP sizes are required for multi-TP family {family_name}."
                )
        else:
            normalized_tp_sizes = _normalize_tp_list(
                family_name,
                requested_tp_sizes,
                supported_tp_sizes,
            )

        unsupported = [
            tp_size for tp_size in normalized_tp_sizes if tp_size not in supported_tp_sizes
        ]
        if unsupported:
            unsupported_text = ",".join(str(item) for item in unsupported)
            supported_text = ",".join(str(item) for item in supported_tp_sizes)
            raise ValueError(
                f"{family_name} does not support TP sizes {unsupported_text}; "
                f"supported TP sizes: {supported_text}"
            )

        inputs[f"model_slot_{slot_idx}"] = family_name
        inputs[f"tp_sizes_slot_{slot_idx}"] = _serialize_tp_sizes(normalized_tp_sizes)

    return inputs


def build_gh_workflow_run_command(
    *,
    repository: str,
    ref: str,
    inputs: dict[str, str],
) -> list[str]:
    command = [
        "gh",
        "workflow",
        "run",
        WORKFLOW_FILE,
        "--repo",
        repository,
        "--ref",
        ref,
    ]
    for key, value in inputs.items():
        command.extend(["-f", f"{key}={value}"])
    return command


def detect_current_ref(repo_root: Path = REPO_ROOT) -> str:
    return (
        subprocess.check_output(
            ["git", "branch", "--show-current"],
            cwd=repo_root,
            text=True,
        )
        .strip()
    )


def detect_repository_slug(repo_root: Path = REPO_ROOT) -> str:
    remote_url = (
        subprocess.check_output(
            ["git", "remote", "get-url", "origin"],
            cwd=repo_root,
            text=True,
        )
        .strip()
    )
    if remote_url.startswith("git@github.com:"):
        return remote_url.removeprefix("git@github.com:").removesuffix(".git")
    if remote_url.startswith("https://github.com/"):
        return remote_url.removeprefix("https://github.com/").removesuffix(".git")
    raise ValueError(f"Unsupported GitHub remote URL format: {remote_url}")


def catalog_for_ui(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    families = []
    for family in catalog["families"]:
        tp_sizes = [int(variant["tp_size"]) for variant in family["variants"]]
        families.append(
            {
                "family": family["choice_label"],
                "supported_tp_sizes": tp_sizes,
                "runner": family.get("runner", ""),
                "variants": family["variants"],
            }
        )
    return families


def dispatch_workflow(payload: dict[str, Any], catalog: dict[str, Any]) -> dict[str, Any]:
    repository = str(payload.get("repository") or detect_repository_slug())
    ref = str(payload.get("ref") or detect_current_ref())
    inputs = build_workflow_inputs(payload, catalog)
    command = build_gh_workflow_run_command(
        repository=repository,
        ref=ref,
        inputs=inputs,
    )
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            completed.stderr.strip()
            or completed.stdout.strip()
            or "gh workflow run failed."
        )
    return {
        "repository": repository,
        "ref": ref,
        "command": command,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "inputs": inputs,
    }


class TriggerRequestHandler(SimpleHTTPRequestHandler):
    catalog = load_catalog()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/api/catalog":
            self._send_json({"families": catalog_for_ui(self.catalog)})
            return
        if parsed.path == "/api/config":
            self._send_json(
                {
                    "repository": detect_repository_slug(),
                    "ref": detect_current_ref(),
                    "workflow_file": WORKFLOW_FILE,
                    "max_model_slots": MAX_MODEL_SLOTS,
                    "manual_selection_guide_default": MANUAL_SELECTION_GUIDE_DEFAULT,
                }
            )
            return
        return super().do_GET()

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/api/dispatch":
            self._send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length)
        try:
            payload = json.loads(body.decode("utf-8"))
            result = dispatch_workflow(payload, self.catalog)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
            return
        except RuntimeError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_GATEWAY)
            return
        except Exception as exc:  # noqa: BLE001
            self._send_json(
                {"error": f"Unexpected server error: {exc}"},
                status=HTTPStatus.INTERNAL_SERVER_ERROR,
            )
            return

        self._send_json({"ok": True, **result})


def main() -> int:
    parser = argparse.ArgumentParser(description="Local ATOM vLLM benchmark trigger")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), TriggerRequestHandler)
    print(f"Serving local benchmark trigger at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
