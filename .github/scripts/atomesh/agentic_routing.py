#!/usr/bin/env python3
"""Bind measured cache-routing calibration to the benchmark's live executions."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
from pathlib import Path
from urllib.request import urlopen

from atom.cache_routing.keys import namespace_digest


def curve(points, coverage):
    previous = (0, 0)
    if not isinstance(points, list) or not points:
        raise ValueError("calibration needs non-empty measured curves")
    for point in points:
        if len(point) != 2 or not all(
            isinstance(n, (float, int)) and not isinstance(n, bool) and math.isfinite(n)
            for n in point
        ):
            raise ValueError("invalid calibration point")
        if point[0] <= previous[0] or point[1] <= 0 or point[1] < previous[1]:
            raise ValueError("calibration curves must be positive and monotonic")
        previous = point
    if previous[0] < coverage:
        raise ValueError(f"measured curve does not cover {coverage} tokens")


def validate_bundle(bundle, image, max_context=1048576):
    if not bundle:
        raise ValueError(
            "kv_cache_aware requires cache_routing_bundle: measured PP4/TP1 → TP4/DCP4 "
            "calibration plus an immutable namespace manifest. No calibration was supplied; "
            "the performance stage is blocked instead of benchmarking load fallback."
        )
    digest = image.partition("@")[2]
    if (
        bundle.get("schema_version") != 1
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", digest)
        or bundle.get("image", "").partition("@")[2] != digest
    ):
        raise ValueError("routing calibration must bind to this exact image digest")
    if not re.fullmatch(r"[0-9a-f]{64}", bundle.get("measurement_artifact_sha256", "")):
        raise ValueError("routing bundle needs the SHA256 of its measurement artifact")
    namespace_digest(bundle["namespace_manifest"])
    nodes = bundle["nodes"]
    if (
        len(nodes) != 2
        or len(set(nodes)) != 2
        or not all(isinstance(n, str) and n for n in nodes)
    ):
        raise ValueError("routing bundle must name the two measured benchmark nodes")
    executions = bundle["calibration"]["executions"]
    if set(executions) != {"p0", "p1", "d0", "d1"}:
        raise ValueError("routing bundle must describe slots p0,p1,d0,d1")
    for slot, entry in executions.items():
        if not isinstance(entry.get("layout_id"), str) or not entry["layout_id"]:
            raise ValueError(f"missing layout for {slot}")
        costs = entry["costs"]
        if slot.startswith("d"):
            step = costs.get("decode_step_ms")
            if (
                not isinstance(step, (float, int))
                or not math.isfinite(step)
                or step <= 0
            ):
                raise ValueError(f"missing measured decode step for {slot}")
            continue
        buckets = costs["prefill_curves"]
        if not buckets or max(b["max_context_tokens"] for b in buckets) < max_context:
            raise ValueError(
                f"prefill calibration for {slot} must cover the 1M trace context"
            )
        for bucket in buckets:
            curve(bucket["points"], bucket["max_context_tokens"])
        curve(costs["h2d_curve"], max_context)
        paths = entry["transfer_paths"]
        if len(paths) != 2 or {p["destination_execution_id"] for p in paths} != {
            "d0",
            "d1",
        }:
            raise ValueError(
                f"both local and cross-node P→D paths must be measured for {slot}"
            )
        for path in paths:
            destination = executions[path["destination_execution_id"]]
            if (
                path.get("verified") is not True
                or path.get("protocol") != "mooncake"
                or path.get("source_layout_id") != entry["layout_id"]
                or path.get("destination_layout_id") != destination["layout_id"]
            ):
                raise ValueError(f"unverified or mismatched transfer path for {slot}")
            curve(path["transfer_curve"], max_context)
    return bundle


def bundle_from_env():
    bundle = json.loads(os.environ.get("ATOMESH_CACHE_ROUTING_BUNDLE_JSON") or "{}")
    return validate_bundle(bundle, os.environ["DOCKER_IMAGE"])


def execution_id(slot):
    return f"{os.environ['ATOMESH_RUN_TOKEN']}/{slot}"


def role_config(bundle, role, rank, host, ip, port):
    # Slot costs are reusable only on the measured node/GPU placement. Give each
    # job fresh execution IDs, then rebind the validated slot measurements.
    if bundle["nodes"][rank] != host.split(".", 1)[0]:
        raise ValueError(
            f"calibration node {bundle['nodes'][rank]} differs from live host {host}"
        )
    expected_devices = "0,1,2,3" if role == "prefill" else "4,5,6,7"
    if os.environ.get("HIP_VISIBLE_DEVICES") != expected_devices:
        raise ValueError("live GPU placement differs from calibrated PP4/DCP4 slots")
    slot = f"{'p' if role == 'prefill' else 'd'}{rank}"
    return {
        "execution_id": execution_id(slot),
        "catalog_url": f"http://{ip}:{port}",
        "namespace_manifest": bundle["namespace_manifest"],
        "canonical_block_size": 16,
        "cpu_poll_interval_seconds": 0.5,
    }


def get_json(url):
    with urlopen(url, timeout=5) as response:
        return json.load(response)


def prepare_router(bundle, prefills, decodes, output):
    calibration = {"executions": {}}
    discovery = {}
    for role, endpoints in (("p", prefills), ("d", decodes)):
        for rank, url in enumerate(endpoints):
            slot = f"{role}{rank}"
            info = get_json(url.rstrip("/") + "/v1/cache/info")
            expected = bundle["calibration"]["executions"][slot]
            if (
                info["execution_id"] != execution_id(slot)
                or info["layout_id"] != expected["layout_id"]
                or info["content_namespace"]
                != namespace_digest(bundle["namespace_manifest"])
                or info["capabilities"]["exact_prefix_reuse"] is not True
            ):
                raise ValueError(
                    f"live catalog for {slot} differs from calibrated identity/layout"
                )
            layout = info["parallel_layout"]
            geometry = (layout["pp_size"], layout["tp_size"], layout["dcp_size"])
            if geometry != ((4, 1, 1) if role == "p" else (1, 4, 4)):
                raise ValueError(
                    f"unexpected PP/TP/DCP geometry for {slot}: {geometry}"
                )
            entry = copy.deepcopy(expected)
            for path in entry.get("transfer_paths", []):
                path["destination_execution_id"] = execution_id(
                    path["destination_execution_id"]
                )
            calibration["executions"][execution_id(slot)] = entry
            discovery[slot] = info
    output.write_text(json.dumps(calibration, indent=2) + "\n")
    output.with_name("cache-routing-preflight.json").write_text(
        json.dumps(
            {
                "policy": "kv_cache_aware",
                "catalogs": discovery,
                "measurement_artifact_sha256": bundle["measurement_artifact_sha256"],
            },
            indent=2,
        )
        + "\n"
    )


def decision_counts(text):
    counts = {"selected": 0, "fallback": 0}
    for line in text.splitlines():
        match = re.fullmatch(
            r'atomesh_kv_cache_routing_decisions_total\{outcome="(selected|fallback)"\}\s+([0-9.eE+-]+)',
            line,
        )
        if match:
            value = float(match[2])
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid routing decision counter")
            counts[match[1]] += value
    return counts


def record_decisions(url, output, before=None):
    with urlopen(url, timeout=5) as response:
        counts = decision_counts(response.read().decode())
    if before is not None:
        baseline = json.loads(before.read_text())
        counts = {k: v - baseline[k] for k, v in counts.items()}
        total = sum(counts.values())
        if any(n < 0 for n in counts.values()):
            raise ValueError("routing counters reset during the benchmark")
        counts["selected_fraction"] = counts["selected"] / total if total else 0
        counts["policy"] = "kv_cache_aware"
    output.write_text(json.dumps(counts, indent=2) + "\n")
    if before is not None and counts["selected"] <= 0:
        raise ValueError(
            "kv_cache_aware made no calibrated selections; all-fallback runs are invalid"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    validate = sub.add_parser("validate-matrix")
    validate.add_argument("--matrix", type=Path, required=True)
    role = sub.add_parser("role")
    role.add_argument("--role", choices=("prefill", "decode"), required=True)
    role.add_argument("--rank", type=int, required=True)
    role.add_argument("--host", required=True)
    role.add_argument("--ip", required=True)
    role.add_argument("--port", type=int, required=True)
    router = sub.add_parser("router")
    router.add_argument("--prefill", action="append", required=True)
    router.add_argument("--decode", action="append", required=True)
    router.add_argument("--output", type=Path, required=True)
    metrics = sub.add_parser("decisions")
    metrics.add_argument("--url", required=True)
    metrics.add_argument("--output", type=Path, required=True)
    metrics.add_argument("--before", type=Path)
    args = parser.parse_args()
    if args.action == "validate-matrix":
        for cell in json.loads(args.matrix.read_text())["include"]:
            if cell["scaling"]["scale"] == 1:
                continue
            router = cell["service"]["router"]
            if any(
                router.get(key) != "kv_cache_aware"
                for key in ("policy", "prefill_policy", "decode_policy")
            ):
                raise ValueError("2P2D must use kv_cache_aware for both P and D")
            validate_bundle(
                cell["benchmark"].get("cache_routing_bundle"), cell["image"]
            )
        return
    if args.action == "decisions":
        record_decisions(args.url, args.output, args.before)
        return
    bundle = bundle_from_env()
    if args.action == "role":
        print(
            json.dumps(
                role_config(bundle, args.role, args.rank, args.host, args.ip, args.port)
            )
        )
    else:
        prepare_router(bundle, args.prefill, args.decode, args.output)


if __name__ == "__main__":
    main()
