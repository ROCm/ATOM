# SPDX-License-Identifier: MIT
"""Fresh compile and resident-grid gate for all V4 TP8 production profiles."""

import argparse
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from atom.benchmarks.results.io import write_json


@dataclass(frozen=True)
class BuildCase:
    profile: str
    rows: int
    tokens_per_seq: int
    compress_ratio: int
    refresh: bool
    phase: str


def production_cases():
    for profile, rows, tokens in (
        ("ordinary", 1, 1),
        ("ordinary", 4, 1),
        ("ordinary", 8, 1),
        ("mtp3_target", 4, 4),
    ):
        for ratio in (128, 4):
            refresh_modes = (True, False) if ratio == 4 and tokens == 1 else (True,)
            for refresh in refresh_modes:
                for phase in (("one",) if rows == 1 else ("attention", "moe")):
                    yield BuildCase(profile, rows, tokens, ratio, refresh, phase)


def compile_case(case, max_model_len):
    import inspect

    from atom.models.deepseek_v4.mono.config import V4Config, shape_dims
    from atom.models.deepseek_v4.mono.kernels.layer import build_dsv4_kernel
    from atom.models.deepseek_v4.mono.runner import V4MonoDecodeRunner
    from atom.mono.runtime.compile import (
        compile_only,
        launcher_resources,
        require_resident_grid,
    )

    cfg = V4Config(
        compress_ratio=case.compress_ratio,
        max_seq=max_model_len,
        kv_fp8=True,
        bf16_kv_roundtrip=False,
        shared_bf16_block128=True,
        routed_down_bf16=True,
        compressor_projection_bf16=True,
        attention_fp8_bf16=True,
        indexer_hadamard=False,
    )
    plan = V4MonoDecodeRunner._launch_plan(case.rows)
    launch = build_dsv4_kernel(
        case.rows,
        cfg.heads,
        8,
        cfg.window,
        scale=cfg.softmax_scale,
        tokens_per_seq=case.tokens_per_seq,
        swiglu_limit=cfg.swiglu_limit,
        hc_sinkhorn_iters=cfg.hc_sinkhorn_iters,
        hc_eps=cfg.hc_eps,
        bf16_kv_roundtrip=cfg.bf16_kv_roundtrip,
        window_rows=cfg.cache_rows,
        phase=case.phase,
        index_topk_refresh=case.refresh,
        **{key: value for key, value in plan.items() if key != "launch_shape"},
        **shape_dims(cfg),
    )
    nargs = sum(
        key != "stream" for key in inspect.signature(launch._original_func).parameters
    )
    with compile_only():
        launch(*([0] * nargs))
    resources = launcher_resources(launch)
    if not resources:
        raise RuntimeError(f"no compiled binary for {case}")
    for resource in resources:
        if resource["scratch"] or resource["lds"] > 163840:
            raise RuntimeError(
                f"unsafe persistent-kernel resource for {case}: {resource}"
            )
    require_resident_grid(launch, 256, 512)
    return resources


def main():
    import os
    from dataclasses import asdict

    import flydsl
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.max_model_len <= 0:
        parser.error("max-model-len must be positive")
    if torch.cuda.device_count() != 8 or any(
        not torch.cuda.get_device_properties(i).gcnArchName.startswith("gfx950")
        or torch.cuda.get_device_properties(i).multi_processor_count != 256
        for i in range(8)
    ):
        raise RuntimeError("V4 Mono qualification requires eight gfx950/256-CU devices")
    report = {
        "version": 1,
        "max_model_len": args.max_model_len,
        "flydsl_module": flydsl.__file__,
        "status": "in_progress",
        "cases": [],
    }
    write_json(args.output, report)
    # A fresh, explicitly owned compiler directory; never erase shared caches.
    with tempfile.TemporaryDirectory(prefix="atom-v4-mono-compile-") as cache:
        os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = cache
        for case in production_cases():
            started = time.monotonic()
            print(f"Compiling {case}", flush=True)
            try:
                resources = compile_case(case, args.max_model_len)
            except Exception as error:
                report.update(
                    status="failed", failed_case=asdict(case), error=str(error)
                )
                write_json(args.output, report)
                raise
            report["cases"].append(
                {
                    **asdict(case),
                    "resources": resources,
                    "seconds": time.monotonic() - started,
                }
            )
            write_json(args.output, report)
    report["status"] = "passed"
    write_json(args.output, report)
    print(
        f"All {len(report['cases'])} production builds passed resident-grid admission"
    )


if __name__ == "__main__":
    main()
