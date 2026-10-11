# SPDX-License-Identifier: MIT
"""Benchmark Qwen3.8 HC FlyDSL stages against the loaded ATOM module path.

This uses synthetic weights on one idle gfx950 GPU. Supply --dump-ir-dir to
save the compiler's final AMDGCN ISA for each of the four FlyDSL stages.
"""

import argparse
import json
import os
import re
import statistics
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch
import triton.testing

from atom.model_ops import linear
from atom.model_ops.qwen4_exp.hyperconnection import Qwen4ExpHyperConnection
from atom.model_ops.qwen4_exp.ops.hc_flydsl import (
    build_hyperconnection,
    combine,
    mix,
)
from atom.model_ops.qwen4_exp.ops.hc_flydsl import (
    mix_fused as flydsl_mix_fused,
)


def measure(fn, repeats):
    fn()
    torch.cuda.synchronize()
    return round(
        statistics.median(
            triton.testing.do_bench_cudagraph(fn, rep=1) * 1000 for _ in range(repeats)
        ),
        3,
    )


def benchmark_batch(hc, samples, repeats):
    x = (
        torch.randn(samples, 10240, device="cuda", dtype=torch.bfloat16) * 0.5
    ).contiguous()
    block = (
        torch.randn(samples, 2560, device="cuda", dtype=torch.bfloat16) * 0.5
    ).contiguous()
    native_mixed, native_residual = hc.mix(x)
    native_combined = hc.combine(block, native_residual)
    mixed, residual = mix(hc, x)
    combined = combine(hc, block, residual)
    mix_nrmse = float(
        torch.linalg.norm((mixed - native_mixed).float())
        / torch.linalg.norm(native_mixed.float())
    )
    combine_nrmse = float(
        torch.linalg.norm((combined - native_combined).float())
        / torch.linalg.norm(native_combined.float())
    )
    normed = torch.empty_like(x)
    low = torch.empty((samples, 320), device="cuda", dtype=torch.bfloat16)
    projected = torch.empty_like(mixed)
    output = torch.empty_like(x)
    norm, down, up, inject = build_hyperconnection(samples, hc.hc_norm.variance_epsilon)
    stages = {
        "norm": lambda: norm(
            x.data_ptr(),
            hc.hc_norm.weight.data_ptr(),
            normed.data_ptr(),
            x.data_ptr(),
            x.data_ptr(),
            x.data_ptr(),
            stream=torch.cuda.current_stream(),
        ),
        "down_silu": lambda: down(
            normed.data_ptr(),
            hc.input_mix_weight_down.weight.data_ptr(),
            low.data_ptr(),
            low.data_ptr(),
            stream=torch.cuda.current_stream(),
        ),
        "up_gated_mean": lambda: up(
            normed.data_ptr(),
            low.data_ptr(),
            hc.input_mix_weight_up.weight.data_ptr(),
            projected.data_ptr(),
            stream=torch.cuda.current_stream(),
        ),
        "inject_combine": lambda: inject(
            x.data_ptr(),
            normed.data_ptr(),
            block.data_ptr(),
            hc.block_inject_weight.weight.data_ptr(),
            output.data_ptr(),
            stream=torch.cuda.current_stream(),
        ),
    }
    native_us = measure(lambda: hc.combine(block, hc.mix(x)[1]), repeats)
    flydsl_us = measure(lambda: combine(hc, block, mix(hc, x)[1]), repeats)
    return {
        "native_pair_us": native_us,
        "flydsl_pair_us": flydsl_us,
        "speedup": round(native_us / flydsl_us, 3),
        "mix_nrmse": round(mix_nrmse, 7),
        "combine_nrmse": round(combine_nrmse, 7),
        "stage_us": {name: measure(fn, repeats) for name, fn in stages.items()},
    }


def benchmark_served(hc, samples, repeats):
    """Measure the actual decoder's deferred-combine HC path."""
    x = (
        torch.randn(samples, 10240, device="cuda", dtype=torch.bfloat16) * 0.5
    ).contiguous()
    block = (
        torch.randn(samples, 2560, device="cuda", dtype=torch.bfloat16) * 0.5
    ).contiguous()
    raw = (
        torch.randn(samples, 4, device="cuda", dtype=torch.bfloat16) * 0.5
    ).contiguous()
    pending = (block, raw)
    expected, (expected_streams, expected_raw) = hc.mix_fused(x, pending)
    actual, (streams, injected) = flydsl_mix_fused(hc, x, pending)
    nrmse = float(
        torch.linalg.norm((actual - expected).float())
        / torch.linalg.norm(expected.float())
    )
    stream_nrmse = float(
        torch.linalg.norm((streams - expected_streams).float())
        / torch.linalg.norm(expected_streams.float())
    )
    raw_nrmse = float(
        torch.linalg.norm((injected - expected_raw).float())
        / torch.linalg.norm(expected_raw.float())
    )
    native_us = measure(lambda: hc.mix_fused(x, pending), repeats)
    flydsl_us = measure(lambda: flydsl_mix_fused(hc, x, pending), repeats)
    return {
        "native_us": native_us,
        "flydsl_us": flydsl_us,
        "speedup": round(native_us / flydsl_us, 3),
        "mixed_nrmse": round(nrmse, 7),
        "streams_nrmse": round(stream_nrmse, 7),
        "inject_nrmse": round(raw_nrmse, 7),
    }


def isa_summary(directory):
    report = {}
    for stage in ("norm", "down", "up", "combine"):
        isa = directory / f"{stage}_kernel_0" / "22_final_isa.s"
        if not isa.exists():
            raise FileNotFoundError(f"FlyDSL did not emit {isa}")
        source = isa.read_text()
        counts = Counter()
        for line in source.splitlines():
            line = line.strip().split(";", 1)[0]
            if line and not line.startswith((".", "#")) and not line.endswith(":"):
                counts[line.split()[0]] += 1

        def directive(name, source):
            found = re.search(
                r"^\s*\.amdhsa_" + name + r"\s+(\d+)", source, re.MULTILINE
            )
            return int(found.group(1)) if found else None

        report[stage] = {
            "vgprs": directive("next_free_vgpr", source),
            "lds_bytes": directive("group_segment_fixed_size", source),
            "dot2": sum(
                n
                for op, n in counts.items()
                if op.startswith(("v_dot2_f32_bf16", "v_dot2c_f32_bf16"))
            ),
            "mfma": sum(n for op, n in counts.items() if op.startswith("v_mfma")),
            "valu": sum(n for op, n in counts.items() if op.startswith("v_")),
            "loads": sum(n for op, n in counts.items() if op.startswith("buffer_load")),
            "stores": sum(
                n for op, n in counts.items() if op.startswith("buffer_store")
            ),
            "lds_ops": sum(n for op, n in counts.items() if op.startswith("ds_")),
            "scratch_ops": sum(n for op, n in counts.items() if "scratch" in op),
            "exp": sum(n for op, n in counts.items() if op.startswith("v_exp")),
            "rcp": sum(n for op, n in counts.items() if op.startswith("v_rcp")),
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--dump-ir-dir", type=Path)
    args = parser.parse_args()
    if any(not 1 <= n <= 16 for n in args.batches):
        raise ValueError("batches must be 1..16")
    if args.dump_ir_dir:
        os.environ["FLYDSL_DUMP_IR"] = "1"
        os.environ["FLYDSL_DUMP_DIR"] = str(args.dump_ir_dir)
    os.environ["ATOM_QWEN38_MONO_HC"] = "0"
    if (
        getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(":")[0]
        != "gfx950"
    ):
        raise RuntimeError("Qwen3.8 HC benchmark needs gfx950")
    linear.get_tp_group = lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    torch.manual_seed(7)
    hc = Qwen4ExpHyperConnection(2560, 4, 320).cuda()
    with torch.no_grad():
        for weight in hc.parameters():
            weight.normal_(0, 0.02)
    hc.process_weights_after_loading()
    report = {
        "gpu": torch.cuda.get_device_name(0),
        "checkpoint": "synthetic",
        "batches": {},
    }
    for samples in args.batches:
        report["batches"][str(samples)] = benchmark_batch(hc, samples, args.repeats)
        if args.dump_ir_dir:
            report["batches"][str(samples)]["isa"] = isa_summary(args.dump_ir_dir)
        report["batches"][str(samples)]["served_fused"] = benchmark_served(
            hc, samples, args.repeats
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
