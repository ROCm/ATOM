"""Bounded native MXFP8 tile comparison; run inside the pinned worker image on one leased GPU."""

import argparse
import ctypes
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import time

PLAN = {
    "N": 9856,
    "K": 6144,
    "M": [16, 24, 32, 48, 64, 80, 160],
    "rotating_weights": 8,
    "configs": [
        [16, 32, 256, 4, 2],
        [16, 64, 256, 4, 2],
        [32, 32, 512, 4, 2],
        [32, 64, 256, 4, 2],
        [32, 64, 512, 8, 2],
        [64, 64, 512, 8, 2],
    ],
    "purpose": "Compare the installed native prequantized MXFP8 GEMM with explicit launch configurations; no worker-source patch and no SA performance point.",
    "limits": "Checkpoint-derived TP1 QKV/index weight geometry. Synthetic inputs, rotating weights and isolated GPU; not an end-to-end speedup claim. Hardware roofline and path ablations are not measured by this initial candidate comparison.",
}


def validate_plan():
    # Q + K + V + index_q + the single replicated index_k head.
    assert PLAN["N"] == 8192 + 512 + 512 + 512 + 128
    assert PLAN["K"] % 128 == 0
    for bm, bn, bk, warps, stages in PLAN["configs"]:
        assert bm > 0 and bn > 0 and PLAN["K"] % bk == 0
        assert warps in (4, 8) and stages == 2
    assert PLAN["rotating_weights"] * PLAN["N"] * PLAN["K"] > 512_000_000 // 2


def kernel_resources(kernel):
    resources = {
        "registers": kernel.n_regs,
        "spills": kernel.n_spills,
        "lds_bytes": kernel.metadata.shared,
        "warps": kernel.metadata.num_warps,
    }
    try:
        hip = ctypes.CDLL("libamdhip64.so")
        blocks = ctypes.c_int()
        # Triton returns a module function handle, not a runtime host-stub pointer.
        rc = hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
            ctypes.byref(blocks),
            ctypes.c_void_p(kernel.function),
            ctypes.c_int(kernel.metadata.num_warps * 64),
            ctypes.c_size_t(0),
        )
        last_error = hip.hipGetLastError()
        resources.update(
            occupancy_query_rc=rc,
            occupancy_last_error=last_error,
            max_blocks_per_cu=blocks.value if rc == 0 else None,
        )
    except (OSError, TypeError, AttributeError) as error:
        resources["occupancy_unavailable"] = repr(error)
    return resources


def benchmark(output):
    import torch
    from triton.compiler.errors import CompilationError
    from triton.runtime.errors import OutOfResources
    from vllm.model_executor.kernels.linear.mxfp8 import rocm_native as native

    if torch.cuda.device_count() != 1:
        raise RuntimeError("Exactly one lease-masked GPU must be visible")
    torch.cuda.set_device(0)
    torch.set_float32_matmul_precision("highest")
    device = torch.cuda.get_device_properties(0)
    if "gfx950" not in device.gcnArchName:
        raise RuntimeError("This diagnostic requires the observed gfx950 architecture")
    output.mkdir(parents=True, exist_ok=False)
    source = Path(inspect.getfile(native))
    result = {
        "plan": PLAN,
        "native_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "gpu": str(device),
        "records": [],
    }
    (output / "rocm_native.py").write_bytes(source.read_bytes())
    started = time.monotonic()
    n, k = PLAN["N"], PLAN["K"]
    torch.manual_seed(20261125)
    weights = []
    for _ in range(PLAN["rotating_weights"]):
        w = (torch.randn((n, k), device="cuda") * 0.25).to(torch.float8_e4m3fn)
        scale = torch.randint(120, 126, (n, k // 32), device="cuda", dtype=torch.uint8)
        weights.append((w, scale))

    def save():
        (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    def timed(call):
        # Distinct weight allocations reduce resident-weight bias; setup is untimed.
        for _ in range(2):
            for index in range(len(weights)):
                call(index)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for index in range(len(weights)):
                call(index)
        passes = []
        for _ in range(2):
            for _ in range(PLAN.get("warmup_graph_replays", 5)):
                graph.replay()
            events = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(24)
            ]
            for begin, end in events:
                begin.record()
                graph.replay()
                end.record()
            torch.cuda.synchronize()
            passes.append(
                statistics.median(
                    a.elapsed_time(b) * 1000 / len(weights) for a, b in events
                )
            )
        return {
            "median_us_passes": passes,
            "repeat_delta_percent": abs(passes[0] - passes[1])
            / statistics.mean(passes)
            * 100,
        }

    # Compiled handles come from the real Triton launch, outside timed graph replay.
    original_run = native._mxfp8_linear_kernel.run
    handles = []

    def capture_handle(*args, **kwargs):
        kernel = original_run(*args, **kwargs)
        if kernel is not None:
            handles.append(kernel)
        return kernel

    native._mxfp8_linear_kernel.run = capture_handle
    try:
        for m in PLAN["M"]:
            x = (torch.randn((m, k), device="cuda") * 0.25).to(torch.float8_e4m3fn)
            sx = torch.randint(120, 126, (m, k // 32), device="cuda", dtype=torch.uint8)
            out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
            w, sw = weights[0]
            # Independent FP32 dequantization/matmul, rounded to the served dtype.
            dx = x.float() * torch.exp2(sx.float() - 127).repeat_interleave(32, 1)
            dw = w.float() * torch.exp2(sw.float() - 127).repeat_interleave(32, 1)
            reference = (dx @ dw.T).to(torch.bfloat16)
            del dx, dw
            baseline = list(native._select_cfg(m, n, k))
            configs = [baseline] + [c for c in PLAN["configs"] if c != baseline]
            if PLAN.get("confirmation_abba"):
                configs += list(reversed(configs))
            for config in configs:
                if time.monotonic() - started > 900:
                    raise TimeoutError("15-minute operator budget reached")
                record = {"m": m, "config": config, "baseline": config == baseline}
                handles.clear()
                try:

                    def call(index):
                        wi, si = weights[index]
                        native._launch(x, sx, wi, si, out, *config)

                    call(0)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(out, reference, rtol=0.02, atol=0.001)
                    rms = (
                        (
                            (out.float() - reference.float()).square().mean()
                            / reference.float().square().mean()
                        )
                        .sqrt()
                        .item()
                    )
                    if rms > 0.004:
                        raise AssertionError(f"relative RMS error {rms} exceeds0.004")
                    record.update(
                        correct=True, relative_rms_error=rms, timing=timed(call)
                    )
                    record["resources"] = kernel_resources(handles[-1])
                except (
                    AssertionError,
                    ValueError,
                    OutOfResources,
                    CompilationError,
                ) as error:
                    record.update(correct=False, error=repr(error))
                    if record["baseline"]:
                        result["records"].append(record)
                        save()
                        raise
                result["records"].append(record)
                save()
                print(json.dumps(record), flush=True)
    finally:
        native._mxfp8_linear_kernel.run = original_run
    result["complete"] = True
    save()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.confirm:
        PLAN.update(
            M=[16, 24, 32, 40, 48, 56, 64, 80],
            configs=[[64, 64, 512, 8, 2]],
            confirmation_abba=True,
            warmup_graph_replays=1000,
        )
    validate_plan()
    if args.plan:
        print(json.dumps(PLAN, indent=2))
    elif args.output is None:
        parser.error("--output is required for GPU execution")
    else:
        benchmark(args.output)
