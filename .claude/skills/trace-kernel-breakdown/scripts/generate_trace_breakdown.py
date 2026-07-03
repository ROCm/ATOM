#!/usr/bin/env python3
"""Generate SGLang/ATOM kernel breakdown CSVs from Chrome trace JSON(.gz).

This script is intentionally stdlib-only. It encodes a framework-aware workflow
for SGLang and ATOM PyTorch/Chrome traces:

- CPU events provide shape/module hints.
- GPU kernel events are the source of final kernel rows and durations.
- ATOM traces usually have prefill/decode annotations.
- SGLang traces may need layer inference from shape and kernel sequence.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


SHAPE_KEYS = {"Input Dims", "Input Strides", "Input type", "Concrete Inputs"}


def load_trace(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    return [e for e in data.get("traceEvents", []) if isinstance(e, dict)]


def ts(e: dict[str, Any]) -> float:
    return float(e.get("ts", 0) or 0)


def dur(e: dict[str, Any]) -> float:
    return float(e.get("dur", 0) or 0)


def name(e: dict[str, Any]) -> str:
    return str(e.get("name", ""))


def args(e: dict[str, Any]) -> dict[str, Any]:
    return e.get("args") or {}


def ext_id(e: dict[str, Any]) -> Any:
    return args(e).get("External id")


def clean_kernel_name(kernel_name: str) -> str:
    n = kernel_name
    if n.startswith("void "):
        n = n[5:]
    if "(" in n and not n.startswith("_Z"):
        n = re.sub(r"\([^)]*\)$", "", n)

    low = n.lower()
    if "vectorized_layer_norm_kernel" in n:
        return "native_layer_norm"
    if "indexselectsmallindex" in low or "vectorized_gather_kernel" in n:
        return "index_select"
    if "catarraybatchedcopy" in low:
        return "cat_copy"
    if "bfloat16tofloat32_copy_kernel" in n:
        return "copy_bf16_to_fp32"
    if "bfloat16_copy_kernel" in n:
        return "copy_fp32_to_bf16"
    if "float8_copy_kernel" in n:
        return "copy_fp8"
    if "direct_copy_kernel" in n:
        return "copy_"
    if "FillFunctor" in n:
        return "fill_"
    if "CUDAFunctor_add" in n:
        return "add"
    if "MulFunctor" in n:
        return "mul"
    if "BinaryFunctor" in n:
        return "binary_elementwise"
    if "AUnaryFunctor" in n:
        return "unary_elementwise"
    if "elementwise_kernel_manual_unroll" in n or "vectorized_elementwise_kernel" in n:
        return "elementwise"
    return n


def valid_dim(d: Any) -> bool:
    return isinstance(d, list) and bool(d) and all(isinstance(x, int) for x in d)


def valid_dims(e: dict[str, Any]) -> list[list[int]]:
    return [d for d in args(e).get("Input Dims", []) if valid_dim(d)]


def first_m(e: dict[str, Any]) -> int | None:
    dims = valid_dims(e)
    return dims[0][0] if dims else None


def infer_mnk(cpu_name: str, kernel_name: str, dims: list[list[int]]) -> tuple[str, str, str]:
    dims = [d for d in dims if valid_dim(d)]
    if not dims:
        return "", "", ""
    low = f"{cpu_name} {kernel_name}".lower()

    if any(x in low for x in ["gemm", "cijk", "batched_gemm"]) and len(dims) >= 2:
        a, b = dims[0], dims[1]
        if len(a) >= 2 and len(b) >= 2:
            m = a[-2]
            k = a[-1]
            n: int | str = ""
            for d in dims[2:]:
                if len(d) >= 2 and d[-2] == m and d[-1] != k:
                    n = d[-1]
                    break
            if n == "":
                n = b[-2] if b[-1] == k else b[-1]
            return str(m), str(n), str(k)

    if any(x in low for x in ["fmoe", "fused_moe", "moe_gemm", "kernel_moe_gemm"]):
        first = dims[0]
        m = first[0] if first else ""
        n: int | str = ""
        k: int | str = ""
        for d in dims:
            if len(d) == 3:
                n, k = d[1], d[2]
                break
        if n == "" and len(first) >= 2:
            n = first[1]
        return str(m), str(n), str(k)

    first = dims[0]
    if len(first) >= 2:
        return str(first[-2]), str(first[-1]), ""
    return str(first[0]), "", ""


def build_indices(trace: list[dict[str, Any]]):
    kernels = sorted(
        [e for e in trace if e.get("ph") == "X" and e.get("cat") == "kernel"],
        key=ts,
    )
    cpu_shape = []
    cpu_by_ext = {}
    kernels_by_ext: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    annotations = []

    for e in trace:
        if e.get("ph") != "X":
            continue
        eid = ext_id(e)
        cat = e.get("cat")
        n = name(e)
        if cat == "kernel" and eid is not None:
            kernels_by_ext[eid].append(e)
        elif cat in ("cpu_op", "user_annotation") and eid is not None and SHAPE_KEYS & set(args(e)):
            cpu_shape.append(e)
            cpu_by_ext[eid] = e
        if cat == "user_annotation" and (n.startswith("prefill[") or n.startswith("decode[")):
            annotations.append(e)

    for v in kernels_by_ext.values():
        v.sort(key=ts)
    annotations.sort(key=ts)
    return kernels, cpu_shape, cpu_by_ext, kernels_by_ext, annotations


def detect_framework(path: Path, annotations: list[dict[str, Any]]) -> str:
    p = str(path).lower()
    if annotations or "/trace/rank_" in p or "atom" in p:
        return "atom"
    return "sglang"


def is_interesting_cpu(e: dict[str, Any]) -> bool:
    low = name(e).lower()
    terms = [
        "aiter",
        "gemm",
        "ck",
        "triton",
        "moe",
        "norm",
        "quant",
        "attention",
        "mha",
        "all_reduce",
        "scatter",
        "catarray",
        "cijk",
        "nccl",
        "sglang",
        "c10d",
        "hadamard",
        "sgl_kernel",
        "mla",
    ]
    return any(t in low for t in terms)


def module_for_kernel(phase: str, kernel_name: str, row_index: int | None = None) -> str:
    k = kernel_name.lower()
    if any(x in k for x in ["nccl", "all_reduce", "cross_device_reduce", "reduce_scatter"]):
        return "all_reduce"
    if any(x in k for x in ["grouped_topk", "moesorting", "fmoe", "fused_moe", "kernel_moe_gemm", "shared_experts"]):
        if "dynamic_per_group_scaled_quant" in k:
            return "moe_quant"
        if "fmoe" in k or "kernel_moe_gemm" in k:
            return "moe_expert"
        if "topk" in k:
            return "moe_topk"
        if "sort" in k:
            return "moe_sort"
        return "moe"
    if "local_device_load_rmsnorm" in k or "add_rmsnorm_quant" in k:
        return "post_attn_norm" if phase == "decode" and row_index and row_index > 25 else "attn_input_norm"
    if "fused_qk_rmsnorm" in k:
        return "attn_qk_norm_quant"
    if "layernorm" in k or "native_layer_norm" in k:
        return "attn_rope_norm"
    if "paged_mqa" in k or "mla_a8w8" in k or "kn_mla_reduce" in k or "main_kernel" == k:
        return "mla_attn" if phase == "decode" else "fmha_prefill"
    if "fuse_qk_rope" in k:
        return "attn_rope_cache_update"
    if "indexer" in k or "cache_mla" in k or "set_mla" in k or "cat_copy" in k:
        return "attn_kv_cache_update"
    if "hadamard" in k or "topk_transform" in k:
        return "attn_rope"
    if "gemm" in k or "cijk" in k:
        return "attn_gemm"
    if "quant" in k:
        return "quant"
    if k in ("mul", "add", "copy_", "copy_bf16_to_fp32", "copy_fp32_to_bf16", "index_select", "fill_"):
        return "attn_rope"
    return "misc"


def nearest_nccl_after(kernels: list[dict[str, Any]], t: float, used: set[int], max_delay_us: float = 2500) -> dict[str, Any] | None:
    cand = [
        k
        for k in kernels
        if id(k) not in used and "ncclDevKernel" in name(k) and 0 <= ts(k) - t <= max_delay_us
    ]
    if not cand:
        return None
    k = min(cand, key=lambda x: ts(x) - t)
    used.add(id(k))
    return k


def rows_from_gpu_window(
    phase: str,
    gpu_window: list[dict[str, Any]],
    cpu_by_ext: dict[Any, dict[str, Any]],
    comm_shapes: dict[int, dict[str, Any]] | None = None,
) -> list[list[str]]:
    rows = []
    comm_shapes = comm_shapes or {}
    for idx, k in enumerate(sorted(gpu_window, key=ts), 1):
        kname = clean_kernel_name(name(k))
        cpu = cpu_by_ext.get(ext_id(k)) or comm_shapes.get(id(k))
        if cpu:
            m, n, kk = infer_mnk(name(cpu), name(k), valid_dims(cpu))
        else:
            m = n = kk = ""
        rows.append([phase, module_for_kernel(phase, kname, idx), kname, f"{dur(k):.3f}", m, n, kk])
    return rows


def atom_breakdown(trace: list[dict[str, Any]], target_prefill_m: int) -> list[list[str]]:
    kernels, cpu_shape, cpu_by_ext, kernels_by_ext, annotations = build_indices(trace)
    out: list[list[str]] = []

    def window(ann: dict[str, Any]) -> list[dict[str, Any]]:
        s = ts(ann)
        e = s + dur(ann)
        return [k for k in kernels if s <= ts(k) <= e]

    # Decode: first full decode annotation with local norm -> fmoe -> reduce.
    decode_layer = None
    for ann in annotations:
        if not name(ann).startswith("decode["):
            continue
        dw = window(ann)
        for li, k in enumerate(dw):
            if "local_device_load_rmsnorm" not in name(k):
                continue
            fmoe = next((i for i in range(li, len(dw)) if "fmoe" in name(dw[i]).lower()), None)
            if fmoe is None:
                continue
            end = next((i for i in range(fmoe, len(dw)) if "reduce_scatter_cross_device_store" in name(dw[i])), None)
            if end is not None:
                seg = dw[li : end + 1]
                if len(seg) >= 20:
                    decode_layer = seg
                    break
        if decode_layer:
            break

    # Prefill: annotation closest to target M, full layer as GPU window.
    prefill_anns = []
    for ann in annotations:
        if name(ann).startswith("prefill["):
            m = re.search(r"tok=(\d+)", name(ann))
            if m:
                prefill_anns.append((abs(int(m.group(1)) - target_prefill_m), ann))
    prefill_layer = []
    if prefill_anns:
        ann = min(prefill_anns, key=lambda x: x[0])[1]
        pw = window(ann)
        # Keep one full attention+MoE layer when possible.
        starts = [i for i, k in enumerate(pw) if "add_rmsnorm_quant" in name(k) or "local_device_load_rmsnorm" in name(k)]
        for a, b, c in zip(starts, starts[1:], starts[2:]):
            seg = pw[a:c]
            if any("fmoe" in name(k).lower() for k in seg):
                prefill_layer = seg
                break
        if not prefill_layer:
            prefill_layer = pw

    if prefill_layer:
        out.extend(rows_from_gpu_window("prefill", prefill_layer, cpu_by_ext))
    if decode_layer:
        out.extend(rows_from_gpu_window("decode", decode_layer, cpu_by_ext))
    return out


def sglang_breakdown(trace: list[dict[str, Any]], target_prefill_m: int) -> list[list[str]]:
    kernels, cpu_shape, cpu_by_ext, kernels_by_ext, annotations = build_indices(trace)
    pops = sorted([e for e in cpu_shape if is_interesting_cpu(e)], key=ts)
    nccls = [k for k in kernels if "ncclDevKernel" in name(k)]

    # Pick prefill M closest to target, then select CPU layer with attn+MoE+comm.
    candidates = sorted(
        {first_m(e) for e in pops if isinstance(first_m(e), int) and (first_m(e) or 0) > 128},
        key=lambda m: abs((m or 0) - target_prefill_m),
    )
    target_m = candidates[0] if candidates else target_prefill_m
    norms = [i for i, e in enumerate(pops) if name(e) == "aiter::add_rmsnorm" and first_m(e) == target_m]

    def has(seg: list[dict[str, Any]], kind: str) -> bool:
        for e in seg:
            low = name(e).lower()
            if kind == "attn" and any(x in low for x in ["hadamard", "fast_topk_transform", "mla", "attention", "mha", "topk_transform_prefill"]):
                return True
            if kind == "moe" and any(x in low for x in ["fmoe", "fused_moe", "moe_sorting", "biased_grouped_topk", "grouped_topk"]):
                return True
            if kind == "comm" and any(x in low for x in ["all_reduce", "c10d", "sglang::inplace", "sglang::outplace", "nccl"]):
                return True
        return False

    selected_cpu = None
    for a, b, c in zip(norms, norms[1:], norms[2:]):
        seg1 = pops[a:b]
        seg2 = pops[b:c]
        if has(seg1, "attn") and has(seg1, "comm") and has(seg2, "moe") and has(seg2, "comm"):
            selected_cpu = pops[a:c]
            break

    out: list[list[str]] = []
    if selected_cpu:
        mapped = []
        comm_shapes = {}
        used_nccl: set[int] = set()
        for cpu in selected_cpu:
            mapped.extend(kernels_by_ext.get(ext_id(cpu), []))
            if any(x in name(cpu).lower() for x in ["all_reduce", "nccl:", "c10d", "sglang::inplace", "sglang::outplace"]):
                k = nearest_nccl_after(kernels, ts(cpu), used_nccl)
                if k:
                    mapped.append(k)
                    comm_shapes[id(k)] = cpu
        if mapped:
            start = min(ts(k) for k in mapped)
            end = max(ts(k) + dur(k) for k in mapped)
            gpu_window = [k for k in kernels if start <= ts(k) <= end]
            out.extend(rows_from_gpu_window("prefill", gpu_window, cpu_by_ext, comm_shapes))

    # Decode: infer from GPU pattern.
    lows = [name(k).lower() for k in kernels]
    decode_layer = None
    for i, n in enumerate(lows):
        if "add_rmsnorm_quant_kernel" not in n:
            continue
        j_attn = next((j for j in range(i + 1, min(i + 90, len(kernels))) if "paged_mqa" in lows[j] or "topk_transform_decode" in lows[j]), None)
        if j_attn is None:
            continue
        j_comm1 = next((j for j in range(j_attn + 1, min(i + 130, len(kernels))) if "cross_device_reduce" in lows[j] or "nccl" in lows[j]), None)
        if j_comm1 is None:
            continue
        j_norm2 = next((j for j in range(j_comm1 + 1, min(i + 160, len(kernels))) if "add_rmsnorm_quant_kernel" in lows[j]), None)
        if j_norm2 is None:
            continue
        j_moe = next((j for j in range(j_norm2 + 1, min(i + 240, len(kernels))) if any(x in lows[j] for x in ["grouped_topk", "moesorting", "kernel_moe_gemm", "fmoe"])), None)
        if j_moe is None:
            continue
        j_comm2 = next((j for j in range(j_moe + 1, min(i + 280, len(kernels))) if "cross_device_reduce" in lows[j] or "nccl" in lows[j]), None)
        if j_comm2 is not None:
            decode_layer = kernels[i : j_comm2 + 1]
            break
    if decode_layer:
        out.extend(rows_from_gpu_window("decode", decode_layer, cpu_by_ext))
    return out


def write_breakdown(rows: list[list[str]], out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phase", "module", "kernel_name", "kernel_time_us", "m", "n", "k"])
        w.writerows(rows)


def aggregate(csv_path: Path):
    totals = defaultdict(float)
    phase_totals = defaultdict(float)
    with csv_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = float(row["kernel_time_us"] or 0)
            phase = row["phase"]
            module = row["module"]
            phase_totals[phase] += t
            totals[(phase, module)] += t
    return phase_totals, totals


def compare(baseline_csv: Path, candidate_csv: Path, out_md: Path):
    baseline_phase, baseline_mod = aggregate(baseline_csv)
    candidate_phase, candidate_mod = aggregate(candidate_csv)
    lines = ["# Trace breakdown comparison", ""]
    for phase in sorted(set(baseline_phase) | set(candidate_phase)):
        baseline_total = baseline_phase.get(phase, 0)
        candidate_total = candidate_phase.get(phase, 0)
        saved_total = baseline_total - candidate_total
        speedup = baseline_total / candidate_total if candidate_total else 0
        lines += [
            f"## {phase}",
            "",
            f"- Baseline total: {baseline_total:.3f} us",
            f"- Candidate total: {candidate_total:.3f} us",
            f"- Saved: {saved_total:.3f} us",
            f"- Speedup: {speedup:.2f}x",
            "",
            "| Module | Baseline us | Candidate us | Saved us | Share of phase saving |",
            "|---|---:|---:|---:|---:|",
        ]
        mods = {m for p, m in baseline_mod if p == phase} | {m for p, m in candidate_mod if p == phase}
        for m in sorted(mods, key=lambda x: baseline_mod.get((phase, x), 0) - candidate_mod.get((phase, x), 0), reverse=True):
            baseline = baseline_mod.get((phase, m), 0)
            candidate = candidate_mod.get((phase, m), 0)
            saved = baseline - candidate
            share = saved / saved_total if saved_total else 0
            lines.append(f"| {m} | {baseline:.3f} | {candidate:.3f} | {saved:.3f} | {share:.1%} |")
        lines.append("")
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--framework", choices=["auto", "atom", "sglang"], default="auto")
    ap.add_argument("--target-prefill-m", type=int, default=7000)
    ap.add_argument("--compare-baseline", type=Path)
    ap.add_argument("--compare-candidate", type=Path)
    # Backward-compatible aliases for the first GLM analysis. Prefer baseline/candidate.
    ap.add_argument("--compare-old", type=Path)
    ap.add_argument("--compare-new", type=Path)
    ap.add_argument("--compare-out", type=Path)
    ns = ap.parse_args()

    baseline = ns.compare_baseline or ns.compare_old
    candidate = ns.compare_candidate or ns.compare_new
    if baseline and candidate and ns.compare_out:
        compare(baseline, candidate, ns.compare_out)
        return

    if not ns.trace or not ns.out:
        raise SystemExit("--trace and --out are required unless using --compare-*")
    trace = load_trace(ns.trace)
    _, _, _, _, annotations = build_indices(trace)
    framework = detect_framework(ns.trace, annotations) if ns.framework == "auto" else ns.framework
    rows = atom_breakdown(trace, ns.target_prefill_m) if framework == "atom" else sglang_breakdown(trace, ns.target_prefill_m)
    write_breakdown(rows, ns.out)
    print(f"framework={framework} rows={len(rows)} out={ns.out}")
    print(dict(Counter(r[0] for r in rows)))


if __name__ == "__main__":
    main()
