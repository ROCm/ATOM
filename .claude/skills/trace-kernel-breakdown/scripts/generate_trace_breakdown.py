#!/usr/bin/env python3
"""Generate SGLang/ATOM kernel breakdown CSVs from Chrome trace JSON(.gz).

This script is intentionally stdlib-only. It encodes the workflow learned from
GLM-5 FP8 SGLang old and ATOM new traces:

- CPU events provide shape/module hints.
- GPU kernel events are the source of final kernel rows and durations.
- ATOM traces usually have prefill/decode annotations.
- SGLang old traces may need layer inference from shape and kernel sequence.
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
BREAKDOWN_HEADER = [
    "phase",
    "variant",
    "module",
    "kernel_name",
    "kernel_time_us",
    "m",
    "n",
    "k",
    "tflops",
    "bandwidth_gb_s",
    "metric_note",
]
SUMMARY_HEADER = [
    "row_type",
    "phase",
    "variant",
    "variant_count",
    "module_group",
    "module",
    "kernel_name",
    "kernel_time_us",
    "ratio",
    "tflops",
    "bandwidth_gb_s",
    "m",
    "n",
    "k",
    "weighted_time_ms",
    "metric_note",
]


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


def is_allreduce_kernel(kernel_name: str) -> bool:
    k = kernel_name.lower()
    return any(x in k for x in ["nccl", "allreduce", "all_reduce", "cross_device_reduce", "reduce_scatter"])


def is_gemm_kernel(kernel_name: str) -> bool:
    k = kernel_name.lower()
    return any(x in k for x in ["gemm", "cijk", "batched_gemm", "mfma_moe"])


def is_moe_kernel(kernel_name: str) -> bool:
    k = kernel_name.lower()
    return any(x in k for x in ["grouped_topk", "moesorting", "mfma_moe", "fmoe", "moe_sort", "mxfp4_moe_sort", "dynamic_per_group_scaled_quant"])


def is_index_logits_or_topk(kernel_name: str) -> bool:
    k = kernel_name.lower()
    return any(x in k for x in ["mqa_logits", "paged_mqa_logits", "radix_topk", "top_k_per_row"])


def is_mla_kernel(kernel_name: str) -> bool:
    k = kernel_name.lower()
    return "mla_a8w8" in k or "kn_mla_reduce" in k or "paged_mqa" in k or "mqa_logits" in k


def parse_int(s: str) -> int | None:
    if not s:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def dtype_bytes(kernel_name: str, role: str) -> float:
    k = kernel_name.lower()
    if "fp4" in k or "mxfp4" in k or "afp4" in k or "wfp4" in k:
        if role == "weight":
            return 0.5
        return 2.0 if "bf16" in k else 1.0
    if "fp8" in k or "a8w8" in k:
        return 1.0
    if "bf16" in k or "bfloat16" in k or "hgemm" in k or "cijk" in k:
        return 2.0
    if "float" in k:
        return 4.0
    return 2.0


def estimate_metrics(module: str, kernel_name: str, kernel_time_us: str, m: str, n: str, k: str) -> tuple[str, str, str]:
    mm = parse_int(m)
    nn = parse_int(n)
    kk = parse_int(k)
    try:
        t_us = float(kernel_time_us or 0)
    except ValueError:
        t_us = 0.0
    if t_us <= 0:
        return "", "", "missing duration"

    notes = []
    tflops = ""
    bandwidth = ""
    low = f"{module} {kernel_name}".lower()
    if mm and nn and kk and (is_gemm_kernel(kernel_name) or "gemm" in low or "moe_expert" in low):
        flops = 2.0 * mm * nn * kk
        tflops = f"{flops / t_us / 1_000_000:.3f}"
        a_bytes = mm * kk * dtype_bytes(kernel_name, "activation")
        b_bytes = kk * nn * dtype_bytes(kernel_name, "weight")
        c_bytes = mm * nn * dtype_bytes(kernel_name, "output")
        bandwidth = f"{(a_bytes + b_bytes + c_bytes) / t_us / 1_000:.3f}"
        notes.append("gemm: flops=2*m*n*k; bytes=A+B+C estimate")
    elif mm and nn and (any(x in low for x in ["attn", "mla", "moe", "topk", "sort", "quant"])):
        elem_bytes = dtype_bytes(kernel_name, "activation")
        # Conservative traffic estimate for non-GEMM attention/MoE kernels when
        # only a 2D tensor shape is known: one read and one write.
        bandwidth = f"{(2.0 * mm * nn * elem_bytes) / t_us / 1_000:.3f}"
        notes.append("bandwidth estimate from 2D tensor read+write")
    elif any(x in low for x in ["attn", "mla", "moe", "gemm"]):
        notes.append("shape unavailable; metric left blank")
    return tflops, bandwidth, "; ".join(notes)


def make_row(
    phase: str,
    variant: str,
    module: str,
    kernel_name: str,
    kernel_time_us: str,
    m: str,
    n: str,
    k: str,
) -> list[str]:
    tflops, bandwidth, note = estimate_metrics(module, kernel_name, kernel_time_us, m, n, k)
    return [phase, variant, module, kernel_name, kernel_time_us, m, n, k, tflops, bandwidth, note]


def pad_row(row: list[str]) -> list[str]:
    if len(row) == len(BREAKDOWN_HEADER):
        return row
    if len(row) == 7:
        phase, module, kernel_name, kernel_time_us, m, n, k = row
        return make_row(phase, "", module, kernel_name, kernel_time_us, m, n, k)
    if len(row) == 8:
        phase, variant, module, kernel_name, kernel_time_us, m, n, k = row
        return make_row(phase, variant, module, kernel_name, kernel_time_us, m, n, k)
    return (row + [""] * len(BREAKDOWN_HEADER))[: len(BREAKDOWN_HEADER)]


def module_group(module: str, kernel_name: str) -> str:
    low = f"{module} {kernel_name}".lower()
    if "all_reduce" in low or "allreduce" in low or "nccl" in low:
        return "all_reduce"
    if "moe" in low or "fmoe" in low or "mfma_moe" in low:
        return "moe"
    if "norm" in low:
        return "norm"
    if "gemm" in low or "cijk" in low or "batched_gemm" in low:
        return "gemm"
    if "attn" in low or "mla" in low or "mqa" in low:
        return "attn"
    if "quant" in low:
        return "quant"
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
    variant: str = "",
    module_override: Any | None = None,
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
        module = module_override(phase, variant, gpu_window, idx - 1, kname) if module_override else module_for_kernel(phase, kname, idx)
        rows.append(make_row(phase, variant, module, kname, f"{dur(k):.3f}", m, n, kk))
    return rows


def atom_annotation_window(ann: dict[str, Any], kernels: list[dict[str, Any]]) -> list[dict[str, Any]]:
    s = ts(ann)
    e = s + dur(ann)
    return [k for k in kernels if s <= ts(k) <= e]


def select_median_duration(candidates: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if not candidates:
        return []
    return sorted(candidates, key=lambda seg: sum(dur(k) for k in seg))[len(candidates) // 2]


def select_first_duration(candidates: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return candidates[0] if candidates else []


def atom_prefill_variant_segments(
    kernels: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    target_prefill_m: int,
) -> dict[str, list[dict[str, Any]]]:
    prefill_anns = []
    for ann in annotations:
        if name(ann).startswith("prefill["):
            m = re.search(r"tok=(\d+)", name(ann))
            if m:
                prefill_anns.append((abs(int(m.group(1)) - target_prefill_m), ann))
    if not prefill_anns:
        return {}
    ann = min(prefill_anns, key=lambda x: x[0])[1]
    win = atom_annotation_window(ann, kernels)

    def is_input_norm(k: dict[str, Any]) -> bool:
        n = name(k)
        return "add_rmsnorm_quant_kernel" in n and "<std::bfloat16_t, std::bfloat16_t, 256, 24" in n

    candidates: list[list[dict[str, Any]]] = []
    for start in [i for i, k in enumerate(win) if is_input_norm(k)]:
        for end in range(start + 1, min(start + 120, len(win))):
            seg = win[start : end + 1]
            if not is_allreduce_kernel(name(win[end])):
                continue
            if any(is_mla_kernel(name(k)) for k in seg) and any(is_moe_kernel(name(k)) for k in seg):
                first_mla = next(i for i, k in enumerate(seg) if is_mla_kernel(name(k)))
                first_moe = next(i for i, k in enumerate(seg) if is_moe_kernel(name(k)))
                if first_mla > first_moe:
                    break
                # GLM IndexShare steady-state layers are compact: shared layers
                # have no indexer score/topk kernels, index layers add ~8 kernels.
                if len(seg) <= 45:
                    candidates.append(seg)
                break

    index = [seg for seg in candidates if any(is_index_logits_or_topk(name(k)) for k in seg)]
    shared = [seg for seg in candidates if not any(is_index_logits_or_topk(name(k)) for k in seg)]
    out = {}
    if index:
        out["index"] = select_median_duration(index)
    if shared:
        out["shared_index"] = select_median_duration(shared)
    return out


def atom_decode_variant_segments(
    kernels: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
    target_decode_tokens: int | None,
) -> dict[str, list[dict[str, Any]]]:
    decode_anns = []
    for ann in annotations:
        if not name(ann).startswith("decode["):
            continue
        if target_decode_tokens is not None and f"tok={target_decode_tokens}" not in name(ann):
            continue
        decode_anns.append(ann)
    if not decode_anns and target_decode_tokens is not None:
        decode_anns = [ann for ann in annotations if name(ann).startswith("decode[")]

    def is_decode_layer_start(k: dict[str, Any], nxt: dict[str, Any] | None) -> bool:
        return nxt is not None and is_gemm_kernel(name(k)) and "fused_qk_rmsnorm" in name(nxt).lower()

    candidates: list[list[dict[str, Any]]] = []
    for ann in decode_anns:
        win = atom_annotation_window(ann, kernels)
        for start in range(0, max(0, len(win) - 18)):
            if not is_decode_layer_start(win[start], win[start + 1] if start + 1 < len(win) else None):
                continue
            attn_comm = next((j for j in range(start + 5, min(start + 28, len(win))) if is_allreduce_kernel(name(win[j]))), None)
            if attn_comm is None:
                continue
            final_comm = next(
                (
                    j
                    for j in range(attn_comm + 3, min(attn_comm + 28, len(win)))
                    if is_allreduce_kernel(name(win[j])) and any(is_moe_kernel(name(x)) for x in win[attn_comm:j])
                ),
                None,
            )
            if final_comm is None:
                continue
            seg = win[start : final_comm + 1]
            if any(is_mla_kernel(name(k)) for k in seg):
                candidates.append(seg)

    dedup: list[list[dict[str, Any]]] = []
    for seg in sorted(candidates, key=lambda s: ts(s[0])):
        if not dedup or abs(ts(seg[0]) - ts(dedup[-1][0])) > 1:
            dedup.append(seg)

    index = [seg for seg in dedup if any(is_index_logits_or_topk(name(k)) for k in seg)]
    shared = [seg for seg in dedup if not any(is_index_logits_or_topk(name(k)) for k in seg)]
    out = {}
    if index:
        out["index"] = select_first_duration(index)
    if shared:
        out["shared_index"] = select_first_duration(shared)
    return out


def atom_variant_module(phase: str, variant: str, seg: list[dict[str, Any]], idx: int, kname: str) -> str:
    k = kname.lower()
    if is_allreduce_kernel(kname):
        return "all_reduce"
    if "add_rmsnorm_quant" in k and "256, 24" in k:
        # First norm before attention, second norm before MoE.
        seen_ar = any(is_allreduce_kernel(name(x)) for x in seg[:idx])
        return "post_attn_norm" if seen_ar else "attn_input_norm"
    if "fused_qk_rmsnorm" in k:
        return "attn_qk_norm_quant"
    if "indexer_qk_rope_quant" in k or "cp_gather_indexer" in k:
        return "sparse_attn_indexer"
    if "mqa_logits" in k or "paged_mqa_logits" in k:
        return "mla_logits"
    if "radix_topk" in k:
        return "mla_topk"
    if "convert_req_index" in k:
        return "mla_index_convert"
    if "fuse_qk_rope" in k:
        return "attn_rope_cache_update"
    if "mla_a8w8" in k or "kn_mla_reduce" in k:
        return "attn"
    if "grouped_topk" in k or "moesorting" in k:
        return "moe"
    if "dynamic_per_group_scaled_quant" in k or "mxfp4_moe_sort" in k or "fused_mx_quant_moe_sort" in k:
        return "moe"
    if "mfma_moe" in k:
        return "moe"
    if is_gemm_kernel(kname):
        seen_first_ar = any(is_allreduce_kernel(name(x)) for x in seg[:idx])
        if seen_first_ar:
            return "gate_gemm" if not any(is_moe_kernel(name(x)) for x in seg[:idx]) else "moe"
        if idx > 0 and any(is_mla_kernel(name(x)) for x in seg[:idx]):
            return "o_gemm"
        return "attn_gemm"
    return module_for_kernel(phase, kname, idx + 1)


def atom_breakdown(trace: list[dict[str, Any]], target_prefill_m: int, target_decode_tokens: int | None = None) -> list[list[str]]:
    kernels, cpu_shape, cpu_by_ext, kernels_by_ext, annotations = build_indices(trace)
    out: list[list[str]] = []

    def window(ann: dict[str, Any]) -> list[dict[str, Any]]:
        s = ts(ann)
        e = s + dur(ann)
        return [k for k in kernels if s <= ts(k) <= e]

    # Decode fallback: first full decode annotation with local norm -> fmoe -> reduce.
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

    # Prefer IndexShare-aware variants when present. GLM-5.2 has "index" layers
    # that compute sparse top-k and "shared_index" layers that reuse it. The
    # same distinction exists in prefill and decode, with different logits kernels.
    prefill_variants = atom_prefill_variant_segments(kernels, annotations, target_prefill_m)
    decode_variants = atom_decode_variant_segments(kernels, annotations, target_decode_tokens)
    if prefill_variants or decode_variants:
        for variant in ("shared_index", "index"):
            if variant in prefill_variants:
                out.extend(rows_from_gpu_window("prefill", prefill_variants[variant], cpu_by_ext, variant=variant, module_override=atom_variant_module))
        for variant in ("shared_index", "index"):
            if variant in decode_variants:
                out.extend(rows_from_gpu_window("decode", decode_variants[variant], cpu_by_ext, variant=variant, module_override=atom_variant_module))
        return out

    # Prefill fallback: annotation closest to target M, full layer as GPU window.
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
        w.writerow(BREAKDOWN_HEADER)
        w.writerows([pad_row(r) for r in rows])


def load_variant_counts(model_config: Path | None) -> dict[str, int]:
    if not model_config:
        return {}
    with model_config.open(encoding="utf-8") as f:
        cfg = json.load(f)
    indexer_types = cfg.get("indexer_types")
    if isinstance(indexer_types, list):
        counts = Counter(str(x) for x in indexer_types)
        out = {
            "index": counts.get("full", 0) + counts.get("index", 0),
            "shared_index": counts.get("shared", 0) + counts.get("shared_index", 0),
        }
        return {k: v for k, v in out.items() if v}
    freq = int(cfg.get("index_topk_freq", 1) or 1)
    layers = int(cfg.get("num_hidden_layers", 0) or 0)
    if freq > 1 and layers:
        # GLM-style fallback: first layers may be explicit in real configs, so
        # prefer indexer_types when available. This is only an approximation.
        index = (layers + freq - 1) // freq
        return {"index": index, "shared_index": layers - index}
    return {}


def variant_count_for(variant: str, counts: dict[str, int]) -> int:
    if variant in counts:
        return counts[variant]
    if variant == "full_indexer":
        return counts.get("index", 1)
    if variant == "shared_indexer":
        return counts.get("shared_index", 1)
    return 1


def write_summary(rows: list[list[str]], out: Path, model_config: Path | None = None):
    padded = [pad_row(r) for r in rows]
    idx = {name: i for i, name in enumerate(BREAKDOWN_HEADER)}
    counts = load_variant_counts(model_config)
    layer_totals: dict[tuple[str, str], float] = defaultdict(float)
    for r in padded:
        layer_totals[(r[idx["phase"]], r[idx["variant"]])] += float(r[idx["kernel_time_us"]] or 0)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(SUMMARY_HEADER)
        weighted_totals: dict[tuple[str, str], float] = defaultdict(float)
        for r in padded:
            phase = r[idx["phase"]]
            variant = r[idx["variant"]]
            t_us = float(r[idx["kernel_time_us"]] or 0)
            total = layer_totals[(phase, variant)]
            count = variant_count_for(variant, counts)
            weighted_ms = t_us * count / 1000.0
            weighted_totals[(phase, variant)] += weighted_ms
            ratio = t_us / total if total else 0
            w.writerow(
                [
                    "kernel",
                    phase,
                    variant,
                    count,
                    module_group(r[idx["module"]], r[idx["kernel_name"]]),
                    r[idx["module"]],
                    r[idx["kernel_name"]],
                    f"{t_us:.3f}",
                    f"{ratio:.2%}",
                    r[idx["tflops"]],
                    r[idx["bandwidth_gb_s"]],
                    r[idx["m"]],
                    r[idx["n"]],
                    r[idx["k"]],
                    f"{weighted_ms:.6f}",
                    r[idx["metric_note"]],
                ]
            )
        for (phase, variant), total_us in sorted(layer_totals.items()):
            count = variant_count_for(variant, counts)
            w.writerow(["layer_time", phase, variant, count, "", "", "", f"{total_us:.3f}", "100.00%", "", "", "", "", "", f"{total_us * count / 1000.0:.6f}", ""])
        phase_weighted: dict[str, float] = defaultdict(float)
        for (phase, _variant), total_ms in weighted_totals.items():
            phase_weighted[phase] += total_ms
        for phase, total_ms in sorted(phase_weighted.items()):
            w.writerow(["phase_weighted_total", phase, "", "", "", "", "", "", "", "", "", "", "", "", f"{total_ms:.6f}", ""])


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
    ap.add_argument("--target-decode-tokens", type=int)
    ap.add_argument("--summary-out", type=Path)
    ap.add_argument("--model-config", type=Path)
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
    rows = atom_breakdown(trace, ns.target_prefill_m, ns.target_decode_tokens) if framework == "atom" else sglang_breakdown(trace, ns.target_prefill_m)
    write_breakdown(rows, ns.out)
    if ns.summary_out:
        write_summary(rows, ns.summary_out, ns.model_config)
    print(f"framework={framework} rows={len(rows)} out={ns.out}")
    print(dict(Counter((pad_row(r)[0], pad_row(r)[1]) for r in rows)))
    if ns.summary_out:
        print(f"summary_out={ns.summary_out}")


if __name__ == "__main__":
    main()
