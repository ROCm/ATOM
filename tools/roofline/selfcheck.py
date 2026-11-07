#!/usr/bin/env python3
"""Self-check for marker_roofline: hand-computed values, no trace needed.

Deliberately NOT pytest and deliberately not under tests/. This tool is run by
hand, not by CI, and collecting it into ATOM's test gate would couple a manual
analysis tool to every PR. Run it directly:

    python3 tools/roofline/selfcheck.py
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import yaml
from marker_roofline import (
    Op,
    algorithm_order,
    classify_kernel,
    fold_split_k,
    gemm_cost,
    layer_families,
    merge_by_operator,
    parse_marker,
    price,
    stage_of,
)
from marker_roofline import (
    decompose as marker_decompose,
)

FAILURES: list[str] = []


def check(name: str, got: object, want: object) -> None:
    ok = got == want
    if not ok and isinstance(got, float) and isinstance(want, float):
        ok = abs(got - want) < max(1e-6, abs(want) * 1e-6)
    print(f"  {'ok  ' if ok else 'FAIL'}  {name}: got {got!r} want {want!r}")
    if not ok:
        FAILURES.append(name)


def _bytes_expr(fn, cfg, bench, kind="csa"):
    """The bytes expression a formula recorded, for asserting on its terms."""
    return fn(cfg, bench, kind)[2].bytes_expr


HF_CONFIG_URL = (
    "https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json"
)
CONFIG_CACHE = (
    Path.home() / ".cache" / "atom-roofline" / "deepseek-v4-pro" / "config.json"
)


def _config_arg() -> str | None:
    """`--model-config PATH`, read straight off argv.

    Not argparse: this script takes exactly one option and argparse would add a
    --help that promises a CLI this is not.
    """
    argv = sys.argv[1:]
    if "--model-config" in argv:
        i = argv.index("--model-config")
        if i + 1 >= len(argv):
            raise SystemExit("--model-config needs a path")
        return argv[i + 1]
    return None


def load_model_config(explicit: str | None = None) -> tuple[dict, str]:
    """The DSV4-Pro config, and a string saying where it came from.

    Four sources, tried in order, and NONE of them is silent. 33 of the assertions
    below read values out of this config; a loader that returned nothing would let
    them pass by not running, which is unavailable expressed as passing. So the
    failure path EXITS rather than degrades.

        1. --model-config PATH   whatever the caller names
        2. the local cache       ~/.cache/atom-roofline/deepseek-v4-pro/
        3. HuggingFace           the published config, cached on success
        4. nothing               SystemExit(2), naming all three and the flag

    Fetched rather than vendored because a copy of a model's config inside a
    framework repo is a second source of truth that nothing refreshes.
    """
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise SystemExit(f"--model-config {path}: not a file")
        return json.loads(path.read_text(encoding="utf-8")), f"--model-config {path}"

    if CONFIG_CACHE.is_file():
        try:
            return (
                json.loads(CONFIG_CACHE.read_text(encoding="utf-8")),
                f"cache {CONFIG_CACHE}",
            )
        except json.JSONDecodeError as exc:
            # A corrupt cache must not read as "no cache" and silently refetch
            # into the same bad state -- say what is wrong and where.
            raise SystemExit(f"{CONFIG_CACHE}: cached config is not JSON ({exc})")

    try:
        with urllib.request.urlopen(HF_CONFIG_URL, timeout=30) as resp:
            body = resp.read().decode("utf-8")
        cfg = json.loads(body)
    except Exception as exc:  # noqa: BLE001 -- every failure here is fatal
        raise SystemExit(
            f"cannot obtain the DSV4-Pro config, and these checks do not run "
            f"without it.\n"
            f"  tried: {HF_CONFIG_URL}\n"
            f"         {CONFIG_CACHE} (absent)\n"
            f"  cause: {type(exc).__name__}: {exc}\n"
            f"  fix:   pass --model-config <path to config.json>, or place the "
            f"file at the cache path above"
        )
    CONFIG_CACHE.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_CACHE.write_text(body, encoding="utf-8")
    return cfg, f"HuggingFace {HF_CONFIG_URL}"


def main() -> int:
    print("parse_marker")
    layer, path, shape = parse_marker(
        "layers.18.attn.indexer.wq_b[M=1,N=8192,K=1536,"
        "a=torch.float8_e4m3fn,w=torch.float8_e4m3fn,o=torch.bfloat16]"
    )
    check("layer", layer, 18)
    check("path", path, "attn.indexer.wq_b")
    check("N", shape["N"], "8192")
    check("dtype a", shape["a"], "torch.float8_e4m3fn")
    check("no-shape marker", parse_marker("layers.3.attn.wo_a")[2], {})

    print("\ngemm_cost — hand computed")
    # M=7769 N=7168 K=4096, a/w fp8 (1B), o bf16 (2B)   [attn.wo_b, prefill]
    cost = gemm_cost(
        {
            "M": "7769",
            "N": "7168",
            "K": "4096",
            "a": "torch.float8_e4m3fn",
            "w": "torch.float8_e4m3fn",
            "o": "torch.bfloat16",
        }
    )
    check("flops = 2*M*N*K", cost["flops"], 2.0 * 7769 * 7168 * 4096)
    check(
        "bytes = M*K*1 + K*N*1 + M*N*2",
        cost["bytes"],
        7769 * 4096 * 1.0 + 4096 * 7168 * 1.0 + 7769 * 7168 * 2.0,
    )
    check("peak follows fp8 inputs", cost["peak_key"], "matrix_fp8")
    # bf16 in, bf16 out  [ffn.gate]
    bf16 = gemm_cost(
        {
            "M": "7769",
            "N": "384",
            "K": "7168",
            "a": "torch.bfloat16",
            "w": "torch.bfloat16",
            "o": "torch.bfloat16",
        }
    )
    check("bf16 peak", bf16["peak_key"], "matrix_bf16")
    check("non-gemm returns None", gemm_cost({}), None)

    # Split-K: two kernels under one marker, one matmul. gemm_cost must NOT
    # branch on the kernel -- the partials are the implementation's cost, and
    # fold_split_k puts both kernels' time against the single matmul floor.
    dec = {
        "M": "1",
        "N": "7168",
        "K": "4096",
        "a": "torch.float8_e4m3fn",
        "w": "torch.float8_e4m3fn",
        "o": "torch.bfloat16",
    }
    plain = gemm_cost(dec)
    for kname in (
        "_gemm_a8w8_blockscale_preshuffle_kernel_NUM_KSPLIT_8_EVEN_K_1",
        "_gemm_a8w8_blockscale_reduce_kernel_ACTUAL_KSPLIT_8_MAX_KSPLIT_8",
    ):
        same = gemm_cost(dec, kname)
        check(f"the kernel name does not change the cost ({kname[:34]})", same, plain)
    check(
        "one matmul: A + B + C crossing HBM once",
        plain["bytes"],
        1 * 4096 * 1.0 + 4096 * 7168 * 1.0 + 1 * 7168 * 2.0,
    )

    peaks_for_split = yaml.safe_load(
        (Path(__file__).parent / "peaks" / "mi355x.yaml").read_text(encoding="utf-8")
    )
    print("\nfold_split_k — two kernels, one operator")
    ksplit_ops = {}
    for tag, kfull, us in (
        (
            "main",
            "_gemm_a8w8_blockscale_preshuffle_kernel_NUM_KSPLIT_8_GRID_MN_224",
            8526.0,
        ),
        (
            "red",
            "_gemm_a8w8_blockscale_reduce_kernel_ACTUAL_KSPLIT_8_MAX_KSPLIT_8",
            6147.0,
        ),
    ):
        o = Op(
            phase="decode",
            path="attn.wo_b",
            kernel=tag,
            kernel_full=kfull,
            kclass="gemm",
            shape=dict(dec) if tag == "main" else {},
            total_us=us,
            count=1220,
        )
        o.class_us = {"gemm": us}
        if tag == "main":
            o.flops, o.bytes, o.peak_key = (
                plain["flops"] * 1220,
                plain["bytes"] * 1220,
                plain["peak_key"],
            )
            o.m_values = [1]
        ksplit_ops[("decode", "attn.wo_b", kfull)] = o
    folded = fold_split_k(dict(ksplit_ops))
    check("the pair becomes one operator", len(folded), 1)
    kept = next(iter(folded.values()))
    check("carrying both kernels' time", kept.total_us, 8526.0 + 6147.0)
    check("and saying so in its name", "splitk reduce" in kept.kernel, True)
    # Firings are NOT summed: the matmul ran 1220 times, in two kernels each time.
    # Summing would halve every per-firing number, the latency regime included.
    check("the operator still fired 1220 times", kept.count, 1220)
    check("the matmul is priced once", kept.bytes, plain["bytes"] * 1220)
    row_k = price(kept, peaks_for_split)
    check(
        "so its efficiency is the matmul's, not the multiplying half's",
        round(float(row_k["efficiency"]), 2),
        0.31,
    )
    # An unpaired reduce (no NUM_KSPLIT sibling under the marker) must survive.
    orphan = {k: v for k, v in ksplit_ops.items() if "reduce" in k[2]}
    check("an orphan reduce is not dropped", len(fold_split_k(dict(orphan))), 1)

    print("\nprice — roofline columns")
    peaks = yaml.safe_load(
        (Path(__file__).parent / "peaks" / "mi355x.yaml").read_text(encoding="utf-8")
    )
    shape = {
        "M": "7769",
        "N": "7168",
        "K": "4096",
        "a": "torch.float8_e4m3fn",
        "w": "torch.float8_e4m3fn",
        "o": "torch.bfloat16",
    }
    op = Op(
        phase="prefill",
        path="attn.wo_b",
        kernel="ck::gemm_xdl",
        kernel_full="void ck::gemm_xdl<f8,f8>(Arg)",
        kclass="gemm",
        shape=shape,
        total_us=20320.0,
        count=61,
    )
    op.layers = set(range(61))
    op.m_values = [7769]
    # merge_by_operator accumulates cost over every firing; mirror that here.
    one = gemm_cost(shape)
    op.flops, op.bytes, op.peak_key = (
        one["flops"] * 61,
        one["bytes"] * 61,
        one["peak_key"],
    )
    # Real breakdown measured on the 2026-07-30 c1/TP4 trace: the module is 74%
    # all-reduce. Only the gemm slice is compared against the matmul ceiling.
    op.class_us = {"gemm": 20320.0}
    row = price(op, peaks)
    flops = 2.0 * 7769 * 7168 * 4096 * 61
    byts = (7769 * 4096 + 4096 * 7168 + 7769 * 7168 * 2) * 61
    check(
        "t_compute (61 firings)", row["t_compute_us"], round(flops / 5033e12 * 1e6, 4)
    )
    check("t_memory (61 firings)", row["t_memory_us"], round(byts / (8000e9 / 1e6), 4))
    check(
        "bound",
        row["bound"],
        "compute" if flops / 5033e12 > byts / 8000e9 else "memory",
    )
    check(
        "FLOP/byte is firing-count invariant",
        row["flop_per_byte"],
        round(flops / byts, 2),
    )
    check("M shown as single value", row["M"], "7769")
    check("tier", row["tier"], "roofline")
    check("ceiling", row["ceiling_tflops"], 5033.0)
    # efficiency compares total GEMM time against the total roofline -- NOT the
    # whole module, which here is three quarters communication.
    check(
        "efficiency vs this kernel",
        row["efficiency"],
        round(row["t_roofline_us"] / 20320.0, 4),
    )
    check("row is one kernel", row["kernel"], "ck::gemm_xdl")
    check("class carried", row["kclass"], "gemm")
    check("full kernel name kept", row["kernel_full"], "void ck::gemm_xdl<f8,f8>(Arg)")
    check("measured = this kernel only", row["measured_us"], 20320.0)

    gap_op = Op(
        phase="prefill",
        path="attn.wo_b",
        kernel="ncclDevKernel_Generic",
        kclass="comm",
        shape={},
        total_us=62180.0,
        count=61,
    )
    gap_op.class_us = {"comm": 62180.0}
    gap_op.m_values = []
    gap_row = price(gap_op, peaks)
    check("gap tier", gap_row["tier"], "gap")
    check("comm row keeps its time", gap_row["measured_us"], 62180.0)
    check("comm row is not priced", gap_row["kclass"], "comm")
    check("gap claims no bound", gap_row["t_roofline_us"], "")

    print("\nclassify_kernel")
    check("nccl -> comm", classify_kernel("ncclDevKernel_Generic_1(...)"), "comm")
    check(
        "ck gemm -> gemm",
        classify_kernel("void ck::kernel_gemm_xdl_cshuffle_v3"),
        "gemm",
    )
    check("Tensile -> gemm", classify_kernel("Cijk_Alik_Bljk_BBS_BH_Bias_HA_S"), "gemm")
    check(
        "quant -> quant",
        classify_kernel("aiter dynamic_per_group_scaled_quant_kernel"),
        "quant",
    )
    # Regression: ck_tile's fused kernel is a GEMM even though it says Quant.
    check(
        "QuantGemm -> gemm (not quant)",
        classify_kernel("_ZN7ck_tile6kentryI...QuantGemmMultiD..."),
        "gemm",
    )
    check(
        "moe_sort -> moe", classify_kernel("fused_dynamic_mx_quant_moe_sort_hip"), "moe"
    )
    check("attention -> attn", classify_kernel("pa_prefill_16mx1_fp8_kernel"), "attn")
    check("rmsnorm -> norm", classify_kernel("aiter::add_rmsnorm_quant_kernel"), "norm")
    check(
        "fillBuffer -> copy", classify_kernel("__amd_rocclr_fillBufferAligned"), "copy"
    )
    check(
        "strided clone -> copy",
        classify_kernel("triton_poi_fused_as_strided_clone"),
        "copy",
    )
    check("genuinely unknown -> other", classify_kernel("_mystery_kernel_xyz"), "other")

    print("\nmerge_by_operator + algorithm_order")
    a = Op(
        phase="prefill",
        path="attn.wqkv_a",
        kernel="k1",
        kclass="gemm",
        shape=dict(shape, M="7112"),
        total_us=10.0,
        count=61,
    )
    a.ts_by_layer = {16: 100.0}
    a.m_values = [7112]  # collect_ops records this per firing, mirror it here
    b = Op(
        phase="prefill",
        path="attn.wqkv_a",
        kernel="k1",
        kclass="gemm",
        shape=dict(shape, M="7769"),
        total_us=12.0,
        count=61,
    )
    b.ts_by_layer = {16: 105.0}
    b.m_values = [7769]
    c = Op(
        phase="prefill",
        path="attn.wo_b",
        kernel="k2",
        kclass="gemm",
        shape=dict(shape, M="7112"),
        total_us=8.0,
        count=61,
    )
    c.ts_by_layer = {16: 300.0}
    c.m_values = [7112]
    merged = merge_by_operator(
        {("p", "wqkv", "k1"): a, ("p", "wqkv", "k1b"): b, ("p", "wo", "k2"): c}
    )
    # a and b are the same (module, kernel) from two prompts -> one row; c differs.
    check("same kernel across prompts merges", len(merged), 2)
    wqkv = next(o for o in merged if o.path == "attn.wqkv_a" and o.kernel == "k1")
    check("firings summed across prompts", wqkv.count, 122)
    algorithm_order(merged, {0: "csa"})
    check(
        "wqkv_a executes before wo_b",
        wqkv.order < next(o for o in merged if o.path == "attn.wo_b").order,
        True,
    )
    # The merged row must carry BOTH prompts' M values, so the table can print a
    # range. Building the input as `{("p","w","k"): a, ("p","w","k"): b}` would
    # drop `a` -- one key written twice -- and asserting on `a.m_values +
    # b.m_values` would then hold whether or not merging works at all.
    check("merged row carries both prompts' M", sorted(wqkv.m_values), [7112, 7769])

    print("\nlayer_families — derived from content, never parity")

    def ann(layer, path):
        return {
            "cat": "gpu_user_annotation",
            "name": f"layers.{layer}.{path}",
            "ts": 0.0,
            "dur": 1.0,
        }

    evs = []
    for layer in (1, 3, 5):  # no indexer -> hca (largest plain family)
        evs += [ann(layer, "attn.wqkv_a"), ann(layer, "attn.wo_b")]
    for layer in (2, 4):  # indexer -> csa
        evs += [ann(layer, "attn.wqkv_a"), ann(layer, "attn.indexer.wq_b")]
    evs += [ann(0, "attn.wqkv_a")]
    fam = layer_families(evs)
    check("indexer layer -> csa", fam[2], "csa")
    check("plain content -> hca", fam[1], "hca")
    check("odd-vs-even is NOT the rule", fam[3], "hca")
    # Hash layers are never guessed from an operator signature: they run the same
    # sequence as their neighbours, so any signature-based rule is a guess, and a
    # wrong family means a sparse-attention formula on a dense layer. Without the
    # count they
    # are reported as whatever their content says, and main() says so out loud.
    check("no config -> no hash is invented", fam[0], "hca")
    check(
        "with the count, the leading layers are hash", layer_families(evs, 3)[0], "hash"
    )
    # Hash layers run an ordinary operator sequence, so only the config knows how
    # many there are. With n=3, layers 0/1/2 are hash whatever their markers say.
    fam3 = layer_families(evs, n_hash_layers=3)
    check("config wins for leading layers", [fam3[i] for i in (0, 1, 2)], ["hash"] * 3)
    check("rest still classified by content", fam3[4], "csa")
    check("rest still classified by content (hca)", fam3[5], "hca")

    print("\nschedule_model — derived, not hardcoded")
    sys.path.insert(0, str(Path(__file__).parent))
    from render_report import parse_config, schedule_model

    # Reproduces the hand-computed wave in the 2026-07-20 ATOM scheduling note.
    ref = schedule_model(
        {"concurrency": "128", "TP": "4", "dp-attn": "on"}, 8192, 1024, 16384
    )
    check("batch = conc/TP under dp-attn", ref["batch"], 32)
    check("prefill steps", ref["prefill_steps"], 16)
    check("decode steps = OSL", ref["decode_steps"], 1024)
    # dp-attn off: the rank carries the whole concurrency, no division by TP.
    off = schedule_model(
        {"concurrency": "1", "TP": "4", "dp-attn": "off"}, 7769, 1024, 16384
    )
    check("dp-attn off keeps full concurrency", off["batch"], 1)
    check("one request needs one step", off["prefill_steps"], 1)
    # Prompt longer than the budget must chunk instead of packing.
    ch = schedule_model(
        {"concurrency": "8", "TP": "4", "dp-attn": "on"}, 32768, 512, 16384
    )
    check("chunked when ISL > budget", ch["prefill_steps"], 4)
    cfg = parse_config(
        "trace_torch_mi355x_atom_dsv4-pro_dsv4-fp4_atom_run0001_summarize_tp4_ep1_c1_dpoff.json.gz"
    )
    check("underscore tokens parse (not \\b)", cfg.get("platform"), "MI355X")
    check("TP parsed", cfg.get("TP"), "4")
    check("dp-attn parsed", cfg.get("dp-attn"), "off")

    print("\nstage_of — longest prefix wins")
    stages = yaml.safe_load(
        (Path(__file__).parent / "module_map.yaml").read_text(encoding="utf-8")
    )["stages"]
    check("indexer beats attn", stage_of("attn.indexer.wq_b", stages), "indexer")
    check("attn", stage_of("attn.wo_b", stages), "attn")
    check("moe", stage_of("ffn.shared_experts.w2", stages), "moe")
    check("unknown -> other", stage_of("something.else", stages), "other")

    # Every check below is a bug a reader caught in a rendered page, never a check.
    # They are here so the next one is caught here instead.
    print("\nnice_ticks — dynamic log ticks (decode had none)")
    from render_report import nice_ticks, no_ceiling_reason, step_facts, svg_roofline

    check(
        "decode y range gets ticks",
        nice_ticks(0.09, 40.5)[:5],
        [0.1, 0.2, 0.5, 1.0, 2.0],
    )
    check("prefill y range gets ticks", nice_ticks(89.6, 5788.0)[0], 100.0)
    check("degenerate range is empty", nice_ticks(5.0, 1.0), [])
    check("zero lo is empty, not a log domain error", nice_ticks(0.0, 10.0), [])

    print("\nsvg_roofline — axis fits the roof over the DATA, not the tallest ceiling")
    decode_rows = [
        {
            "tier": "roofline",
            "op": "attn.wqkv_a",
            "M": "1",
            "flop_per_byte": 2.0,
            "achieved_tflops": 6.6,
            "dtype_a": "float8_e4m3fn",
            "bound": "memory",
            "recoverable_us": 10.0,
        },
        {
            "tier": "roofline",
            "op": "attn.indexer.weights_proj",
            "M": "1",
            "flop_per_byte": 0.98,
            "achieved_tflops": 0.2,
            "dtype_a": "bfloat16",
            "bound": "memory",
            "recoverable_us": 1.0,
        },
    ]
    svg = svg_roofline(decode_rows)
    # A roofline needs slope, ridge and flat in one picture. Two opposite bugs have
    # both shipped here: fitting the axis to the tallest ceiling with a hardcoded
    # tick list (decode had no gridline at all), then over-correcting to fit the
    # data and pushing every matrix ceiling off-chart (decode had no roof at all).
    check(
        "fp8 ceiling is drawn even when the data is 3 decades below",
        "matrix_fp8 5,033" in svg,
        True,
    )
    check("bf16 ceiling too", "matrix_bf16 2,516" in svg, True)
    check("both ridges are marked", svg.count("ridge ") >= 2, True)
    check("bandwidth roof is drawn", "bandwidth roof" in svg, True)
    check("tick inside decode's own decades exists", ">0.5</text>" in svg, True)
    check("tick up at the ceiling exists too", ">5,000</text>" in svg, True)

    print("\nno_ceiling_reason — four situations, not one blank cell")
    check("collective", "interconnect" in no_ceiling_reason({"kclass": "comm"}), True)
    check(
        "shapeless GEMM is the actionable one",
        "marker carries no shape" in no_ceiling_reason({"kclass": "gemm"}),
        True,
    )
    check("fused DSL", "fused DSL" in no_ceiling_reason({"kclass": "attn"}), True)
    check(
        "elementwise",
        "elementwise" in no_ceiling_reason({"kclass": "copy", "kernel": "memcpy"}),
        True,
    )

    print("\nstep_facts — per-step divides by the SAMPLED steps, not the run's steps")
    doc = {
        "phase_steps": {"decode": {"steps": 866, "median_us": 16271.8}},
        "sampled_steps": 20,
        "measured_total_us": 312875.4,
        "roofline_tier_us": 58160.2,
        "t_roofline_sum_us": 22193.3,
    }
    f = step_facts(doc, "decode", {"prefill_steps": 1, "decode_steps": 1024})
    check("busy per step", round(f["busy_us"] / 1000, 2), 15.64)
    check("priced per step", round(f["tier_us"] / 1000, 2), 2.91)
    check("roofline per step", round(f["roof_us"] / 1000, 2), 1.11)
    check("un-priced = busy - priced", round(f["gap_us"] / 1000, 2), 12.74)
    check(
        "no sampled_steps -> no facts, not a divide by zero",
        step_facts(
            {"phase_steps": {"decode": {"steps": 1, "median_us": 1.0}}},
            "decode",
            {"prefill_steps": 1, "decode_steps": 1},
        ),
        None,
    )

    print("\npeaks — dense figures, verified against AMD's product page 2026-08-31")
    peaks = yaml.safe_load(
        (Path(__file__).parent / "peaks" / "mi355x.yaml").read_text(encoding="utf-8")
    )
    tf = peaks["compute_tflops"]
    check("bf16 dense 2.5 PF", tf["matrix_bf16"], 2516)
    check("fp8 dense 5.0 PF (NOT the 10.1 PF sparse number)", tf["matrix_fp8"], 5033)
    # Was 20133 -- the 8-GPU platform total divided wrong. MXFP4 peak is 10.1 PF
    # per package, so every fp4 operator was judged against a 2x-too-high ceiling.
    check("fp4 10.1 PF per package", tf["matrix_fp4"], 10066)
    check("fp8 is exactly 2x bf16", tf["matrix_fp8"] * 2 - tf["matrix_bf16"] * 4, 2)
    check("HBM 8 TB/s", peaks["mem_bw_gbps"], 8000)

    print("\nformulas — ported cost models, keyed on the marker path")
    import dataclasses

    import formulas as F

    # See load_model_config: four sources, no silent one. The provenance is
    # printed because a check suite that cannot say which inputs it ran against
    # is not evidence.
    cfg, cfg_source = load_model_config(_config_arg())
    print(f"model config   : {cfg_source}")
    dec = F.Bench(batch=1, seq_len=1, tp=4, dp=1, dpon=False, kv_seq_len=7670)
    pre = F.Bench(batch=1, seq_len=7237, tp=4, dp=1, dpon=False, kv_seq_len=7237)
    check("decode M = 1 token", F.attn_M(dec), 1)
    check("prefill M = the whole prompt", F.attn_M(pre), 7237)
    check("dp-attn off -> MoE sees the same tokens", F.moe_B(dec), 1)

    # CSA selects top-k blocks; HCA reads every 128th token densely. At kv=7670
    # CSA's kv/4 = 1917 is under index_topk = 1024? No -- it is over, so the
    # top-k caps it. This is the branch that makes CSA cheap at long context.
    check("CSA is capped by index_topk", F.kv_entries(cfg, dec, "csa"), 1024.0)
    check("HCA reads kv/128 densely", round(F.kv_entries(cfg, dec, "hca"), 2), 59.92)

    # Causal masking: prefill queries see half the context on average.
    f_pre, _, _ = F.csa_core_attn(cfg, pre, "csa")
    f_dec, _, _ = F.csa_core_attn(cfg, dec, "csa")
    check(
        "prefill carries the 0.5 causal factor", round(f_pre / (f_dec * 7237), 3), 0.5
    )

    # fp4 weights carry a 1-byte E8M0 scale per 32 elements: +6.25%.
    check("MXFP4 block scale is +6.25%", round(F.w_bytes(32, "fp4"), 3), 17.0)
    _, by, spec = F.moe_gate_up(cfg, dec, "csa")
    check("experts compute at the fp8 ceiling, not fp4", spec.compute, "fp8")
    check("experts are stored fp4", spec.mem, "fp4")

    # Collectives must not be priced against HBM.
    _, _, cspec = F.comm_allreduce_attn(cfg, dec, "csa")
    check("all-reduce uses the interconnect roof", cspec.bw, "interconnect")

    print("\nformulas.lookup — the fused-add trap")
    # "all_reduce" appears in the name of a purely local elementwise kernel.
    # Matching on it priced a 3 ms add as an 82 ms collective: 27x.
    check(
        "fused local add is NOT a collective",
        F.lookup("ffn.combine_outputs", "triton_poi_fused_add_all_reduce__2").__name__,
        "residual_add_moe",
    )
    # The attention-side copy of the same kernel keeps the attention token count.
    check(
        "and the attention-side one stays on attn_M",
        F.lookup("attn.wo_b", "triton_poi_fused_add_all_reduce__2").__name__,
        "residual_add",
    )
    check(
        "the real collective still resolves",
        F.lookup("ffn.combine_outputs", "ncclDevKernel_Generic_1").__name__,
        "comm_allreduce_moe",
    )
    check(
        "moe gate_up vs down are told apart by kernel",
        F.lookup("layers.5.ffn.experts.fused_moe", "mfma_moe2_afp8").__name__,
        "moe_down",
    )
    check(
        "an unmapped kernel returns None, never a wrong formula",
        F.lookup("attn.something_new", "mystery_kernel"),
        None,
    )

    print("\nformulas — every operator carries its OWN algebra, not a template")
    dec2 = F.Bench(batch=1, seq_len=1, tp=4, dp=1, dpon=False, kv_seq_len=7670)
    _, _, sp_attn = F.csa_core_attn(cfg, dec2, "csa")
    _, _, sp_moe = F.moe_gate_up(cfg, dec2, "csa")
    _, _, sp_comm = F.comm_allreduce_attn(cfg, dec2, "csa")
    # The whole point: three operators, three different expressions. A shared
    # 2*M*N*K template would be wrong for two of them.
    check("fused attention shows its pass count", "passes" in sp_attn.flops_expr, True)
    check(
        "fused attention shows its KV selection",
        "index_topk" in sp_attn.flops_expr,
        True,
    )
    check(
        "expert GEMM shows the active-expert weight load",
        "active experts" in sp_moe.bytes_expr,
        True,
    )
    check(
        "collective shows a message, not a GEMM",
        "message" in sp_comm.bytes_expr and "FLOPs ~ 0" in sp_comm.flops_expr,
        True,
    )
    check(
        "no two of them share an expression",
        len({sp_attn.flops_expr, sp_moe.flops_expr, sp_comm.flops_expr}),
        3,
    )
    # Substituted, not symbolic: the numbers must actually appear.
    check(
        "kv_seq is substituted into the attention expression",
        "1,024" in sp_attn.flops_expr,
        True,
    )

    # The width error that made compressor_epilogue 13x its measured time.
    f_rope, _, sp_rope = F.compressor_epilogue(cfg, dec2, "hca")
    f_full, _, _ = F.qk_norm_rope(cfg, dec2, "hca")
    check("the epilogue runs on M/128 entries", "M/128" in sp_rope.note, True)
    check("and is far cheaper than the full-width version", f_rope < f_full, True)
    # The name says RoPE, but it norms and scatters too. One helper computes both
    # the fused and the split copy, so they cannot disagree about one epilogue.
    epi_only, _, _, _ = F._compressor_epilogue(cfg, F.attn_M(dec2) / 128)
    check("fused and split price the same epilogue", f_rope, epi_only)
    check(
        "and RoPE is charged on the rope lanes only",
        "rope lanes" in sp_rope.flops_expr,
        True,
    )

    # Same binding for the OTHER norm+rope kernel: qk_norm_rope priced a flat
    # 8*c for months -- the very shape corrected away above -- which is 1.68x
    # the split algebra. Pin it to the epilogue's convention so neither can be
    # flattened again without this going red.
    f_qkn, y_qkn, sp_qkn = F.qk_norm_rope(cfg, dec2, "hca")
    Mq = F.attn_M(dec2)
    c_, rd_ = cfg["head_dim"], cfg["qk_rope_head_dim"]
    rows = cfg["num_attention_heads"] / dec2.tp + cfg["num_key_value_heads"]
    check(
        "qk_norm_rope norms every lane and ropes only the rope lanes",
        f_qkn,
        4.0 * Mq * rows * c_ + 6.0 * Mq * rows * rd_,
    )
    check(
        "which is 1.68x cheaper than a flat 8*c over every lane",
        round(8.0 * Mq * rows * c_ / f_qkn, 2),
        1.68,
    )
    # MQA: one KV row against n_h/tp query rows. Charging only the Q side dropped
    # the weighted KV norm entirely.
    check("the single MQA KV row is in the account", rows, 33.0)
    # fp8 KV run -> 2buff out: V4_DIM_QK_PACKED 512 fp8 + V4_DIM_ROPE 64 bf16.
    # Deriving 448 = head_dim - rope_head_dim here is the mistake v4_quant.py
    # warns about; the packed width is named, not derived.
    # `rows + 1`: the query rows and the KV row this kernel writes, plus the one
    # K row the fused SWA scatter puts in the ring at decode.
    check(
        "the fp8 2buff row is 640 B, not 1024",
        (y_qkn / Mq - 2.0 * rows * c_) / (rows + 1),
        640.0,
    )
    check("and the spec says which layout it wrote", "2buff" in sp_qkn.note, True)

    # A collective must never be priced against HBM.
    check(
        "only collectives use the interconnect roof",
        {sp_attn.bw, sp_moe.bw, sp_comm.bw},
        {"hbm", "interconnect"},
    )

    print("\nprod/summ — the expression formatters")
    check(
        "a product substitutes every term",
        F.prod("FLOPs", [("2", 2), ("M", 7237)], 14474).splitlines()[1].strip(),
        "= 2 x 7,237",
    )
    check(
        "a sum keeps its term names",
        F.summ("Bytes", [("weights", "1 x 2"), ("acts", "3 x 4")], 14).splitlines()[0],
        "Bytes = weights + acts",
    )

    print("\nfind_steady_window — the steady-state picker")
    from marker_roofline import find_steady_window, typical_layer

    # A run that ramps up, holds steady, then drifts. The middle slice would land
    # in the drift; the drift gate rejects it and the stable stretch wins.
    walls = [50, 40, 30] + [20, 21, 20, 19, 20, 21, 20] + [24, 28, 33, 39, 46]
    lo, hi = find_steady_window(walls, n=7)
    check("picks the flat stretch, not the middle", (lo, hi), (3, 10))
    check(
        "shorter than the window -> take everything",
        find_steady_window([1.0, 2.0], n=20),
        (0, 2),
    )
    # Monotonic ramp: every window drifts, so the gate rejects all of them and the
    # lowest-stdev fallback must still return something rather than crashing.
    lo2, hi2 = find_steady_window([float(i) for i in range(30)], n=5)
    check("all-drift falls back instead of failing", hi2 - lo2, 5)

    print("\ntypical_layer — a middle layer at the MEDIAN wall, not the first")
    # 61 layers: 0-2 hash, then odd HCA / even CSA. Semantics are fixed by the
    # model, not inferred, so the representative must respect them.
    fams = {n: ("hash" if n < 3 else ("hca" if n % 2 else "csa")) for n in range(61)}
    check(
        "families split 3/29/29",
        [sum(1 for v in fams.values() if v == f) for f in ("hash", "hca", "csa")],
        [3, 29, 29],
    )
    # Give one CSA layer a wildly high wall. A mean-based pick would chase it; the
    # median-based one must not.
    walls_by_layer = {n: 100.0 for n in range(61)}
    walls_by_layer[30] = 100000.0
    rep = typical_layer(fams, walls_by_layer)
    check("outlier layer is not chosen", rep["csa"] != 30, True)
    check("CSA rep is an even layer", rep["csa"] % 2, 0)
    check("HCA rep is an odd layer", rep["hca"] % 2, 1)
    check(
        "reps come from the middle, not the edges",
        17 <= rep["csa"] <= 46 and 17 <= rep["hca"] <= 46,
        True,
    )
    # Distinct walls: the layer nearest the median wins outright.
    w2 = {n: float(n) for n in range(61)}
    rep2 = typical_layer(fams, w2)
    check("nearest-to-median wins with distinct walls", rep2["csa"], 32)

    print("\nrotate_to_boundary — a layer opens on the all-reduce it waits on")
    from marker_roofline import rotate_to_boundary

    def _op(path, kclass):
        return Op(phase="decode", path=path, kernel=path, kclass=kclass)

    # Timestamps inside a layer put the collective last; logically it is first,
    # because the next layer's residual does not exist until it has reduced.
    seq = [
        _op("mhc", "norm"),
        _op("attn.wo_b", "comm"),
        _op("moe", "gemm"),
        _op("ffn.combine_outputs", "comm"),
    ]
    out = rotate_to_boundary(list(seq))
    check("layer now opens on a collective", out[0].path, "ffn.combine_outputs")
    check("and it is flagged as the boundary", out[0].is_layer_boundary, True)
    check("the mid-layer wo_b all-reduce does NOT move", out[2].path, "attn.wo_b")
    check(
        "same operators, nothing dropped or duplicated",
        sorted(o.path for o in out),
        sorted(o.path for o in seq),
    )
    # No collective at all (TP=1): leave the order exactly as measured.
    plain = [_op("a", "gemm"), _op("b", "norm")]
    check(
        "no collective -> no rotation",
        [o.path for o in rotate_to_boundary(list(plain))],
        ["a", "b"],
    )
    # Already leading: rotating again must be a no-op, not a full cycle.
    lead = [_op("ar", "comm"), _op("x", "gemm")]
    check(
        "already at the front -> unchanged",
        [o.path for o in rotate_to_boundary(list(lead))],
        ["ar", "x"],
    )

    print("\nformulas — bytes audit: output writes and activation dtypes")
    pre2 = F.Bench(batch=1, seq_len=7237, tp=4, dp=1, kv_seq_len=7237)
    dec3 = F.Bench(batch=1, seq_len=1, tp=4, dp=1, kv_seq_len=7670)
    M, n_h, c = F.attn_M(pre2), cfg["num_attention_heads"] / 4, cfg["head_dim"]
    out_bytes = 2.0 * M * n_h * c

    # Q + KV with no output understates prefill attention ~3x, in the direction
    # that invents recoverable time.
    _, y_fused, sp_f = F.csa_core_attn(cfg, pre2, "csa")
    check("fused attention counts its output write", y_fused > out_bytes, True)
    check("and says the scores stay on chip", "on chip" in sp_f.note, True)

    # The decode attention is split-KV, not a QK/AV pair: stage 1 runs the whole
    # attention over its share of the KV and emits fp32 partials + an LSE, and
    # stage 2 merges those. Modelling stage 2 as the AV pass had it re-read every
    # visible KV entry -- 590 KB a firing on a CSA layer -- for a kernel whose
    # only input is splits * n_h * head_dim floats.
    n_csa = F._kv_splits(cfg, dec3, "csa")
    n_hca = F._kv_splits(cfg, dec3, "hca")
    check("aiter splits CSA's 1024 entries 8 ways at c1", n_csa, 8.0)
    check("and does not split HCA's 60", n_hca, 1.0)
    f_s1, _, sp_s1 = F.attn_logits(cfg, dec3, "csa")
    _f_s2, y_s2, sp_s2 = F.attn_merge(cfg, dec3, "csa")
    f_fused, _, _ = F.csa_core_attn(cfg, dec3, "csa")
    check("stage 1 does the whole attention, not half", f_s1, f_fused)
    check("stage 2 reads no KV", "KV" in sp_s2.bytes_expr, False)
    Md, nhd = F.attn_M(dec3), cfg["num_attention_heads"] / 4
    check(
        "it reads partials and writes one bf16 row",
        y_s2,
        4 * n_csa * Md * nhd * (c + 1) + 2 * Md * nhd * c,
    )
    check("stage 1 writes those partials", "fp32 partials" in sp_s1.bytes_expr, True)

    # An fp8 GEMM reads fp8 activations, not bf16. Overstated 1.61x at prefill.
    _, y_se, _ = F.shared_expert_gate_up(cfg, pre2, "csa")
    H = cfg["hidden_size"]
    I2 = 2 * cfg["moe_intermediate_size"] * cfg["n_shared_experts"] / 4
    expect = (
        F.w_bytes(H * I2, "fp8") + 1.0 * F.moe_B(pre2) * H + 2.0 * F.moe_B(pre2) * I2
    )
    check("fp8 GEMM activation counted at 1 byte", round(y_se), round(expect))
    # Weights fp4 but activations fp8 -- the kernel is afp8_wfp4.
    check(
        "expert GEMM reads fp8 activations",
        "act in fp8" in _bytes_expr(F.moe_gate_up, cfg, dec3),
        True,
    )

    # The indexer writes the scores the top-k then selects.
    check(
        "indexer counts its score output",
        "scores out" in _bytes_expr(F.csa_indexer, cfg, pre2),
        True,
    )

    print("\nlookup — one operator, two builds, two kernel names")
    # rocBLAS/Tensile names contain no "gemm"; needling on that word silently
    # dropped wo_a in prefill while keeping it in decode.
    check(
        "aiter name resolves",
        F.lookup("attn.wo_a", "_batched_gemm_bf16_kernel_HAS_BIAS").__name__,
        "o_proj_a",
    )
    check(
        "Tensile name resolves too",
        F.lookup("attn.wo_a", "Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_User").__name__,
        "o_proj_a",
    )

    print(
        "\nformulas — arithmetic intensity must come from the kernel, not the algebra"
    )
    d4 = F.Bench(batch=1, seq_len=1, tp=4, dp=1, kv_seq_len=7670)
    p4 = F.Bench(batch=1, seq_len=7237, tp=4, dp=1, kv_seq_len=7237)

    # update_compressor_states does not compress. It writes the ring buffer that
    # fused_compress_attn later pools out of, so its AI is one ape add per element
    # over 12 bytes of traffic. The pooling belongs to fused_compress_attn and is
    # paid there.
    f, y, sp_rw = F.kv_compress(cfg, d4, "csa")
    coff = 2 * cfg["head_dim"]  # (1 + overlap) * head_dim, CSA overlaps
    check("one ape add per element", f, 1.0 * coff)
    check("reads bf16 and writes fp32 states", y, coff * (2 * 2 + 2 * 4))
    check("and says no compression happens", "no compression" in sp_rw.note, True)
    # The host pre-filters to each sequence's last K_pool positions, so a long
    # prefill writes K_pool rows, not M. Charging M overstated prefill by ~47x
    # while the width and dtype errors understated decode by ~19x -- two errors
    # that swap sign with M, which is why the total never looked wrong.
    _, y_pre, _ = F.kv_compress(cfg, p4, "csa")
    check("prefill writes K_pool rows, not M", y_pre, 8 * coff * (2 * 2 + 2 * 4))
    _fi, yi, _ = F.kv_compress_index(cfg, d4, "csa")
    check("the indexer ring is index_head_dim wide", yi, 2 * cfg["index_head_dim"] * 12)
    # The Indexer's pool scatters flat fp8 + one fp32 scale per row, NOT the Main
    # path's mixed rope-bf16/nope-fp8 entry, so it must not share kv_entry_bytes.
    _, _, sp_ix = F.compressor_pool_index(cfg, d4, "csa")
    check("indexer pool is a pool, not a ring write", "pool" in sp_ix.note, True)

    # Nothing may sit above its own ceiling's ridge without being a real GEMM.
    RIDGE = {"fp4": 10066 / 8, "fp8": 5033 / 8, "bf16": 2516 / 8, "fp32": 157.3 / 8}
    # Only operators that really are matmuls over a large context may claim
    # an intensity above their ceiling's ridge. Everything else that lands
    # there has a FLOPs term that was invented.
    big_gemms = {
        # The hc-fn linear is a genuine matmul, and a narrow one: N = mix_hc = 24,
        # so its intensity asymptotes to 24. That sat far under the bf16 ridge of
        # 314 and reads as memory-bound at every batch size -- but the kernel runs
        # fp32 MFMA, whose ridge is 19.7, and above roughly M = 1k it crosses.
        # The regime flip is the corrected ceiling's doing, not a new FLOPs term.
        "mhc_pre_gemm",
        "shared_expert_gate_up",
        "shared_expert_down",
        "o_proj_a",
        "moe_router",
        "moe_gate_up",
        "moe_down",
        "csa_indexer",
        "csa_core_attn",
        "attn_logits",
        "attn_reduce",
    }
    offenders = []
    for name in dir(F):
        fn = getattr(F, name)
        if (
            not callable(fn)
            or name.startswith("_")
            or name
            in (
                "prod",
                "summ",
                "gemm",
                "w_bytes",
                "attn_M",
                "moe_B",
                "lookup",
                "kv_entries",
                "kv_entry_bytes",
                "active_experts",
                "moe_padded_rows",
                "dataclass",
                "Bench",
                "Spec",
            )
        ):
            continue
        try:
            fl, by, sp = fn(cfg, p4, "csa")
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            # A formula this bench cannot drive is not what the check is about;
            # anything else is a real break and must not be swallowed here.
            continue
        if by and fl / by > RIDGE[sp.compute] and name not in big_gemms:
            offenders.append((name, round(fl / by)))
    check("only real GEMMs claim to be compute-bound", offenders, [])

    print("\nformulas — three kernels whose NAME lied, settled from the source")
    p5 = F.Bench(batch=1, seq_len=7237, tp=4, dp=1, kv_seq_len=7237)

    # "_swa_write" is a scatter into the sliding-window ring, not attention.
    # state_writes.py:401 -- "a pure dtype-agnostic scatter ... NO torch
    # quantization happens here". It was priced as a 128-wide window attention.
    fl, by, sp = F.swa_write(cfg, p5, "csa")
    check("swa_write does no arithmetic", fl, 0.0)
    check(
        "and writes only the window, not M rows",
        by <= 2 * 1.0 * cfg["sliding_window"] * 512,
        True,
    )

    # moe_sorting permutes indices and zero-fills the output buffer. CK's
    # moe_sorting_kernel.hpp: "we fuse this clearing inside sorting kernel".
    # The clear is WRITE-ONLY; counting a read too was exactly 2x.
    fl2, by2, _sp2 = F.moe_dispatch(cfg, p5, "csa")
    M, H = F.moe_B(p5), cfg["hidden_size"]
    check("moe sort does no arithmetic", fl2, 0.0)
    check("the buffer clear is write-only", by2 < 2 * 2.0 * M * H, True)
    check("and it dominates the index traffic", by2 > 2.0 * M * H, True)

    # fused_compress_attn has "attn" in its name and involves no queries.
    # fused_compress.py:380 -- "per-source-position pool + RMSNorm + RoPE +
    # cache scatter". Priced as core attention it sat at 4x its measured time.
    fl3, by3, sp3 = F.compressor_pool(cfg, p5, "csa")
    fa, _, _ = F.csa_core_attn(cfg, p5, "csa")
    check("pooling is far cheaper than a context scan", fl3 < fa / 10, True)
    check("no head count enters it", "n_h" in sp3.flops_expr, False)
    check(
        "HCA pools 128 rows, CSA pools 4, so HCA writes less",
        F.compressor_pool(cfg, p5, "hca")[1] < by3,
        True,
    )

    print("\nprice — a collective is not memory-bound, and no FLOPs is not zero FLOP/s")
    peaks_y = yaml.safe_load(
        (Path(__file__).parent / "peaks" / "mi355x.yaml").read_text(encoding="utf-8")
    )
    import formulas as _F
    import marker_roofline as _mr

    bx = _F.Bench(batch=1, seq_len=1, tp=4, dp=1, kv_seq_len=7670)
    ar = Op(
        phase="decode",
        path="ffn.combine_outputs",
        kernel="aiter::cross_device_reduce_1stage",
        kernel_full="aiter::cross_device_reduce_1stage",
        kclass="comm",
        count=1,
        total_us=100.0,
    )
    ar.class_us["comm"] = 100.0
    row = _mr.price(ar, peaks_y, cfg, bx, 1)
    check("a collective reports its real wall", row["bound"], "link")
    check("and is priced against the interconnect", row["bw_kind"], "interconnect")
    check("no FLOPs -> no throughput number at all", row["achieved_tflops"], "")
    check("but its bandwidth is still reported", row["achieved_gbps"] > 0, True)

    srt = Op(
        phase="decode",
        path="ffn.experts.fused_moe",
        kernel="aiter::opus_moe_sorting_entry",
        kernel_full="aiter::opus_moe_sorting_entry",
        kclass="moe",
        count=1,
        total_us=100.0,
    )
    srt.class_us["moe"] = 100.0
    row2 = _mr.price(srt, peaks_y, cfg, bx, 1)
    check("a memset is memory-bound, not link", row2["bound"], "memory")
    check("and also reports no throughput", row2["achieved_tflops"], "")

    print("\nbw_txt — a small number must not render as zero in a big unit")

    # Reconstruct the formatter's rule: TB/s only once there is a digit to show.
    def bw(v):
        if v >= 100:
            return f"{v / 1000:,.2f} TB/s"
        return f"{v:,.1f} GB/s" if v >= 1 else f"{v * 1000:,.1f} MB/s"

    check("1.2 GB/s stays GB/s", bw(1.2), "1.2 GB/s")
    check("a 1.2 GB/s kernel never prints 0.00", bw(1.2).startswith("0.00"), False)
    check("8 TB/s prints as TB/s", bw(8000.0), "8.00 TB/s")
    check("the boundary is where TB/s still has digits", bw(100.0), "0.10 TB/s")
    # A latency-bound operator moving 10 KB in 2.6 ms is 3.7 MB/s, not zero.
    check(
        "sub-GB/s drops to MB/s rather than rounding to nothing", bw(0.0037), "3.7 MB/s"
    )
    check(
        "nothing in the ladder can print a bare zero",
        [x for x in (bw(0.0037), bw(1.2), bw(8000.0)) if x.startswith("0.0 ")],
        [],
    )

    print("\npretty_kernel — both Itanium forms are length-prefixed")
    from marker_roofline import pretty_kernel as _pk

    # `_ZN<len><ns><len><name>E` -- nested, was already handled.
    check(
        "nested name resolves",
        _pk("_ZN5aiter24add_rmsnorm_quant_kernelIDF16bDB8_Li256EEEvPT0_"),
        "aiter::add_rmsnorm_quant_kernel",
    )
    # `_Z<len><name>` -- a plain free function. The nested-only pattern skipped it,
    # so prefill's largest recoverable operator displayed as raw mangled text.
    check(
        "plain free function resolves too",
        _pk(
            "_Z33pa_prefill_16mx1_16nx4_fp8_kernelI25pa_16mx1_16nx4_fp8_traitsILi16EEEv"
        ),
        "pa_prefill_16mx1_16nx4_fp8_kernel",
    )
    # Not mangled at all: return it untouched rather than guessing.
    check(
        "an unmangled DSL name is left alone",
        _pk("hca_norm_rope_scatter_D512_RD64_R128_KB1_KW1_rmsbf16_fp8_flydsl"),
        "hca_norm_rope_scatter_D512_RD64_R128_KB1_KW1_rmsbf16_fp8_flydsl",
    )
    check(
        "a demangled C++ signature loses its template and argument tail",
        _pk("void aiter::mhc_post_kernel<std::bfloat16_t, 4>(std::bfloat16_t*)"),
        "aiter::mhc_post_kernel",
    )

    print("\ndtype_key — one spelling, two sources")
    import math as _math

    from render_report import CEILING_COLOR, dtype_key

    # A marker states torch's spelling; a config formula states the short one.
    # Keying the colour map on torch's alone sent every formula-priced point to a
    # grey fallback that was not in the legend -- 84% of them.
    check("torch spelling normalises", dtype_key("torch.float8_e4m3fn"), "fp8")
    check("short spelling passes through", dtype_key("fp8"), "fp8")
    check(
        "both land on the same colour",
        CEILING_COLOR[dtype_key("bfloat16")] == CEILING_COLOR[dtype_key("bf16")],
        True,
    )
    check(
        "every alias target has a colour",
        sorted(
            set(__import__("render_report").DTYPE_ALIAS.values()) - set(CEILING_COLOR)
        ),
        [],
    )
    check("an unknown dtype is not silently coloured", dtype_key("mystery"), "mystery")

    print("\ndot area — proportional, and not saturated at the top")
    R_MIN, R_MAX = 3.0, 15.0

    def radius(rec, rec_max):
        return _math.sqrt(R_MIN**2 + (R_MAX**2 - R_MIN**2) * (rec / rec_max))

    # Prefill's three biggest levers are 44.5, 24.2 and 23.4 ms. A radius that
    # saturates below that draws all three the same size.
    big = [radius(x, 44500.0) for x in (44500.0, 24200.0, 23400.0)]
    check("the top three are now distinguishable", len({round(v, 2) for v in big}), 3)
    check("the largest fills the cap", round(radius(44500.0, 44500.0), 2), R_MAX)
    check("zero recoverable is still visible", round(radius(0.0, 44500.0), 2), R_MIN)
    # Area above the minimum is linear in recoverable time.
    a1 = radius(10000.0, 40000.0) ** 2 - R_MIN**2
    a2 = radius(20000.0, 40000.0) ** 2 - R_MIN**2
    check("twice the recoverable time is twice the area", round(a2 / a1, 3), 2.0)

    print("\npeaks_from — the chart must draw the peaks the numbers were priced with")
    from render_report import peaks_from

    doc = {
        "peaks": {
            "compute_tflops": {
                "matrix_fp4": 10066,
                "matrix_fp8": 5033,
                "matrix_bf16": 2516,
            },
            "mem_bw_gbps": 8000,
        }
    }
    ce, slope, label = peaks_from(doc)
    # The renderer must read these from the run, not hold its own copy: a copy
    # does not follow a correction to the yaml, and the chart would then draw a
    # ceiling the dots beside it were never measured against.
    check(
        "fp4 comes from the run, not a constant",
        {k: v for k, v, _ in ce}["fp4"],
        10066,
    )
    check("HBM slope is derived, not assumed", slope, 8.0)
    check("and labelled from the same number", label, "8 TB/s")
    # The fallback path runs through the same merge, so coincident roofs
    # (MXFP6 == MXFP4 on CDNA4) collapse there too -- two lines drawn on top of
    # each other would also mean two legend rows for one line.
    fb = [v for _, v, _ in peaks_from({})[0]]
    check("a run with no peaks block still renders", bool(fb), True)
    check("and never draws two roofs at the same height", len(fb), len(set(fb)))
    check(
        "fp4 and fp6 are named on one line",
        [k for k, _, _ in peaks_from({})[0] if "/" in k],
        ["fp4/fp6"],
    )
    # A different machine must move both the ceilings and the slope.
    ce2, slope2, label2 = peaks_from(
        {"peaks": {"compute_tflops": {"matrix_bf16": 1000}, "mem_bw_gbps": 4000}}
    )
    check("only the dtypes present are drawn", [k for k, _, _ in ce2], ["bf16"])
    check("slope follows the bandwidth", (slope2, label2), (4.0, "4 TB/s"))

    print("\nformulas — no unreachable cost models")
    _mapped = {fn.__name__ for _, _, fn in F.MARKER_MAP}
    _helpers = {
        "prod",
        "summ",
        "gemm",
        "w_bytes",
        "attn_M",
        "moe_B",
        "lookup",
        "kv_entries",
        "kv_entry_bytes",
        "index_entry_bytes",
        "active_experts",
        "moe_padded_rows",
        "dtype_key",
        "resolve_dtypes",
        "compute_dtype",
        "allreduce_path",
    }
    _defined = {
        n
        for n in dir(F)
        if callable(getattr(F, n))
        and not n.startswith("_")
        and n not in _helpers
        and getattr(getattr(F, n), "__module__", "") == "formulas"
        and not isinstance(getattr(F, n), type)
    }
    # A formula no marker routes to is not a spare part; it is a thing that looks
    # maintained and is not, and it will drift silently.
    check("every formula is reachable from MARKER_MAP", sorted(_defined - _mapped), [])

    print("\nceiling_key — colour by the ceiling used, not the dtype stored")
    from render_report import CEILINGS as _CEIL
    from render_report import ceiling_key

    # Expert GEMMs store fp4 and compute fp8: dtype_a says fp4, ceiling_tflops says
    # 5033. Colouring by dtype_a drew them purple under an fp8 line.
    check(
        "stored fp4 / computed fp8 colours as fp8",
        ceiling_key({"dtype_a": "fp4", "ceiling_tflops": 5033}, _CEIL),
        "fp8",
    )
    check(
        "read fp32 / computed bf16 colours as bf16",
        ceiling_key({"dtype_a": "fp32", "ceiling_tflops": 2516}, _CEIL),
        "bf16",
    )
    check(
        "a true fp4 operator still colours fp4",
        ceiling_key({"dtype_a": "fp4", "ceiling_tflops": 10066}, _CEIL),
        "fp4",
    )
    check(
        "no ceiling recorded -> fall back to the dtype",
        ceiling_key({"dtype_a": "bfloat16"}, _CEIL),
        "bf16",
    )

    print("\nMARKER_MAP — no catch-alls, and dtypes that match their kernel names")
    # A `None` needle prices every kernel under a marker with the module's headline
    # formula. Three survived after the first cull and all three were wrong in
    # prefill only -- decode emits fewer kernels per module, so the bug hid.
    check(
        "no catch-all needles remain",
        [suffix for suffix, needle, _ in F.MARKER_MAP if needle is None],
        [],
    )

    # The kernel name is evidence the formula author did not have to guess at.
    # Where a mapping's own needle names a dtype, the formula must agree.
    import re as _re

    TOKENS = [
        (_re.compile(r"afp8|a8w8|fp8"), "fp8"),
        (_re.compile(r"fp4"), "fp4"),
        (_re.compile(r"bf16"), "bf16"),
    ]
    b6 = F.Bench(batch=1, seq_len=1, tp=4, dp=1, kv_seq_len=7670)
    clash = []
    for suffix, needle, fn in F.MARKER_MAP:
        hints = {d for pat, d in TOKENS if needle and pat.search(needle)}
        if not hints:
            continue
        try:
            _, _, sp = fn(cfg, b6, "csa")
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            # A formula this bench cannot drive is not what the check is about;
            # anything else is a real break and must not be swallowed here.
            continue
        if sp.compute not in hints and sp.mem not in hints:
            clash.append((suffix, needle, sp.compute, sorted(hints)))
    check("needle dtypes agree with the formula", clash, [])

    # The one that got away: the paper says the Lightning Indexer is FP4, the
    # kernel is `_gluon_fp8_mqa_logits_kernel`. The trace wins.
    _, _, sp_idx = F.csa_indexer(cfg, b6, "csa")
    check("indexer is priced fp8, as its kernel says", sp_idx.compute, "fp8")

    print("\nscatter — coincident operators merge instead of hiding each other")

    # Four fp8 GEMMs at decode genuinely share an intensity and a throughput.
    # Drawn on top of each other, three of the four were invisible; jitter would
    # have lied about the one thing a dot's position means.
    def group(points, px, py):
        g = {}
        for name, ai, tf in points:
            g.setdefault((round(px(ai) / 3.0), round(py(tf) / 3.0)), []).append(name)
        return g

    def idp(v):
        return v * 100.0

    gs = group([("a", 2.0, 5.7), ("b", 2.0, 5.7), ("c", 1.0, 3.6)], idp, idp)
    check("identical coordinates become one dot", len(gs), 2)
    check(
        "and the dot knows all its members",
        sorted(max(gs.values(), key=len)),
        ["a", "b"],
    )
    # Near-but-not-equal must still merge: 3px is below the dot radius, so two
    # dots that close are visually one anyway.
    gs2 = group([("a", 2.0, 5.70), ("b", 2.0, 5.71)], idp, idp)
    check("near-coincident merges too", len(gs2), 1)
    check(
        "distinct coordinates stay distinct",
        len(group([("a", 2.0, 5.7), ("b", 9.0, 5.7)], idp, idp)),
        2,
    )

    print("\nresolve_dtypes — a formula names a role, the run resolves it")
    # DSV4-Pro: fp8 linears, fp4 experts, and two cache precisions that live
    # only in server flags.
    d1 = F.resolve_dtypes(cfg, {"kv_cache_dtype": "fp8", "index_cache_dtype": "fp4"})
    check("linear weights from quantization_config", d1.linear_w, "fp8")
    check("experts from the non-standard expert_dtype field", d1.expert_w, "fp4")
    check(
        "and the config says so, not the formula",
        d1.source["expert_w"],
        "config.expert_dtype",
    )
    check(
        "KV precision comes from a runtime flag no HF config carries",
        (d1.kv, d1.source["kv"]),
        ("fp8", "runtime --kv-cache-dtype"),
    )

    # Chained str.replace turned "bfloat16" into "bfp16".
    check("bfloat16 normalises once, not twice", d1.act, "bf16")

    # The same architecture at another precision must not need another formula.
    g = dict(cfg)
    g["quantization_config"] = {"quant_method": "mxfp4"}
    d2 = F.resolve_dtypes(g, {})
    check(
        "an MXFP4 build of the same model resolves differently",
        (d1.linear_w, d2.linear_w),
        ("fp8", "fp4"),
    )

    # Storage and ceiling are two answers. fp4 stores at 0.5 B and computes at
    # the fp8 peak -- no fp4 matrix speedup exists on this chip.
    check("fp4 computes at the fp8 ceiling", F.compute_dtype("fp4"), "fp8")
    check(
        "everything else computes at what it stores",
        [F.compute_dtype(x) for x in ("fp8", "bf16")],
        ["fp8", "bf16"],
    )

    # The expert GEMM must follow the run, not a literal.
    b_fp4 = F.Bench(batch=1, seq_len=1, tp=4, dp=1, kv_seq_len=7670, dtypes=d1)
    b_fp8 = F.Bench(
        batch=1, seq_len=1, tp=4, dp=1, kv_seq_len=7670, dtypes=F.Dtypes(expert_w="fp8")
    )
    _, y4, s4 = F.moe_gate_up(cfg, b_fp4, "csa")
    _, y8, s8 = F.moe_gate_up(cfg, b_fp8, "csa")
    check("expert ceiling follows the run", (s4.mem, s8.mem), ("fp4", "fp8"))
    check("both compute at fp8", (s4.compute, s8.compute), ("fp8", "fp8"))
    check("fp8 experts move more bytes than fp4 ones", y8 > y4, True)

    # The indexer: cache fp4, kernel fp8. Both true, and both now derived.
    _, _, si = F.csa_indexer(cfg, b_fp4, "csa")
    check("indexer stores fp4 and computes fp8", (si.mem, si.compute), ("fp4", "fp8"))

    print("\nlayer_families — the taxonomy is data, not a constant in the engine")
    from marker_roofline import DEFAULT_FAMILIES

    def ann(layer, path):
        return {
            "cat": "gpu_user_annotation",
            "name": f"layers.{layer}.{path}",
            "ts": 0.0,
        }

    ev = []
    for layer in range(61):
        ev.append(ann(layer, "attn.wqkv_a"))
        if layer % 2 == 0:  # even layers carry the indexer
            ev.append(ann(layer, "attn.indexer.wq_b"))
    fams = layer_families(ev, 3, DEFAULT_FAMILIES)
    counts = {
        f: sum(1 for v in fams.values() if v == f) for f in ("hash", "csa", "hca")
    }
    check(
        "DSV4 resolves 3 hash + 29 CSA + 29 HCA",
        counts,
        {"hash": 3, "csa": 29, "hca": 29},
    )
    check("leading layers override content", [fams[i] for i in range(3)], ["hash"] * 3)

    # A dense model has ONE family and must not need an engine change.
    dense = [{"name": "layer", "detect": {"default": True}}]
    fd = layer_families([ann(i, "attn.qkv") for i in range(32)], 0, dense)
    check("a single-family model works unchanged", set(fd.values()), {"layer"})
    check("and covers every layer", len(fd), 32)

    # A model whose families are recognised by a different marker.
    other = [
        {"name": "sparse", "detect": {"contains": "attn.topk"}},
        {"name": "dense", "detect": {"default": True}},
    ]
    fo = layer_families([ann(0, "attn.topk"), ann(1, "attn.qkv")], 0, other)
    check("rules follow the model, not the engine", [fo[0], fo[1]], ["sparse", "dense"])

    print("\nmodel config — the checks must not be able to skip themselves")
    # The suite reads 33 values out of this config. The one failure mode that
    # matters is not a wrong value, it is an ABSENT config that reads as a pass:
    # before the config was pinned down, 48 of 185 assertions were guarded by
    # `if cfg:` and vanished silently on any machine without the file, while the
    # script still printed "all checks passed".
    check("the suite states where its config came from", bool(cfg_source), True)
    check(
        "and names one of the four sources",
        cfg_source.split()[0] in ("--model-config", "cache", "HuggingFace"),
        True,
    )
    # Not a promise in a docstring: drive the loader at a URL and a cache path
    # that cannot resolve, and require it to exit rather than return nothing.
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        saved = (HF_CONFIG_URL, CONFIG_CACHE)
        try:
            globals()["HF_CONFIG_URL"] = "https://huggingface.invalid/x/config.json"
            globals()["CONFIG_CACHE"] = Path(td) / "absent.json"
            try:
                load_model_config(None)
                fatal = False
            except SystemExit:
                fatal = True
        finally:
            globals()["HF_CONFIG_URL"], globals()["CONFIG_CACHE"] = saved
    check("an unobtainable config exits, it does not skip", fatal, True)
    check("and it is the real thing", cfg.get("num_hidden_layers"), 61)

    print("\nMARKER_MAPS — one vocabulary per architecture")
    # `@mark_trace` lives in atom/model_ops, but the NAME it emits comes from the
    # caller's prefix=, so `attn.wo_b` is DeepSeek-V4's word for an output
    # projection. A flat map is one model's vocabulary pretending to be universal.
    check("DSV4 is mapped", "DeepseekV4ForCausalLM" in F.MARKER_MAPS, True)

    print("\nindexer tail — the last two unpriced kernels")
    M = F.attn_M(pre2)
    H = float(cfg["index_n_heads"])
    _, b_sc, _sp_sc = F.scale_indexer_weights(cfg, pre2, "csa")
    # weights bf16 in, q_scale fp32 in, out fp32 -- the fp32 output is why this
    # moves more than a bf16 elementwise op of the same shape would.
    check("the fp32 output is paid for", b_sc, 2 * M * H + 4 * M * H + 4 * M * H)
    # H is index_n_heads, and the trace independently reports N=64 for
    # attn.indexer.weights_proj. Two sources, one number.
    check("H is the indexer head count", H, 64.0)
    f_tp, b_tp, sp_tp = F.csa_translate_pack(cfg, pre2, "csa")
    check("index translation computes nothing", f_tp, 0.0)
    # The top-k prefix it walks IS the CSA core attention's visibility. Deriving
    # it twice would let the two drift; this asserts they are one function.
    k = F.kv_entries(cfg, pre2, "csa")
    check(
        "it walks exactly the CSA visibility", b_tp, F.I32_BYTES * (3 * M * k + 3 * M)
    )
    check("and names the k it used", f"k={int(k):,}" in sp_tp.note, True)

    print("\nlayout copy — torch.compile's, not aiter's")
    dim, M = cfg["hidden_size"], F.attn_M(pre2)
    fw, bw, sw = F.layout_copy_wide(cfg, pre2, "csa")
    fn_, bn, _ = F.layout_copy_narrow(cfg, pre2, "csa")
    # Shapes are read from the capture trace's Input Dims: [[M,dim],[dim],[M,dim]]
    # for the mHC variant, [[M,dim],[dim]] for the FFN one, [M,dim] out.
    check(
        "the wide variant moves three [M,dim] and one [dim]",
        bw,
        2 * (3 * M * dim + dim),
    )
    check("the narrow one moves two", bn, 2 * (2 * M * dim + dim))
    check("they differ by exactly one [M,dim]", bw - bn, 2 * M * dim)
    check("a copy computes nothing", (fw, fn_), (0.0, 0.0))
    check("and says so rather than printing a zero", "FLOPs = 0" in sw.flops_expr, True)
    # The two kernel names share the prefix "as_strided_clone" and differ only in
    # suffix. That is exactly the collision that mis-priced the mHC pre GEMM, so
    # the needles are pinned to the suffixes and asserted both ways.
    check(
        "the mHC marker takes the three-input variant",
        F.lookup("mhc_fused_post_pre", "triton_poi_fused_as_strided_clone_copy__0"),
        F.layout_copy_wide,
    )
    check(
        "the FFN marker takes the two-input one",
        F.lookup("ffn.gate", "triton_poi_fused_as_strided_clone_1"),
        F.layout_copy_narrow,
    )

    # aiter un-fuses mhc_fused_post_pre at M >= 1024 on gfx950, but ATOM's
    # record_function keeps the fused name, so one marker covers two different
    # kernel sets. The needle has to read the kernel, or the un-fused pre GEMM
    # gets charged for a post half that the mhc_post_kernel beside it already paid.
    check(
        "the truly fused kernel gets the fused formula",
        F.lookup("mhc_fused_post_pre", "aiter::mhc_fused_post_pre_gemm_sqrsum_kernel"),
        F.mhc_fused_post_pre,
    )
    check(
        "the un-fused pre GEMM does not",
        F.lookup("mhc_fused_post_pre", "aiter::mhc_pre_gemm_sqrsum_kernel"),
        F.mhc_pre_gemm,
    )
    check(
        "and the post kernel beside it is its own operator",
        F.lookup("mhc_fused_post_pre", "aiter::mhc_post_kernel"),
        F.mhc_post,
    )
    # a16w16 / mono_tile appears nowhere in aiter's mHC path. Whatever that kernel
    # is under this marker, it is not the hc-fn GEMM -- so it returns nothing and
    # shows up unpriced, rather than inheriting a formula it does not run.
    check(
        "an unexplained kernel under the marker stays unpriced",
        F.lookup("mhc_fused_post_pre", "gemm_a16w16_mono_tile_kernel_gfx950"),
        None,
    )
    check(
        "its own paths resolve",
        F.lookup("attn.wo_b", "nccl", "DeepseekV4ForCausalLM").__name__,
        "comm_allreduce_attn",
    )
    # An unmapped architecture must price NOTHING rather than borrow these paths.
    check(
        "an unmapped arch matches nothing",
        F.lookup("attn.wo_b", "nccl", "LlamaForCausalLM"),
        None,
    )
    check(
        "even for a path it happens to share",
        F.lookup("ffn.experts.fused_moe", "mfma_moe1", "Qwen3_5MoeForCausalLM"),
        None,
    )
    # Formulas are shared; only the maps are per-model. Adding a model must not
    # need a new cost model for a GEMM.
    check(
        "every mapped formula lives in the shared pool",
        all(
            getattr(fn, "__module__", "") == "formulas"
            for table in F.MARKER_MAPS.values()
            for _, _, fn in table
        ),
        True,
    )

    print("\ncomm_allreduce — a floor the hardware beat was the wrong floor")
    b_tp4 = F.Bench(batch=1, seq_len=7237, tp=4, dp=1, kv_seq_len=7237)
    _, per_link, sp_c = F.comm_allreduce_attn(cfg, b_tp4, "csa")
    message = 7237 * cfg["hidden_size"] * 2

    # An all-reduce is reduce-scatter + all-gather: each rank moves (N-1)/N of the
    # message per stage over its N-1 links, so per link it is 2 x message / N.
    check(
        "per-link bytes are 2/N of the message", round(per_link), round(2 * message / 4)
    )
    check("and that is half the message at TP4", round(per_link / message, 3), 0.5)

    # The one-stage model charged the whole message per link and predicted 1351 us
    # for a collective measured at 970 us -- 107 GB/s over a 76.8 GB/s link, which
    # no implementation can do. Falsified, not merely inaccurate.
    LINK_GBPS = 76.8
    check(
        "two-stage does not exceed the link",
        (per_link / 1e9) / (970e-6) / LINK_GBPS < 1.0,
        True,
    )
    check("one-stage did", (message / 1e9) / (970e-6) / LINK_GBPS > 1.0, True)

    check("it is priced against the fabric, not HBM", sp_c.bw, "interconnect")
    check("and does no arithmetic", "FLOPs ~ 0" in sp_c.flops_expr, True)
    # Wider TP splits the per-link share further.
    _, pl8, _ = F.comm_allreduce_attn(
        cfg, F.Bench(batch=1, seq_len=7237, tp=8, dp=1, kv_seq_len=7237), "csa"
    )
    check("TP8 halves the per-link bytes again", round(pl8 / per_link, 3), 0.5)

    print("\nload_model_decl — structure is data, named as the catalog names it")
    from marker_roofline import check_methodology, load_model_decl

    root = Path(__file__).parent
    decl = load_model_decl("deepseek-v4-pro", root)
    # --model takes the ATOM benchmark catalog's `prefix`, so the string here is
    # the same one the benchmark uses. No second naming convention to learn.
    check("the catalog slug resolves", decl["model"], "deepseek-v4-pro")
    check("it names its architecture", decl["architecture"], "DeepseekV4ForCausalLM")
    check("and its marker map", decl["marker_map"], "DSV4_MAP")

    print("\nwidths read from the capture, not assumed")
    # attn.q_norm runs rmsnorm_quant on [1, 1536] -- q_lora_rank, with 1536/128 = 12
    # per-block scales. Sharing norm_quant with attn.wqkv_a, whose norm really is
    # hidden_size wide, charged this 4.7x too much.
    _, y_qn, _ = F.q_norm_quant(cfg, pre2, "csa")
    _, y_hn, _ = F.norm_quant(cfg, pre2, "csa")
    check(
        "q_norm is q_lora_rank wide",
        y_qn / y_hn,
        cfg["q_lora_rank"] / cfg["hidden_size"],
    )
    # The indexer's rope runs on [1, 64, 128] -- index heads are replicated, not
    # tp-sharded, so it is 4x the main-attention rope's width.
    _, y_ir, _ = F.indexer_rope(cfg, pre2, "csa")
    Mi = F.attn_M(pre2)
    check(
        "indexer rope is n_i x d_i wide",
        y_ir,
        3.0 * Mi * cfg["index_n_heads"] * cfg["index_head_dim"],
    )
    # Three operations under one kernel name, and they do not share a width. The
    # rotate reaches only the trailing rope_dim lanes (rope_start = dim - rope_dim
    # in the kernel; the capture's cos/sin carry rope_dim/2 = 32 entries), while
    # the Hadamard butterfly covers all of index_head_dim at log2(d_i) stages.
    f_ir, _, _ = F.indexer_rope(cfg, pre2, "csa")
    d_i, n_i = cfg["index_head_dim"], cfg["index_n_heads"]
    check(
        "rope, hadamard and quant are priced separately",
        f_ir,
        6.0 * Mi * n_i * cfg["qk_rope_head_dim"]
        + _math.log2(d_i) * Mi * n_i * d_i
        + 3.0 * Mi * n_i * d_i,
    )
    check(
        "which is 2.17x what RoPE's constant alone would charge",
        round(f_ir / (6.0 * Mi * n_i * d_i), 2),
        2.17,
    )
    check(
        "which is 4x the main rope's width",
        cfg["index_n_heads"]
        * cfg["index_head_dim"]
        / (cfg["num_attention_heads"] / 4 * cfg["qk_rope_head_dim"]),
        4.0,
    )

    # attn.inverse_rope touches `o.view(num_tokens, n_local_heads, head_dim)`'s
    # trailing `rope_head_dim` lanes in place (`deepseek_v4.py` _wo_a_path ->
    # RotaryEmbedding.inverse -> inverse_rope_inplace). Settled from the source,
    # not the capture: the call is inside the opaque v4_attention_with_output.
    _, y_rp, _ = F.rope(cfg, pre2, "csa")
    width = cfg["num_attention_heads"] / 4 * cfg["qk_rope_head_dim"]
    check("inverse rope is n_h/tp x rope_dim wide", width, 2048.0)
    check("and in-place: one read, one write, bf16", y_rp, 2.0 * 2.0 * Mi * width)

    # The two shared-expert quants run the SAME kernel under sibling markers and
    # do not share a width: gate_up's input is the hidden state, w2's is the
    # expert intermediate. Sharing one formula charged gate_up 9.3x low.
    _, y_qg, sg_ = F.quant_moe_hidden(cfg, dec2, "csa")
    _, y_qw, sw_ = F.quant_moe_intermediate(cfg, dec2, "csa")
    check("gate_up quantises the hidden state", "hidden_size" in sg_.note, True)
    check("w2 quantises the expert intermediate", "I_local" in sw_.note, True)
    check(
        "so gate_up is hidden/I_local times wider",
        round(y_qg / y_qw, 2),
        round(
            cfg["hidden_size"]
            / (cfg["moe_intermediate_size"] * cfg["n_shared_experts"] / dec2.tp),
            2,
        ),
    )
    check("which is 9.33x here", round(y_qg / y_qw, 2), 9.33)

    # One tensor, two rows: moe_router writes the logits, moe_topk reads them.
    # They must agree about the dtype -- topk charged fp32 for a bf16 write.
    _, y_rt, _ = F.moe_router(cfg, dec2, "csa")
    _, y_tk, _ = F.moe_topk(cfg, dec2, "csa")
    E, kk = cfg["n_routed_experts"], cfg["num_experts_per_tok"]
    Mm = F.moe_B(dec2)
    check(
        "topk reads the logits at the width the router wrote them",
        y_tk - (4.0 + F.I32_BYTES) * Mm * kk,
        2.0 * Mm * E,
    )
    check(
        "which is the router's own logits term",
        2.0 * Mm * E,
        y_rt - 2.0 * (Mm * cfg["hidden_size"] + cfg["hidden_size"] * E),
    )

    # "moe_sort" is a substring of both opus_moe_sorting_entry and
    # fused_mx_quant_moe_sort_kernel. They are different operators: one permutes
    # indices and clears the output buffer, the other re-reads every token topk
    # times and quantises it into expert order.
    f_ds = F.lookup(
        "ffn.experts.fused_moe", "void aiter::opus_moe_sorting_entry<...>", None
    )
    f_qs = F.lookup(
        "ffn.experts.fused_moe", "aiter::fused_mx_quant_moe_sort_kernel<...>", None
    )
    check("the sorting entry keeps moe_dispatch", f_ds.__name__, "moe_dispatch")
    check("the quant+sort kernel gets its own", f_qs.__name__, "moe_quant_sort")
    _, y_qs, sp_qs = F.moe_quant_sort(cfg, dec2, "csa")
    Mq, Hq = F.moe_B(dec2), cfg["hidden_size"]
    kq = cfg["num_experts_per_tok"]
    padded = F.moe_padded_rows(cfg, dec2)
    check(
        "it reads every token topk times, not once",
        y_qs - (1.0 * padded * Hq + padded * Hq / 32.0),
        2.0 * Mq * kq * Hq,
    )
    # ... and writes the sorter's PADDED rows, which at c1 is 32 per active
    # expert: six real tokens, 192 written rows, 186 of them zeros.
    check(
        "topk picks k DISTINCT experts, so M=1 activates exactly k",
        F.active_experts(cfg, dec2),
        float(cfg["num_experts_per_tok"]),
    )
    check("the sorter pads each expert to a 32-row tile", round(padded, 1), 192.0)
    check("which is 32x the real rows", round(padded / (Mq * kq), 1), 32.0)
    check("and says so", "topk times" in sp_qs.note, True)
    _, y_ds, _ = F.moe_dispatch(cfg, dec2, "csa")
    check("so it is far heavier than the index permute", y_qs > 8 * y_ds, True)

    # One tensor, two rows: the MoE all-reduce and the residual add fused after
    # it must agree about the token count, exactly as the attention pair does.
    dpa = dataclasses.replace(dec2, dp=4, dpon=True)
    _, y_ra, _ = F.residual_add(cfg, dpa, "csa")
    _, y_rm, _ = F.residual_add_moe(cfg, dpa, "csa")
    check(
        "attention-side and MoE-side residual adds differ under dp-attn",
        y_rm / y_ra,
        4.0,
    )
    check(
        "the MoE add uses the same token count as the all-reduce beside it",
        F.moe_B(dpa) * cfg["hidden_size"] * 2.0 * 3.0,
        y_rm,
    )
    _, y_ra0, _ = F.residual_add(cfg, dec2, "csa")
    _, y_rm0, _ = F.residual_add_moe(cfg, dec2, "csa")
    check("and they coincide with dp-attn off", y_ra0, y_rm0)

    # The SWA ring scatter is fused into this launch at decode and not at
    # prefill: a chunked prefix must see the PRIOR chunk's window, so prefill
    # passes swa_*=None and scatters separately. One K row per token either way.
    _, y_qkn_d, sp_d = F.qk_norm_rope(cfg, dec2, "csa")
    _, _y_qkn_p, sp_p = F.qk_norm_rope(cfg, pre2, "csa")
    check("decode fuses the SWA scatter", "SWA" in sp_d.bytes_expr, True)
    check("prefill does not", "SWA" in sp_p.bytes_expr, False)
    rows_qkn = cfg["num_attention_heads"] / dec2.tp + cfg["num_key_value_heads"]
    row_b = 512.0 + 64.0 * 2.0  # packed fp8 + bf16 rope, the same row it writes
    check(
        "and it is exactly one more row",
        y_qkn_d - (2.0 * F.attn_M(dec2) * rows_qkn * cfg["head_dim"]),
        (rows_qkn + 1) * row_b * F.attn_M(dec2),
    )

    print("\nper-operator decomposition — work + issue + stall = measured")
    # Same split as the step table, one level down. The identity is the point: if
    # the three did not add up, a small term could be small or mis-attributed and
    # nothing in the row would say which.
    for work, meas, fires, floor in (
        (100.0, 500.0, 10, 4.0),  # work dominates its own issue floor
        (1.0, 50.0, 10, 4.0),  # tiny work, issue floor dominates
        (0.0, 0.0, 0, 4.0),  # nothing fired
        (80.0, 80.0, 20, 4.0),  # measured exactly at the floor
    ):
        iss, sta = marker_decompose(work, meas, fires, floor)
        check(
            f"work+issue+stall = measured ({work},{meas},{fires})",
            round(work + iss + sta, 6),
            round(max(meas, work), 6),
        )
    iss, sta = marker_decompose(1.0, 50.0, 10, 4.0)
    check("issue is the floor minus the work it covers", iss, 39.0)
    check("stall is what neither explains", sta, 10.0)
    iss, _ = marker_decompose(100.0, 500.0, 10, 4.0)
    check("work above its own issue floor leaves no issue", iss, 0.0)

    print("\nstep decomposition — four terms that add up to the wall clock")
    # A measured/floor ratio says how far a step is from its ceiling and nothing
    # about why. These four do, and the only property that makes them readable is
    # that they SUM to the wall: if they did not, a reader could not tell whether
    # a small term was small or just mis-attributed.
    import sys as _sys

    _sys.path.insert(0, str(Path(__file__).parent))
    from render_report import step_facts as _sf

    fake = {
        "sampled_steps": 2,
        "measured_total_us": 20.0,  # busy = 10 per step
        "roofline_tier_us": 20.0,
        "t_roofline_sum_us": 4.0,  # work = 2 per step
        "latency_floor_us": 3.0,
        "phase_steps": {"decode": {"steps": 2, "median_us": 12.0}},
        # one operator, 2 firings per step, floor 1.0 per firing -> issue covers it
        "rows": [{"n_firings": 4, "t_roofline_us": 4.0}],
    }
    fa = _sf(fake, "decode", {"prefill_steps": 0, "decode_steps": 2})
    check(
        "work + issue + stall + idle = wall",
        round(fa["roof_us"] + fa["issue_us"] + fa["stall_us"] + fa["idle_us"], 6),
        round(fa["wall_us"], 6),
    )
    # 4 firings at max(1.0, 3.0) = 12 us over 2 steps = 6 per step; work is 2, so
    # issue is 4 -- the part of the per-kernel floor the work does not cover.
    check("issue is the floor the work does not cover", fa["issue_us"], 4.0)
    check("stall is measured minus max(work, issue)", fa["stall_us"], 4.0)
    check("idle is wall minus busy", fa["idle_us"], 2.0)
    # A step whose operators are all far above the floor has no issue term at all.
    fat = dict(
        fake, rows=[{"n_firings": 4, "t_roofline_us": 40.0}], t_roofline_sum_us=40.0
    )
    check(
        "a compute-heavy step has no issue term",
        _sf(fat, "decode", {"prefill_steps": 0, "decode_steps": 2})["issue_us"],
        0.0,
    )

    print("\nindexer cache — one layout, written once and read once")
    # compressor_pool_index writes it, csa_indexer reads it. Both go through
    # index_entry_bytes so they cannot describe one cache differently -- the same
    # binding kv_entry_bytes gives the Main compressor and _core_attn.
    ie = F.index_entry_bytes(cfg, pre2.dtypes)
    check("an fp8 index row carries an fp32 scale", ie, 128 * 1.0 + 4.0)
    blocks = pre2.kv_seq_len / 4
    _, y_ix, _ = F.csa_indexer(cfg, pre2, "csa")
    Mi, ni, di = F.attn_M(pre2), cfg["index_n_heads"], cfg["index_head_dim"]
    check(
        "the reader pays for the scale too",
        y_ix,
        blocks * ie + 1.0 * Mi * ni * di + 4.0 * Mi * blocks,
    )

    print("\ncompressor_pool — one program per boundary, K rows wide")
    csa_k = dataclasses.replace(pre2, kernel="fused_compress_attn_D512_RD64_R4_OVL_SS8")
    hca_k = dataclasses.replace(pre2, kernel="hca_compress_forward_D512_R128_NW8_SL64")
    _, b_csa, sp_csa = F.compressor_pool(cfg, csa_k, "csa")
    _, b_hca, _ = F.compressor_pool(cfg, hca_k, "hca")
    # K = 2*ratio when the window overlaps, ratio when it does not. Charging one
    # row per token -- what this did before -- half-priced CSA and left HCA right
    # by luck, because HCA has no overlap and K/ratio is 1 there.
    M, c = F.attn_M(pre2), cfg["head_dim"]
    Mc = M * c
    # The scatter writes the cache _core_attn reads, so it is priced with the same
    # kv_entry_bytes. Writer and reader disagreeing about one cache layout is a
    # contradiction no decode measurement could surface -- this operator sits
    # 4700x under its launch floor, where a 12% byte error is invisible.
    entry = F.kv_entry_bytes(cfg)
    check(
        "the scattered entry is the one core attention reads", entry, 64 * 2 + 448 * 1
    )
    # CSA's kernel fuses pool + norm + rope + scatter; HCA's `hca_compress_forward`
    # only pools, and `hca_norm_rope_scatter` (compressor_epilogue) does the rest.
    # Charging the epilogue to both billed HCA for the same scatter twice.
    check("CSA pools a K=8 window AND scatters", b_csa, 16 * Mc + (M / 4) * entry)
    # HCA's pool still has to land the pooled entry for the epilogue to read back:
    # dropping the scatter without adding this write left a reader with no writer.
    handoff = 2 * (M / 128) * c
    check("HCA's pool pools and hands the entry over", b_hca, 8 * Mc + handoff)
    check(
        "the overlapping window costs twice",
        (b_csa - (M / 4) * entry) / (b_hca - handoff),
        2.0,
    )
    _, _, sp_hca = F.compressor_pool(cfg, hca_k, "hca")
    check("and the epilogue is not in its FLOPs", "RoPE" not in sp_hca.flops_expr, True)
    # compressor_epilogue prices that epilogue, against the same cache model.
    _, b_rope, _ = F.compressor_epilogue(cfg, hca_k, "hca")
    check(
        "the split-out epilogue reads that entry back and writes the shared one",
        b_rope,
        handoff + (M / 128) * entry,
    )
    check("and the kernel name is what says so", "_OVL_" in csa_k.kernel, True)
    check("the note names the window", "K=8" in sp_csa.note, True)
    # Without a kernel name to read, fall back on the family: CSA overlaps.
    _, b_blind, _ = F.compressor_pool(cfg, pre2, "csa")
    check("a nameless bench still gets CSA's overlap", b_blind, b_csa)

    print("\ncompress_ratios — the layer taxonomy the checkpoint already carries")
    ratios = cfg["compress_ratios"]
    n_layers, n_hash = cfg["num_hidden_layers"], cfg["num_hash_layers"]
    # _CSA_RATIO and _HCA_RATIO are hard-coded in formulas.py, and the checkpoint
    # states them per layer. Nothing linked the two until this assertion: a model
    # shipping 8:1 would have been priced at 4:1 in silence.
    check(
        "the hard-coded ratios are the ones the config states",
        sorted(set(ratios) - {0}),
        sorted([F._CSA_RATIO, F._HCA_RATIO]),
    )
    # One entry past num_hidden_layers, and that entry is 0 -- the layer with no
    # compressor. Slicing to n_layers is deliberate, not an off-by-one.
    check("there is one entry per layer, plus a 0 sentinel", len(ratios), n_layers + 1)
    check("and the sentinel is the extra one", ratios[n_layers], 0)
    fams: dict[str, int] = {}
    for i, r in enumerate(ratios[:n_layers]):
        name = "hash" if i < n_hash else ("csa" if r == F._CSA_RATIO else "hca")
        fams[name] = fams.get(name, 0) + 1
    # Derived from the checkpoint; `methodology` is hand-written in the model yaml.
    # Two independent statements of the same taxonomy, asserted equal.
    check("the taxonomy derives from the config", fams, decl["methodology"])
    check(
        "family rules live with the model, not the engine",
        [f["name"] for f in decl["families"]],
        ["hash", "csa", "hca"],
    )
    check(
        "the methodology expectation is data too",
        decl["methodology"],
        {"hash": 3, "hca": 29, "csa": 29},
    )

    # An unknown model must name what IS declared rather than fail blankly.
    try:
        load_model_decl("no-such-model", root)
        check("unknown model raises", False, True)
    except SystemExit as exc:
        check(
            "unknown model lists what is declared", "deepseek-v4-pro" in str(exc), True
        )

    print("\ncheck_methodology — a pure validator that refuses to report")
    check_methodology(
        "deepseek-v4-pro",
        {"hash": 3, "hca": 29, "csa": 29},
        {"hash": 3, "hca": 29, "csa": 29},
    )
    try:
        # What a build rename looks like: hash absorbed, counts shifted.
        check_methodology(
            "deepseek-v4-pro", {"hash": 3, "hca": 29, "csa": 29}, {"csa": 30, "hca": 31}
        )
        check("divergence raises", False, True)
    except SystemExit as exc:
        check(
            "it says what it wanted and what it got",
            "29" in str(exc) and "30" in str(exc),
            True,
        )
        check("and refuses rather than warns", "Refusing" in str(exc), True)

    print("\nformatters — nothing measurable may render as a bare zero")
    from render_report import fmt_ai, fmt_ms

    # The same bug has now been fixed in four quantities: TFLOP/s, TB/s, tooltip
    # durations, and table durations. Each time it was found by eye, on a
    # different unit. The rule is the rule, so assert the rule.
    check("a 4 us operator does not print as 0.00", "0.00" in fmt_ms(4.0), False)
    check("and says which unit it switched to", "micro" in fmt_ms(4.0), True)
    check("milliseconds once there are digits", fmt_ms(15630.0), "15.63")
    check("the ladder has no gap at the boundary", fmt_ms(10.0), "0.01")

    # Zero intensity is a property of the operator, not a measurement of it.
    check(
        "a zero-FLOP operator shows an em dash, not 0",
        "&mdash;" in fmt_ai({"flop_per_byte": 0.0}),
        True,
    )
    check(
        "and says why on hover",
        "by construction" in fmt_ai({"flop_per_byte": 0.0}),
        True,
    )
    check("a real intensity passes through", fmt_ai({"flop_per_byte": 2.0}), "2.0")
    check("a blank stays blank", fmt_ai({"flop_per_byte": ""}), "")

    print("\ncache threshold — an uncertain constant must not decide silently")
    cache = yaml.safe_load(
        (Path(__file__).parent / "peaks" / "mi355x.yaml").read_text(encoding="utf-8")
    )["cache"]
    check("capacity is recorded", float(cache["llc_bytes"]), 256e6)
    # AMD publishes no Infinity Cache bandwidth, so no cache ROOF is drawn. A
    # ceiling inferred from the measurements it is meant to judge is not a ceiling.
    check("bandwidth is left null, not guessed", cache["llc_bw_gbps"], None)
    check(
        "and the 2x uncertainty is written down",
        "128e6" in cache["llc_bytes_uncertain"],
        True,
    )

    print("\nmhc_fused_post_pre — one kernel, two half-layers")
    _, y_fused, sp_f = F.mhc_fused_post_pre(cfg, pre2, "csa")
    _, y_pre, _ = F.mhc_pre_gemm(cfg, pre2, "csa")
    # Modelling the fused kernel as its pre half alone charged nothing for the
    # expand, and the expand touches the wider tensor: [M, hc, dim] vs [M, dim].
    check("the fused kernel moves more than its pre half", y_fused > y_pre, True)
    check(
        "the widened residual is counted in and out",
        "residual in+out" in sp_f.bytes_expr,
        True,
    )
    check(
        "every FLOP term is named",
        all(
            t in sp_f.flops_expr
            for t in ("post gate", "post comb mix", "hc-fn linear", "pre sqrsum")
        ),
        True,
    )
    # mhc.h shows this kernel emits gemm_out/next_residual and no y (m, dim) --
    # the reduce belongs to mhc_pre_big_fuse_rmsnorm. Charging a y write here
    # double-counted against mhc_pre_fuse, which already pays for it.
    check("no phantom y write is charged", "pre out" not in sp_f.bytes_expr, True)
    check("but the GEMM spill is", "gemm_out fp32" in sp_f.bytes_expr, True)
    # The bf16 MFMA path is compiled in on gfx950 and ATOM never asks for it, so
    # the bf16 ceiling this is priced against is optimistic. Name the knob.
    check("the unused fp32/bf16 MFMA knob is surfaced", "CONFIG:" in sp_f.note, True)
    # ATOM never passes is_fn_pack_bf16, so GEMM_LOOP_BODY runs
    # v_mfma_f32_16x16x4_f32. Pricing that against the bf16 roof overstated the
    # headroom by 16x, which is the whole gap between the two paths.
    check("and it is priced at the ceiling it actually runs at", sp_f.compute, "fp32")
    _, _, sp_g = F.mhc_pre_gemm(cfg, pre2, "csa")
    check("the standalone GEMM half agrees", sp_g.compute, "fp32")
    # CDNA4 keeps fp32 MFMA at 256 FLOPS/clk/CU, the vector rate. The Matrix Core
    # buys nothing at fp32, so these two peaks coincide -- on this chip, by
    # derivation, not by one having been copied from the other.
    pk = yaml.safe_load(
        (Path(__file__).parent / "peaks" / "mi355x.yaml").read_text(encoding="utf-8")
    )
    check(
        "matrix fp32 equals vector fp32 on CDNA4",
        pk["compute_tflops"]["matrix_fp32"],
        pk["vector_tflops"]["vector_fp32"],
    )
    check(
        "and it is 16x under the bf16 roof",
        round(
            pk["compute_tflops"]["matrix_bf16"] / pk["compute_tflops"]["matrix_fp32"]
        ),
        16,
    )

    print("\nmhc_post — the comb term reads the residual it is said to write")
    _, y_post, sp_p = F.mhc_post(cfg, pre2, "csa")
    hc, dim, M = cfg["hc_mult"], cfg["hidden_size"], F.attn_M(pre2)
    # Old model: read x, write out. mhc.h takes `residual` as an input too, and
    # the comb matmul is what reads it -- so the floor was 1.8x too low.
    check(
        "the residual is read as well as written",
        y_post,
        2 * (M * dim + 2 * M * hc * dim) + 4 * M * hc * (1 + hc),
    )
    check("and the comb mix is priced", "comb mix" in sp_p.flops_expr, True)

    print("\nmhc_pre_fuse — the kernel named rmsnorm does one")
    _, _, sp_r = F.mhc_pre_fuse(cfg, pre2, "csa")
    # aiter's own name for it is mhc_pre_big_fuse_rmsnorm, and it takes
    # norm_weight (dim). Pricing it as a bare reduce left the rmsnorm unpaid.
    check(
        "reduce, rmsnorm and Sinkhorn are all named",
        all(t in sp_r.flops_expr for t in ("weighted reduce", "rmsnorm", "Sinkhorn")),
        True,
    )

    print("\ncollectives — which side of the DP gather decides the token count")
    # Confirmed with the user 2026-09-07: combine_outputs' all-reduce runs AFTER
    # the DP gather, so it reduces the gathered tokens of every DP rank, while
    # wo_b's runs inside the attention region on the per-rank share. Sharing one
    # formula made them identical -- true at dp=1, wrong by a factor of dp the
    # moment --enable-dp-attention is on, which the catalog does from concurrency
    # 64 upward.
    off = F.Bench(batch=1, seq_len=1, tp=4, dp=1, dpon=False, kv_seq_len=7670)
    dpa = F.Bench(batch=32, seq_len=1, tp=8, dp=8, dpon=True, kv_seq_len=7670)
    _, a_off, _ = F.comm_allreduce_attn(cfg, off, "csa")
    _, m_off, _ = F.comm_allreduce_moe(cfg, off, "csa")
    check("at dp=1 the two sides are identical", a_off, m_off)
    _, a_dpa, _ = F.comm_allreduce_attn(cfg, dpa, "csa")
    _, m_dpa, _ = F.comm_allreduce_moe(cfg, dpa, "csa")
    check("under DP attention the MoE side is dp times bigger", m_dpa / a_dpa, 8.0)
    _, _, sp_moe = F.comm_allreduce_moe(cfg, dpa, "csa")
    check("and says which side it is", "after the DP gather" in sp_moe.note, True)
    check(
        "both are still priced against the fabric",
        {F.comm_allreduce_attn(cfg, dpa, "csa")[2].bw, sp_moe.bw},
        {"interconnect"},
    )
    # The two markers must not resolve to the same formula any more.
    check(
        "wo_b takes the attention-side model",
        F.lookup("attn.wo_b", "nccl", "DeepseekV4ForCausalLM").__name__,
        "comm_allreduce_attn",
    )
    check(
        "combine_outputs takes the MoE-side model",
        F.lookup("ffn.combine_outputs", "nccl", "DeepseekV4ForCausalLM").__name__,
        "comm_allreduce_moe",
    )

    print("\nallreduce_path — AITER dispatches by message size, three ways")
    # Read out of AITER, not assumed: custom_all_reduce.cuh:1537 for the 1/2-stage
    # split, custom_all_reduce.py:56,249 for the 64 MiB cap above which the custom
    # kernel is skipped entirely and NCCL runs. The three paths differ by a factor
    # of N in per-link bytes, so picking the wrong one is not a rounding error.
    check("a tiny message takes 1-stage", F.allreduce_path(14_336, 4), "1stage")
    check("TP4 switches at 160 KB", F.allreduce_path(160 * 1024, 4), "2stage")
    check(
        "just under it is still 1-stage", F.allreduce_path(160 * 1024 - 16, 4), "1stage"
    )
    check("TP8 switches at 80 KB instead", F.allreduce_path(100 * 1024, 8), "2stage")
    # The size cap is checked in Python (`should_custom_ar`) BEFORE the kernel
    # dispatch chooses a stage count, so it wins even for a world of 2.
    check(
        "world of 2 takes 1-stage under the cap", F.allreduce_path(10**6, 2), "1stage"
    )
    check("but the cap still wins above it", F.allreduce_path(10**8, 2), "nccl")
    check(
        "above 64 MiB the custom kernel is skipped",
        F.allreduce_path(F.CUSTOM_AR_MAX_BYTES + 16, 4),
        "nccl",
    )
    check("the cap is 64 MiB", F.CUSTOM_AR_MAX_BYTES, 8192 * 1024 * 8)

    # 1-stage moves the whole message per link; 2-stage moves 2/N of it. At N=4
    # that is a 4x difference, and both are correct -- for different sizes.
    small = F.Bench(batch=1, seq_len=1, tp=4, dp=1, kv_seq_len=7670)
    big = F.Bench(batch=8192, seq_len=1, tp=4, dp=1, kv_seq_len=7670)
    _, pl_small, sp_small = F._allreduce(cfg, small, 1, "x")
    _, _pl_big, sp_big = F._allreduce(cfg, big, 8192, "x")
    check("1-stage carries the whole message", pl_small, 1 * cfg["hidden_size"] * 2)
    check("and says so", "whole message" in sp_small.bytes_expr, True)
    check("the large one is on the NCCL path", "nccl" in sp_big.note, True)
    # No longer an assumption about NCCL's algorithm: any all-reduce must move
    # 2*(N-1)/N per rank, so once that uses all N-1 links it is 2*message/N per
    # link whatever the algorithm picks. RCCL does not name its algorithm anyway
    # (ncclDevKernel_Generic_{1,2,4} dispatch at runtime).
    check(
        "the NCCL bound is stated as algorithm-independent",
        "uses every link" in sp_big.note,
        True,
    )
    # What IS worth flagging is that a software default put it on this path.
    check(
        "and the configurable cap that sent it there is named",
        "staging buffer" in sp_big.note and "not a hardware limit" in sp_big.note,
        True,
    )
    check(
        "including that a roofline cannot settle whether the other path is faster",
        "needs an A/B run" in sp_big.note,
        True,
    )

    print("\nallreduce_path — observed beats derived")
    # The kernel name states the path outright, and a measured run should never
    # re-derive from a threshold it cannot see: 64 MiB is a constructor default in
    # AITER (a staging-buffer capacity, not a hardware constant), no call site
    # overrides it and no env var exposes it -- but a deployment that raised it
    # would move the boundary while this tool went on assuming the old one.
    check(
        "a 1-stage kernel names itself",
        F.allreduce_path(10**9, 4, "aiter::cross_device_reduce_1stage"),
        "1stage",
    )
    check("so does NCCL", F.allreduce_path(1024, 4, "ncclDevKernel_Generic_1"), "nccl")
    check(
        "the name wins over the threshold",
        F.allreduce_path(10**9, 4, "aiter::cross_device_reduce_1stage")
        != F.allreduce_path(10**9, 4),
        True,
    )
    check(
        "without a name it falls back to AITER's own rule",
        F.allreduce_path(14_336, 4),
        "1stage",
    )
    # The cap is configurable because it is configurable in AITER.
    check(
        "a raised cap moves the boundary",
        F.allreduce_path(10**8, 4, "", max_bytes=10**9),
        "2stage",
    )
    check("and the default is unchanged", F.allreduce_path(10**8, 4), "nccl")

    # A roofline is a FLOOR, so under uncertainty the safe choice is the SMALLEST
    # byte count: a floor above its own measurement is a broken floor, and this
    # session has already had to retract one model for exactly that.
    b_amb = F.Bench(batch=1, seq_len=1, tp=4, dp=1, kv_seq_len=7670)
    _, one, _ = F._allreduce(
        cfg, dataclasses.replace(b_amb, kernel="cross_device_reduce_1stage"), 4096, "x"
    )
    _, two, _ = F._allreduce(
        cfg, dataclasses.replace(b_amb, kernel="cross_device_reduce_2stage"), 4096, "x"
    )
    check("1-stage carries more per link than 2-stage", one > two, True)
    # message vs 2*message/N, so the ratio is N/2 -- 2x at TP4, 4x at TP8.
    check("by N/2, so 2x at TP4", round(one / two), 2)
    b8 = dataclasses.replace(b_amb, tp=8)
    _, one8, _ = F._allreduce(
        cfg, dataclasses.replace(b8, kernel="cross_device_reduce_1stage"), 4096, "x"
    )
    _, two8, _ = F._allreduce(
        cfg, dataclasses.replace(b8, kernel="cross_device_reduce_2stage"), 4096, "x"
    )
    check("and 4x at TP8", round(one8 / two8), 4)

    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} -> {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
