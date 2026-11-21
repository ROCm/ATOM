#!/usr/bin/env python3
"""Tune aiter FlyDSL conv3d tiles for the shapes its shipped tables do not cover.

ROCm/aiter#5370 ships per-model tuned tables keyed by the exact convolution the
caller makes. Its Wan2.1 rows were tuned for the stock causal-conv path, which
``F.pad``s every input and then convolves with ``padding=0`` (e.g. C=96,
6x370x546, pad 0). ``conv_pad_in_kernel`` makes the same convolution as
6x368x544 with ``padding=(0, 1, 1)``, so none of those rows match and every call
falls back to the heuristic tile. This produces the missing rows, in the PR's
CSV format, so aiter's own loader picks them up.

Two steps:

    # 1. record: encode once through the patched VAE, keep the shapes the
    #    table missed (veomni_patches/aiter_conv.py counts every lookup)
    python tune_conv3d.py record --model wan --padk -o untuned.csv

    # 2. tune: sweep candidates on every GPU, one process per GPU
    python tune_conv3d.py tune -i untuned.csv -o wan21_vae_padk_bf16_tuned_conv3d.csv

Method, copied from the PR's csrc/flydsl_conv3d/conv3d_tune.py so the rows are
comparable to the shipped ones rather than a second opinion on them:

* candidates are ``conv3d_policy.get_flydsl_conv3d_configs`` (which unions the
  heuristic ladder in), split-K derived by ``_resolve_splitk``, not swept;
* every candidate is timed NDHWC in and NDHWC out with ``run_perftest``
  (rotating inputs, kernel time from the profiler), and checked against
  ``F.conv3d`` in BF16 at rtol = atol = 2e-2 with no element allowed over;
* a row is written only if its best candidate beats the heuristic's own pick
  -- the production entry point with no tile forced, best of 3 -- by at least
  ``--min-gain`` (3%, the PR's ``--update_improved`` bar). A shape that does
  not clear it keeps the heuristic, as the PR does.

Needs the aiter overlay (overlay_aiter.py --apply) and the sidecar flydsl.
"""

import argparse
import csv
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))

KEY_COLUMNS = (
    "N", "C", "D", "H", "W", "K", "kT", "kH", "kW",
    "stride_d", "stride_h", "stride_w", "pad_d", "pad_h", "pad_w",
    "dil_d", "dil_h", "dil_w", "groups", "bias",
)  # fmt: skip
TUNED_COLUMNS = (
    "gfx", "cu_num", *KEY_COLUMNS, "libtype", "tile_m", "tile_n", "wave_m", "wave_n", "wgm",
    "splitK", "us", "kernelName", "err_ratio", "tflops", "bw",
)  # fmt: skip
RTOL = ATOL = 2e-2
BASELINE_REPS = 3


def _conv_kernels():
    from veomni_patches import aiter_conv

    if aiter_conv.load() is None:
        raise SystemExit(
            "aiter flydsl_conv_implicit unavailable; run overlay_aiter.py --apply"
        )
    return sys.modules["aiter.ops.flydsl.conv_kernels"]


def _policy():
    """conv3d_policy, which asks chip_info for an LDS size older aiter cannot give."""
    import importlib

    chip_info = importlib.import_module("aiter.jit.utils.chip_info")
    if not hasattr(chip_info, "get_lds_capacity_bytes"):
        # The PR's table value for the only arch this kernel supports.
        lds = {"gfx950": 160 * 1024}

        def get_lds_capacity_bytes(gfx=None):
            return lds[(gfx or chip_info.get_gfx()).split(":", 1)[0].lower()]

        chip_info.get_lds_capacity_bytes = get_lds_capacity_bytes
    _conv_kernels()
    from veomni_patches import aiter_conv

    return aiter_conv.import_flydsl_module("conv3d_policy")


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------


def record(args):
    import torch

    sys.path.insert(0, _HERE)
    import diffusers
    from lumen_vae_conv import patch_vae_convs
    from veomni_patches import aiter_conv

    _conv_kernels()
    if args.model == "wan":
        path = os.environ.get("WAN_DIR", "/work/models-extra/Wan2.1-T2V-1.3B")
        vae = diffusers.AutoencoderKLWan.from_pretrained(
            path, subfolder="vae", torch_dtype=torch.bfloat16
        )
        shape = (1, 3, args.frames, args.height or 368, args.width or 544)
    else:
        path = os.environ.get("QWEN_IMAGE_DIR", "/work/models/Qwen-Image")
        vae = diffusers.AutoencoderKLQwenImage.from_pretrained(
            path, subfolder="vae", torch_dtype=torch.bfloat16
        )
        shape = (1, 3, 1, args.height or 1024, args.width or 1024)
    vae = vae.cuda().eval()
    patch_vae_convs(vae, video=args.model == "wan", causal_pad_in_kernel=args.padk)
    x = torch.rand(shape, device="cuda", dtype=torch.bfloat16) * 2 - 1
    with torch.no_grad():
        vae.encode(x)
    torch.cuda.synchronize()
    misses = aiter_conv.tuned_misses()
    print(f"lookups over one encode of {shape}: {aiter_conv.tuned_lookup_stats()}")
    with open(args.output, "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(KEY_COLUMNS)
        for key in misses:
            wr.writerow(key)
    print(f"wrote {len(misses)} untuned shapes -> {args.output}")


# ---------------------------------------------------------------------------
# tune
# ---------------------------------------------------------------------------


def _read_rows(path):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for r in rows:
        key = tuple(
            (str(r[c]).strip().lower() in ("1", "true")) if c == "bias" else int(r[c])
            for c in KEY_COLUMNS
        )
        out.append(key)
    return out


def _tune_one(key, max_configs, log):
    import torch
    import torch.nn.functional as F
    from aiter.jit.utils.chip_info import get_cu_num, get_gfx
    from aiter.test_common import run_perftest

    ck = _conv_kernels()
    policy = _policy()
    kv = dict(zip(KEY_COLUMNS, key))
    n, c, d, h, w, k = (kv[x] for x in ("N", "C", "D", "H", "W", "K"))
    kt, kh, kw, groups, has_bias = (
        kv["kT"],
        kv["kH"],
        kv["kW"],
        kv["groups"],
        kv["bias"],
    )
    params = {
        "stride": (kv["stride_d"], kv["stride_h"], kv["stride_w"]),
        "padding": (kv["pad_d"], kv["pad_h"], kv["pad_w"]),
        "dilation": (kv["dil_d"], kv["dil_h"], kv["dil_w"]),
        "groups": groups,
    }
    oe = ck.out_extent
    do = oe(d, kv["pad_d"], kv["dil_d"], kt, kv["stride_d"])
    ho = oe(h, kv["pad_h"], kv["dil_h"], kh, kv["stride_h"])
    wo = oe(w, kv["pad_w"], kv["dil_w"], kw, kv["stride_w"])
    m_gemm, n_gemm = n * do * ho * wo, k // groups
    k_gemm = (c // groups) * kt * kh * kw
    cu = get_cu_num()

    torch.manual_seed(0)
    x = torch.randn((n, d, h, w, c), device="cuda", dtype=torch.bfloat16)
    wt = torch.randn((k, c // groups, kt, kh, kw), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((k,), device="cuda", dtype=torch.float32) if has_bias else None
    ref = F.conv3d(
        x.permute(0, 4, 1, 2, 3).contiguous(),
        wt,
        bias.to(x.dtype) if has_bias else None,
        **params,
    )
    ref = ref.permute(0, 2, 3, 4, 1).contiguous()

    def conv(**kw_):
        return ck.flydsl_conv_implicit(
            x, wt, bias, input_layout="NDHWC", output_layout="NDHWC", **params, **kw_
        )

    def timed(**kw_):
        return run_perftest(
            ck.flydsl_conv_implicit,
            x,
            wt,
            bias,
            input_layout="NDHWC",
            output_layout="NDHWC",
            **params,
            **kw_,
        )[1]

    conv()
    base_us = min(timed() for _ in range(BASELINE_REPS))
    heur_tile = ck._pick_tile(m_gemm, k, groups, x.device)
    heur_wgm = ck._pick_wgm(m_gemm, k, groups, heur_tile, x.device)

    cgp = ck._pad_channels(c // groups)
    crs = cgp * kt * kh * kw
    best = None
    configs = policy.get_flydsl_conv3d_configs(
        m_gemm, n_gemm, groups, cu, max_configs=max_configs
    )
    t0 = time.time()
    for tile_m, tile_n, wave_m, wave_n, wgm in configs:
        tile = (tile_m, tile_n, wave_m, wave_n)
        sk = ck._resolve_splitk(None, m_gemm, crs, k, None, tile, groups, num_cu=cu)
        try:
            y = conv(tile=tile, wgm=wgm, splitk=sk)
            bad = ~torch.isclose(y.float(), ref.float(), rtol=RTOL, atol=ATOL)
            err = bad.float().mean().item()
            if err > 0:
                continue
            us = timed(tile=tile, wgm=wgm, splitk=sk)
        except RuntimeError as exc:
            log(f"    skip {tile} wgm={wgm}: {type(exc).__name__}: {str(exc)[:80]}")
            continue
        if best is None or us < best[0]:
            best = (us, tile, wgm, sk)
    gain = (base_us - best[0]) / base_us if best else 0.0
    log(
        f"  {key}: M={m_gemm} N={n_gemm} K={k_gemm}  {len(configs)} candidates in {time.time() - t0:.0f}s  "
        f"heuristic {heur_tile} wgm={heur_wgm} {base_us:.1f} us  best "
        f"{best[1] if best else None} wgm={best[2] if best else None} {best[0] if best else float('nan'):.1f} us  "
        f"gain {gain * 100:+.1f}%"
    )
    if best is None:
        return None, gain
    us, tile, wgm, sk = best
    tflops = round(m_gemm * n_gemm * k_gemm * 2 / (us * 1e6), 2)
    moved = (
        n * c * d * h * w + k * (c // groups) * kt * kh * kw + n * k * do * ho * wo
    ) * 2
    row = {
        "gfx": get_gfx(), "cu_num": cu, **kv, "libtype": "flydsl",
        "tile_m": tile[0], "tile_n": tile[1], "wave_m": tile[2], "wave_n": tile[3], "wgm": wgm, "splitK": sk,
        "us": round(us, 4), "kernelName": policy.tile_kernel_name(*tile, wgm), "err_ratio": 0.0,
        "tflops": tflops, "bw": round(moved / (us * 1e-6) / 1e9, 2),
        "_base_us": round(base_us, 4),
    }  # fmt: skip
    return row, gain


def worker(args):
    keys = _read_rows(args.input)
    mine = [keys[int(i)] for i in args.rows.split(",") if i != ""]
    out = []
    with open(args.log, "a") as logf:

        def log(msg):
            print(msg, flush=True)
            logf.write(msg + "\n")
            logf.flush()

        for key in mine:
            row, gain = _tune_one(key, args.max_configs, log)
            if row is not None:
                row["_gain"] = round(gain, 4)
                out.append(row)
    with open(args.part, "w", newline="") as fh:
        wr = csv.DictWriter(
            fh, fieldnames=[*TUNED_COLUMNS, "_base_us", "_gain"], lineterminator="\n"
        )
        wr.writeheader()
        wr.writerows(out)


def tune(args):
    keys = _read_rows(args.input)
    gpus = [g for g in args.gpus.split(",") if g != ""]
    work = {g: [] for g in gpus}
    for i in range(len(keys)):
        work[gpus[i % len(gpus)]].append(str(i))
    tmp = f"{args.output}.parts"
    os.makedirs(tmp, exist_ok=True)
    procs = []
    for g, rows in work.items():
        if not rows:
            continue
        env = dict(os.environ, HIP_VISIBLE_DEVICES=g, CUDA_VISIBLE_DEVICES=g)
        cmd = [
            sys.executable, os.path.abspath(__file__), "_worker", "-i", args.input, "--rows", ",".join(rows),
            "--part", f"{tmp}/gpu{g}.csv", "--log", f"{tmp}/gpu{g}.log", "--max-configs", str(args.max_configs),
        ]  # fmt: skip
        procs.append((g, subprocess.Popen(cmd, env=env)))
    failed = [g for g, p in procs if p.wait() != 0]
    if failed:
        raise SystemExit(f"tuning failed on GPU(s) {failed}; see {tmp}/gpu*.log")

    rows = []
    for g, _ in procs:
        with open(f"{tmp}/gpu{g}.csv") as fh:
            rows.extend(csv.DictReader(fh))
    order = {k: i for i, k in enumerate(keys)}
    rows.sort(key=lambda r: order[tuple(_read_key(r))])
    kept = [r for r in rows if float(r["_gain"]) >= args.min_gain]
    with open(args.output, "w", newline="") as fh:
        wr = csv.DictWriter(
            fh, fieldnames=TUNED_COLUMNS, extrasaction="ignore", lineterminator="\n"
        )
        wr.writeheader()
        wr.writerows(kept)
    print(
        f"\n{len(keys)} shapes, {len(rows)} tuned, {len(kept)} cleared the {args.min_gain * 100:.0f}% bar -> {args.output}"
    )
    for r in rows:
        mark = "kept" if r in kept else "heuristic stays"
        print(
            f"  C={r['C']} {r['D']}x{r['H']}x{r['W']} K={r['K']} k={r['kT']}{r['kH']}{r['kW']}: "
            f"{float(r['_base_us']):8.1f} -> {float(r['us']):8.1f} us ({float(r['_gain']) * 100:+5.1f}%)  {mark}"
        )


def _read_key(r):
    return tuple(
        (str(r[c]).strip().lower() in ("1", "true")) if c == "bias" else int(r[c])
        for c in KEY_COLUMNS
    )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser(
        "record", help="write the shapes one encode missed in the tuned table"
    )
    r.add_argument("--model", choices=("wan", "qwen"), required=True)
    r.add_argument("--padk", action="store_true", help="with conv_pad_in_kernel")
    r.add_argument("--frames", type=int, default=81)
    r.add_argument("--height", type=int, default=None)
    r.add_argument("--width", type=int, default=None)
    r.add_argument("-o", "--output", required=True)
    t = sub.add_parser(
        "tune", help="tune the shapes of an untuned CSV, one process per GPU"
    )
    t.add_argument("-i", "--input", required=True)
    t.add_argument("-o", "--output", required=True)
    t.add_argument(
        "--gpus", default=os.environ.get("HIP_VISIBLE_DEVICES", "0,1,2,3,4,5,6,7")
    )
    t.add_argument("--max-configs", type=int, default=96)
    t.add_argument("--min-gain", type=float, default=0.03)
    wk = sub.add_parser("_worker")
    wk.add_argument("-i", "--input", required=True)
    wk.add_argument("--rows", required=True)
    wk.add_argument("--part", required=True)
    wk.add_argument("--log", required=True)
    wk.add_argument("--max-configs", type=int, default=96)
    args = ap.parse_args()
    {"record": record, "tune": tune, "_worker": worker}[args.cmd](args)


if __name__ == "__main__":
    main()
