# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Wan2.1 VAE on GPU: AITER conv3d vs torch conv3d vs Diffusers.

Validates and times ATOM's Wan2.1 VAE (``atom.diffusion.models.wan``) on the
real checkpoint. For every resolution (default 480x832 and 368x544, 81
frames) and every conv3d backend (default ``flydsl,triton,torch``):

1. Encodes a deterministic synthetic clip (moving sinusoids plus a little
   noise, in [-1, 1]) and decodes a fixed latent -- the torch backend's
   encoding of that clip, so every backend decodes the same input -- timing
   each stage after ``--warmup`` untimed runs (the first FlyDSL call of a
   shape JIT-compiles it; warmup keeps that out of the numbers).
2. Reports which backend and layout every conv actually ran on. A backend
   that is unsupported on this GPU falls a rung (see
   ``ATOM_DIFFUSION_CONV3D_BACKEND`` in docs/environment_variables.md), and
   the table says so rather than timing the fallback under the wrong name.
3. Compares each backend's latents and frames with the torch backend's and
   with Diffusers' ``AutoencoderKLWan`` (``--ref-dtype``, fp32 by default):
   PSNR (frames over their [-1, 1] range, latents over the reference's
   range) and max abs diff. Reconstruction PSNR against the input clip is
   only a coarse sign that the weights loaded, not an accuracy metric: the
   VAE is lossy, more so on small canvases.
4. ``--breakdown`` adds one run with CUDA events around every top-level
   block (``conv_in``, each ``down_blocks`` / ``up_blocks`` entry,
   ``mid_block``, ...), summed over all temporal chunks.
5. ``--profile DIR`` wraps one encode + decode per backend in
   ``torch.profiler``, writes a Chrome trace per run to DIR and prints the
   top kernels by self device time.

Exits 1 if any AITER backend's decoded frames are below ``--min-psnr``
against the torch backend.

Run on an MI355X (gfx950), inside ``rocm/atom-dev``::

    export AITER_LOG_LEVEL=WARNING
    hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --include "vae/*" \\
        --local-dir /data/models/Wan2.1-T2V-1.3B-Diffusers
    python -m atom.diffusion.examples.wan_vae_bench \\
        --model /data/models/Wan2.1-T2V-1.3B-Diffusers --breakdown \\
        --json /data/outputs/wan_vae_bench.json

    # kernel-level traces, one resolution:
    python -m atom.diffusion.examples.wan_vae_bench \\
        --model /data/models/Wan2.1-T2V-1.3B-Diffusers \\
        --resolutions 480x832 --profile /data/outputs/wan_vae_traces

    # NDHWC vs NCDHW activations into the same kernels:
    python -m atom.diffusion.examples.wan_vae_bench \\
        --model /data/models/Wan2.1-T2V-1.3B-Diffusers --layout both

``--model`` also takes a Hugging Face repo id (only ``vae/*`` is fetched) or
a ``Wan2.1_VAE.pth`` file; the Diffusers comparison needs a Diffusers
checkpoint and is skipped for a ``.pth``. Diffusers must be installed for the
comparison (``pip install diffusers``); ``--no-diffusers`` skips it.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from collections import defaultdict

import torch

from atom.diffusion.models.wan import (
    CONV3D_BACKENDS,
    WanVAE,
    conv3d_backend_override,
    conv3d_call_counts,
    load_wan_vae,
    reset_conv3d_call_counts,
)
from atom.diffusion.models.wan.vae import resolve_vae_dir

logger = logging.getLogger("wan_vae_bench")

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model", required=True, help="Diffusers root/vae dir, repo id or .pth"
    )
    p.add_argument("--resolutions", default="480x832,368x544", help="HxW list")
    p.add_argument("--frames", type=int, default=81, help="1 + 4k frames")
    p.add_argument(
        "--backends",
        default="flydsl,triton,torch",
        help=f"conv3d backends to run, from {CONV3D_BACKENDS}; torch is always run",
    )
    p.add_argument("--layout", choices=("ndhwc", "ncdhw", "both"), default="ndhwc")
    p.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--skip-encode", action="store_true", help="decode only")
    p.add_argument("--no-diffusers", action="store_true")
    p.add_argument("--ref-dtype", choices=tuple(DTYPES), default="fp32")
    p.add_argument("--breakdown", action="store_true", help="per-block CUDA timing")
    p.add_argument("--profile", metavar="DIR", help="torch.profiler traces into DIR")
    p.add_argument("--min-psnr", type=float, default=35.0)
    p.add_argument("--json", metavar="PATH", help="write all results as JSON")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def resolve_model(model: str) -> str:
    if os.path.exists(model):
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(model, allow_patterns=["vae/*"])


def synthetic_clip(frames, height, width, *, seed, device, dtype):
    """Smooth moving content, so reconstruction PSNR means something."""
    gen = torch.Generator().manual_seed(seed)
    t = torch.arange(frames, dtype=torch.float32).view(frames, 1, 1)
    y = torch.linspace(0, 1, height).view(1, height, 1)
    x = torch.linspace(0, 1, width).view(1, 1, width)
    channels = [
        torch.sin(2 * math.pi * (fx * x + fy * y + speed * t))
        for fx, fy, speed in ((3.0, 2.0, 0.05), (2.0, 5.0, -0.03), (4.0, 3.0, 0.08))
    ]
    clip = 0.8 * torch.stack(channels)
    clip = clip + 0.02 * torch.randn(clip.shape, generator=gen)
    return clip.clamp(-1, 1).unsqueeze(0).to(device=device, dtype=dtype)


def compare(got: torch.Tensor, ref: torch.Tensor, *, peak: float | None = None):
    got, ref = got.float(), ref.float()
    mse = torch.mean((got - ref) ** 2).item()
    if peak is None:
        peak = (ref.max() - ref.min()).item()
    psnr = math.inf if mse == 0 else 10 * math.log10(peak**2 / mse)
    return {"psnr_db": psnr, "max_abs": (got - ref).abs().max().item()}


def timed(fn, *, warmup, iters):
    out = None
    for _ in range(warmup):
        out = fn()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return out, times


def top_level_blocks(vae) -> dict[str, torch.nn.Module]:
    blocks = {}
    for side in ("encoder", "decoder"):
        net = getattr(vae, side)
        blocks[f"{side}.conv_in"] = net.conv_in
        children = net.down_blocks if side == "encoder" else net.up_blocks
        prefix = "down_blocks" if side == "encoder" else "up_blocks"
        if side == "decoder":
            blocks[f"{side}.mid_block"] = net.mid_block
        for i, child in enumerate(children):
            blocks[f"{side}.{prefix}.{i}"] = child
        if side == "encoder":
            blocks[f"{side}.mid_block"] = net.mid_block
        blocks[f"{side}.norm_out"] = net.norm_out
        blocks[f"{side}.conv_out"] = net.conv_out
    blocks["quant_conv"] = vae.quant_conv
    blocks["post_quant_conv"] = vae.post_quant_conv
    return blocks


def block_breakdown(vae, fn) -> dict[str, float]:
    """Milliseconds per top-level block over one ``fn()``, via CUDA events."""
    events = defaultdict(list)
    handles = []
    for name, module in top_level_blocks(vae).items():

        def pre(mod, args, name=name):
            start = torch.cuda.Event(enable_timing=True)
            start.record()
            events[name].append([start, None])

        def post(mod, args, out, name=name):
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            events[name][-1][1] = end

        handles.append(module.register_forward_pre_hook(pre))
        handles.append(module.register_forward_hook(post))
    try:
        fn()
        torch.cuda.synchronize()
    finally:
        for handle in handles:
            handle.remove()
    return {
        name: sum(s.elapsed_time(e) for s, e in pairs)
        for name, pairs in events.items()
        if pairs
    }


def profile_run(vae, clip, latents, *, path, encode):
    from torch.profiler import ProfilerActivity, profile, record_function

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        if encode:
            with record_function("wan_vae.encode"):
                vae.encode(clip)
        with record_function("wan_vae.decode"):
            vae.decode(latents)
        torch.cuda.synchronize()
    prof.export_chrome_trace(path)
    averages = prof.key_averages()
    for key in ("self_device_time_total", "self_cuda_time_total"):
        try:
            table = averages.table(sort_by=key, row_limit=25)
            break
        except (AttributeError, KeyError, ValueError):
            continue
    print(f"  trace: {path}\n{table}")


def run_backend(vae, backend, clip, latents, args):
    """Encode + decode under one backend cap; outputs, timings, dispatch counts."""
    result = {"requested": backend}
    with conv3d_backend_override(backend):
        reset_conv3d_call_counts()
        if not args.skip_encode:
            result["latents"], times = timed(
                lambda: vae.encode(clip), warmup=args.warmup, iters=args.iters
            )
            result["encode_s"] = times
        result["frames"], times = timed(
            lambda: vae.decode(latents), warmup=args.warmup, iters=args.iters
        )
        result["decode_s"] = times
        result["calls"] = {
            f"{b}/{lay}": n for (b, lay), n in conv3d_call_counts().items()
        }
        result["peak_gib"] = torch.cuda.max_memory_allocated() / 2**30
        if args.breakdown:
            fn = (
                (lambda: vae.decode(latents))
                if args.skip_encode
                else (lambda: vae.decode(vae.encode(clip)))
            )
            result["breakdown_ms"] = block_breakdown(vae, fn)
    return result


def diffusers_reference(vae_dir, clip, latents, args):
    from diffusers import AutoencoderKLWan

    dtype = DTYPES[args.ref_dtype]
    ref = AutoencoderKLWan.from_pretrained(vae_dir, torch_dtype=dtype)
    ref = ref.to(args.device).eval()
    out = {"dtype": args.ref_dtype}
    with torch.no_grad():
        if not args.skip_encode:
            out["latents"], out["encode_s"] = timed(
                lambda: ref.encode(clip.to(dtype)).latent_dist.mode(),
                warmup=args.warmup,
                iters=args.iters,
            )
        out["frames"], out["decode_s"] = timed(
            lambda: ref.decode(latents.to(dtype)).sample,
            warmup=args.warmup,
            iters=args.iters,
        )
    return out


def fmt_times(times):
    if not times:
        return "-"
    return f"{min(times) * 1e3:9.1f} ms (mean {sum(times) / len(times) * 1e3:.1f})"


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not torch.cuda.is_available():
        print(
            "no GPU visible: this script validates the GPU conv3d paths",
            file=sys.stderr,
        )
        return 2
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    unknown = sorted(set(backends) - set(CONV3D_BACKENDS))
    if unknown:
        raise SystemExit(f"unknown backends {unknown}; choose from {CONV3D_BACKENDS}")
    # torch first: it is the baseline every other backend is compared with.
    backends = sorted(
        set(backends) | {"torch"}, key=CONV3D_BACKENDS.index, reverse=True
    )
    layouts = {"ndhwc": [True], "ncdhw": [False], "both": [True, False]}[args.layout]
    dtype = DTYPES[args.dtype]
    if args.profile:
        os.makedirs(args.profile, exist_ok=True)

    model = resolve_model(args.model)
    is_pth = os.path.isfile(model)
    props = torch.cuda.get_device_properties(args.device)
    print(f"device: {props.name} ({getattr(props, 'gcnArchName', '?')})")
    vae: WanVAE = load_wan_vae(model, device=args.device, dtype=dtype)

    report = {
        "device": props.name,
        "arch": getattr(props, "gcnArchName", ""),
        "runs": [],
    }
    failures = []
    for res in args.resolutions.split(","):
        height, width = (int(v) for v in res.lower().split("x"))
        clip = synthetic_clip(
            args.frames, height, width, seed=args.seed, device=args.device, dtype=dtype
        )
        with conv3d_backend_override("torch"):
            fixed_latents = vae.encode(clip)
        print(
            f"\n=== {height}x{width}x{args.frames}  latents {list(fixed_latents.shape)}"
        )

        reference = None
        if not args.no_diffusers and not is_pth:
            reference = diffusers_reference(
                resolve_vae_dir(model), clip, fixed_latents, args
            )
            torch.cuda.empty_cache()

        for channels_last in layouts:
            vae.channels_last = channels_last
            layout = "NDHWC" if channels_last else "NCDHW"
            results = {}
            for backend in backends:
                torch.cuda.reset_peak_memory_stats()
                results[backend] = run_backend(vae, backend, clip, fixed_latents, args)
                if args.profile:
                    path = os.path.join(
                        args.profile,
                        f"wan_vae_{height}x{width}_{layout}_{backend}.json",
                    )
                    with conv3d_backend_override(backend):
                        profile_run(
                            vae,
                            clip,
                            fixed_latents,
                            path=path,
                            encode=not args.skip_encode,
                        )

            base = results["torch"]
            print(f"--- layout {layout}")
            for backend, r in results.items():
                row = {
                    "resolution": f"{height}x{width}",
                    "frames": args.frames,
                    "layout": layout,
                    "backend": backend,
                    "conv3d_calls": r["calls"],
                    "encode_s": r.get("encode_s"),
                    "decode_s": r["decode_s"],
                    "peak_gib": r["peak_gib"],
                    "recon_vs_input": compare(r["frames"], clip, peak=2.0),
                }
                if backend != "torch":
                    row["frames_vs_torch"] = compare(
                        r["frames"], base["frames"], peak=2.0
                    )
                    if "latents" in r:
                        row["latents_vs_torch"] = compare(r["latents"], base["latents"])
                if reference is not None:
                    row["frames_vs_diffusers"] = compare(
                        r["frames"], reference["frames"], peak=2.0
                    )
                    if "latents" in r:
                        row["latents_vs_diffusers"] = compare(
                            r["latents"], reference["latents"]
                        )
                if "breakdown_ms" in r:
                    row["breakdown_ms"] = r["breakdown_ms"]
                report["runs"].append(row)

                ran = ", ".join(f"{k}: {v}" for k, v in sorted(r["calls"].items()))
                print(f"[{backend:>6}] convs ran on {{{ran}}}")
                if not any(k.startswith(backend + "/") for k in r["calls"]):
                    print(f"         !! nothing ran on {backend}: unsupported here")
                print(f"         encode {fmt_times(r.get('encode_s'))}")
                print(
                    f"         decode {fmt_times(r['decode_s'])}   peak {r['peak_gib']:.1f} GiB"
                )
                for key in (
                    "frames_vs_torch",
                    "latents_vs_torch",
                    "frames_vs_diffusers",
                    "latents_vs_diffusers",
                    "recon_vs_input",
                ):
                    if key in row:
                        m = row[key]
                        print(
                            f"         {key:<21} PSNR {m['psnr_db']:6.2f} dB  max|d| {m['max_abs']:.4g}"
                        )
                if (
                    backend != "torch"
                    and row["frames_vs_torch"]["psnr_db"] < args.min_psnr
                ):
                    failures.append(f"{height}x{width} {layout} {backend}")
                if "breakdown_ms" in row:
                    for name, ms in row["breakdown_ms"].items():
                        print(f"           {name:<28} {ms:9.2f} ms")

        if reference is not None:
            print(
                f"[diffusers {reference['dtype']}] encode {fmt_times(reference.get('encode_s'))}"
                f"  decode {fmt_times(reference['decode_s'])}"
            )
            report["runs"].append(
                {
                    "resolution": f"{height}x{width}",
                    "backend": f"diffusers-{reference['dtype']}",
                    "encode_s": reference.get("encode_s"),
                    "decode_s": reference["decode_s"],
                }
            )

    if args.json:
        with open(args.json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nresults: {args.json}")
    if failures:
        print(f"\nFAIL: frames below {args.min_psnr} dB vs torch: {failures}")
        return 1
    print("\nPASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
