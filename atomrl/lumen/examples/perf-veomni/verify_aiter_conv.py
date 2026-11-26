#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Single-GPU check of Lumen's AITER-backed convolution in a real VAE.

Encodes one input at the shape training uses through the stock BF16 VAE and
through the patched one (lumen_vae_conv -> atomrl.lumen.ops.conv -> AITER), and reports:

  * latents against an FP32 encode, patched next to stock BF16
  * encode time
  * the conv kernels and their GPU time for one encode
  * with --padk, that conv_pad_in_kernel leaves the latents bit-identical

    verify_aiter_conv.py --model qwen            # 1 x 1024 x 1024
    verify_aiter_conv.py --model wan --padk      # 81 x 368 x 544

Requires an AITER release that provides the public Conv2d/Conv3d APIs.
"""

import argparse
import collections
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lumen_vae_conv import patch_vae_convs, snr_db


def timed(fn, iters, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


def load_vae(which):
    import diffusers

    if which == "wan":
        path = os.environ.get("WAN_DIR", "/work/models-extra/Wan2.1-T2V-1.3B")
        return diffusers.AutoencoderKLWan.from_pretrained(
            path, subfolder="vae", torch_dtype=torch.float32
        )
    path = os.environ.get("QWEN_IMAGE_DIR", "/work/models/Qwen-Image")
    return diffusers.AutoencoderKLQwenImage.from_pretrained(
        path, subfolder="vae", torch_dtype=torch.float32
    )


def gpu_kernels(fn):
    """Kernel name -> (calls, total ms) for one call of fn."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        path = fh.name
    prof.export_chrome_trace(path)
    with open(path) as fh:
        events = json.load(fh)["traceEvents"]
    os.remove(path)
    out = collections.defaultdict(lambda: [0, 0.0])
    for ev in events:
        if ev.get("cat") == "kernel":
            out[ev["name"]][0] += 1
            out[ev["name"]][1] += ev.get("dur", 0.0) / 1000.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=("wan", "qwen"), required=True)
    ap.add_argument(
        "--padk", action="store_true", help="Wan: also measure conv_pad_in_kernel"
    )
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--iters", type=int, default=5)
    args = ap.parse_args()

    video = args.model == "wan"
    h = args.height or (368 if video else 1024)
    w = args.width or (544 if video else 1024)
    t = args.frames if video else 1

    torch.manual_seed(0)
    x32 = torch.rand(1, 3, t, h, w, device="cuda") * 2 - 1
    vae = load_vae(args.model).cuda().eval()
    with torch.no_grad():
        ref32 = vae.encode(x32).latent_dist.parameters.float()
    vae = vae.to(torch.bfloat16)
    x = x32.to(torch.bfloat16)

    def enc(vae=vae, x=x):
        with torch.no_grad():
            return vae.encode(x).latent_dist.parameters

    stock = enc().float()
    t_stock = timed(enc, args.iters)
    print(f"model {args.model}  input {tuple(x.shape)}  latents {tuple(ref32.shape)}")
    print(
        f"stock BF16      : {t_stock:8.1f} ms/encode   SNR vs FP32 {snr_db(ref32, stock):5.1f} dB"
    )
    del enc, vae
    torch.cuda.empty_cache()

    modes = [("pre-pad", False)] + (
        [("pad-in-kernel", True)] if (video and args.padk) else []
    )
    results = {}
    for mode, padk in modes:
        vae_m = load_vae(args.model).cuda().eval().to(torch.bfloat16)
        stats = patch_vae_convs(vae_m, video=video, causal_pad_in_kernel=padk)

        def enc_m(v=vae_m):
            with torch.no_grad():
                return v.encode(x).latent_dist.parameters

        lat = enc_m().float()
        ms = timed(enc_m, args.iters)
        kernels = gpu_kernels(enc_m)
        results[mode] = lat
        print(
            f"{mode:15} : {ms:8.1f} ms/encode   SNR vs FP32 {snr_db(ref32, lat):5.1f} dB   "
            f"{len(stats['patched'])} convs patched"
        )
        conv = sorted(
            (
                (n, r)
                for n, r in kernels.items()
                if "conv3d" in n or "transpose_kernel" in n
            ),
            key=lambda kv: -kv[1][1],
        )
        for n, (calls, ms_k) in conv:
            print(f"      {calls:4d} x  {ms_k:7.2f} ms  {n}")
        del vae_m
        torch.cuda.empty_cache()

    if len(modes) == 2:
        d = (results["pad-in-kernel"] - results["pre-pad"]).abs().max().item()
        print(
            f"pad-in-kernel vs pre-pad latents max|d| = {d:.3e}"
            + ("  (bit-identical)" if d == 0 else "")
        )


if __name__ == "__main__":
    main()
