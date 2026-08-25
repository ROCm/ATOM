# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Decode and pretty-print ATOM debug dump files.

Handles the dump formats written by ``atom.utils.debug_helper.dump``:

- MiniMax-M3 per-layer hidden dump  (``{kind}_layer{LL}_{stage}_rank{R}.pt``)
    keys: hidden, shape, layer, stage, kind, mean, var
- fused_moe kernel I/O dump          (``moe_{kind}_{layer}_rank{R}.pt``)
    keys: _layer_name, _kind, _rank, inputs{...}, output, _output_shape
- generic forward / weight dumps     (any dict of name -> tensor / scalar)

For every tensor it prints shape / dtype / device and summary stats
(mean / std / min / max, plus nan/inf counts). Non-tensor values (scalars,
enums like ActivationType / QuantType) are printed verbatim — these are the
metadata needed to replay a kernel offline.

Usage::

    # one file
    python -m atom.utils.debug_helper.decode_dump path/to/moe_prefill_..._rank0.pt

    # every *.pt in a directory
    python -m atom.utils.debug_helper.decode_dump --dir /tmp/moe_io

    # also print raw tensor values (can be large)
    python -m atom.utils.debug_helper.decode_dump file.pt --full
"""

from __future__ import annotations

import argparse
import glob
import os
from typing import Any, Optional

import torch


def _tensor_summary(t: torch.Tensor) -> str:
    """One-line shape / dtype / device + numeric stats for a tensor."""
    head = f"Tensor shape={tuple(t.shape)} dtype={t.dtype} device={t.device}"
    # Float stats need a float view; int/uint/fp8 packed tensors get a lighter
    # summary (min/max only) since mean/std may be meaningless or unsupported.
    try:
        if t.numel() == 0:
            return head + " (empty)"
        if t.is_floating_point():
            tf = t.detach().float()
            n_nan = int(torch.isnan(tf).sum().item())
            n_inf = int(torch.isinf(tf).sum().item())
            finite = tf[torch.isfinite(tf)]
            if finite.numel() > 0:
                stats = (
                    f"mean={finite.mean().item():.6e} std={finite.std().item():.6e} "
                    f"min={finite.min().item():.6e} max={finite.max().item():.6e}"
                )
            else:
                stats = "no finite values"
            extra = f" nan={n_nan} inf={n_inf}" if (n_nan or n_inf) else ""
            return f"{head} | {stats}{extra}"
        # integer / bool / packed types
        ti = t.detach()
        try:
            lo = ti.min().item()
            hi = ti.max().item()
            return f"{head} | min={lo} max={hi}"
        except Exception:
            return f"{head} | (no min/max for this dtype)"
    except Exception as e:  # never let a summary crash the whole decode
        return f"{head} | <stat error: {e}>"


def _print_value(name: str, val: Any, indent: str, full: bool) -> None:
    if isinstance(val, torch.Tensor):
        print(f"{indent}{name}: {_tensor_summary(val)}")
        if full:
            print(f"{indent}    {val.detach().cpu()}")
    elif isinstance(val, dict):
        print(f"{indent}{name}: dict ({len(val)} keys)")
        for k in val:
            _print_value(str(k), val[k], indent + "    ", full)
    elif isinstance(val, (list, tuple)):
        print(f"{indent}{name}: {type(val).__name__} (len {len(val)})")
        for i, v in enumerate(val):
            _print_value(f"[{i}]", v, indent + "    ", full)
    elif val is None:
        print(f"{indent}{name}: None")
    else:
        # scalars, enums (ActivationType / QuantType), strings, bools
        print(f"{indent}{name}: {val!r}  (type={type(val).__name__})")


def decode_file(path: str, full: bool = False) -> None:
    """Load one dump file and print its contents."""
    print("=" * 78)
    print(f"FILE: {path}")
    print("=" * 78)
    # weights_only=False: dumps may contain aiter enums (ActivationType,
    # QuantType) that are not in torch's safe-globals allowlist.
    obj = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(obj, dict):
        _print_value("(root)", obj, "", full)
        print()
        return

    # Header line summarizing the recognized dump kind, if any.
    kind = obj.get("_kind") or obj.get("kind")
    layer = obj.get("_layer_name")
    if layer is None and "layer" in obj:
        layer = f"layer{obj['layer']}"
    stage = obj.get("stage")
    tags = [str(x) for x in (kind, layer, stage) if x is not None]
    if tags:
        print("DUMP:", " | ".join(tags))

    # Print every top-level key. inputs/output get nested expansion via
    # _print_value; scalar metadata prints inline.
    for key in obj:
        _print_value(key, obj[key], "", full)
    print()


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Decode and print ATOM debug dump (.pt) files."
    )
    ap.add_argument("files", nargs="*", help="One or more .pt dump files.")
    ap.add_argument("--dir", help="Decode every *.pt file in this directory.")
    ap.add_argument(
        "--full",
        action="store_true",
        help="Also print raw tensor values (may be very large).",
    )
    args = ap.parse_args(argv)

    paths: list[str] = list(args.files)
    if args.dir:
        paths.extend(sorted(glob.glob(os.path.join(args.dir, "*.pt"))))
    if not paths:
        ap.error("no files given; pass file paths and/or --dir")

    missing = [p for p in paths if not os.path.isfile(p)]
    for p in missing:
        print(f"[skip] not a file: {p}")
    paths = [p for p in paths if os.path.isfile(p)]

    for p in paths:
        try:
            decode_file(p, full=args.full)
        except Exception as e:
            print(f"[error] failed to decode {p}: {e}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
