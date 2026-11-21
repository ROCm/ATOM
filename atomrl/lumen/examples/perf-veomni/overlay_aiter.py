#!/usr/bin/env python3
"""Add the aiter modules Lumen needs but upstream aiter does not ship.

Lumen imports aiter Triton modules at the top level of its own modules
(cross_entropy, sparse_mla_dsv4_train) that aiter 0.1.12.post2 does not ship.
Without them `import atomrl.lumen` raises ModuleNotFoundError. Copying whole
directories of newer aiter Python over an install is not an option: it expects
newer CK/csrc source, which breaks the runtime JIT build. This adds only the
missing files to the aiter that is already installed in this container.

Safety: only files that do not already exist are written, so nothing aiter
currently ships is modified. Anything already present is reported and skipped --
use --force only if you have a reason, since overwriting an installed file is
what the Dockerfile comment warns about.

    python overlay_aiter.py           # dry run, lists what would be written
    python overlay_aiter.py --apply
    python overlay_aiter.py --revert

The files under aiter_overlay/ come from two places, and are copied into the
example rather than fetched so that reproducing this does not require cloning
several hundred MB of aiter:

* ops/triton/** -- official AITER pinned at ff745fb0, byte
  for byte: the Triton modules Lumen calls, plus the helpers they import that
  0.1.12.post2 lacks (utils/config_utils.py, utils/tuned_config_utils.py,
  _triton_kernels/common/segment_tile.py). Each was checked to import on
  0.1.12.post2 with only the files it already ships. fusions/mhc.py,
  mxfp8_attention_kernel.py and block_scale_fp8_attention_kernel.py are left
  out: they need newer versions of files 0.1.12.post2 already has
  (ops/mhc.py, utils/logger.py, arch_info.is_cdna4), which this overlay never
  replaces. Lumen guards those imports, so only the DSv4 hyper-connection and
  the FP8/MXFP8 Triton attention paths are unavailable on such an install.
* ops/flydsl/conv_kernels.py, ops/flydsl/conv3d_policy.py,
  ops/flydsl/kernels/conv/*, configs/*conv3d*.csv and the Qwen-Image / Wan2.1
  tables in configs/model_configs/ -- the FlyDSL implicit-GEMM convolution and
  its per-model tuned tile tables, as merged into aiter main by ROCm/aiter#5370
  (merge 305c421e, 2026-09-24). An aiter that already includes that change has
  all of them, so they are reported present and skipped. The PR also edits
  three files aiter ships (ops/flydsl/__init__.py, jit/core.py,
  aot/flydsl/common.py); those are NOT overlaid onto an older install. Lumen
  imports aiter.ops.flydsl.conv_kernels directly instead of through the package
  export, and supplies the tuned-table path itself
  (veomni_patches/aiter_conv.py), so the installed files stay as shipped.
* configs/model_configs/wan21_vae_padk_bf16_*_conv3d.csv -- Lumen's own rows
  for the shapes conv_pad_in_kernel produces, made by tune_conv3d.py. aiter
  merges every model_configs/*bf16_tuned_conv3d*.csv, so they are picked up
  with no code.
"""

import argparse
import json
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("AITER_OVERLAY_SRC", os.path.join(_HERE, "aiter_overlay", "aiter"))
MANIFEST = os.environ.get(
    "AITER_OVERLAY_MANIFEST",
    os.path.join(os.environ.get("LOG_DIR", "/tmp"), "aiter_overlay_manifest.json"),
)


def aiter_root():
    import aiter

    return os.path.dirname(aiter.__file__)


# The tuned conv3d tables are data aiter reads at runtime, so they overlay too.
_OVERLAY_SUFFIXES = (".py", ".csv")


def planned(root):
    """Yield (src, dst, rel) for every file in the overlay tree."""
    for dirpath, _, files in os.walk(SRC):
        for name in sorted(files):
            if not name.endswith(_OVERLAY_SUFFIXES):
                continue
            src = os.path.join(dirpath, name)
            rel = os.path.relpath(src, SRC)
            yield src, os.path.join(root, rel), rel


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="write the files")
    ap.add_argument(
        "--force", action="store_true", help="also overwrite existing files"
    )
    ap.add_argument("--revert", action="store_true", help="remove what --apply added")
    args = ap.parse_args()

    root = aiter_root()
    print(f"aiter install: {root}")
    print(f"overlay source: {SRC}\n")

    if args.revert:
        if not os.path.exists(MANIFEST):
            print(f"no manifest at {MANIFEST}; nothing recorded to revert")
            return 1
        with open(MANIFEST) as fh:
            added = json.load(fh)["added"]
        for rel in added:
            dst = os.path.join(root, rel)
            if os.path.exists(dst):
                os.remove(dst)
                print(f"  removed {rel}")
            else:
                print(f"  already gone {rel}")
        # Drop stale bytecode so the removal takes effect.
        for dirpath, dirnames, _ in os.walk(root):
            for d in list(dirnames):
                if d == "__pycache__":
                    shutil.rmtree(os.path.join(dirpath, d), ignore_errors=True)
        os.remove(MANIFEST)
        print(f"\nreverted {len(added)} files, manifest removed")
        return 0

    to_add, existing = [], []
    for src, dst, rel in planned(root):
        (existing if os.path.exists(dst) else to_add).append((src, dst, rel))

    print(f"{len(to_add)} new, {len(existing)} already present\n")
    for _, _, rel in to_add:
        print(f"  NEW      {rel}")
    for _, _, rel in existing:
        print(f"  PRESENT  {rel}   (skipped; --force to overwrite)")

    targets = to_add + (existing if args.force else [])
    if not args.apply:
        print(f"\ndry run -- would write {len(targets)} files. Re-run with --apply.")
        return 0

    written = []
    provided = {
        dst for _, dst, rel in targets if os.path.basename(rel) == "__init__.py"
    }
    for src, dst, rel in targets:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        # A package dir needs __init__.py to be importable; aiter ships them for
        # existing dirs, but fusions/ may be new here. Data directories are not
        # packages, and a directory whose __init__.py the overlay carries needs
        # no empty marker.
        pkg = os.path.dirname(dst)
        while (
            rel.endswith(".py")
            and pkg != root
            and not os.path.exists(os.path.join(pkg, "__init__.py"))
            and os.path.join(pkg, "__init__.py") not in provided
        ):
            open(os.path.join(pkg, "__init__.py"), "a").close()
            written.append(os.path.relpath(os.path.join(pkg, "__init__.py"), root))
            print(f"  created  {written[-1]}  (package marker)")
            pkg = os.path.dirname(pkg)
        shutil.copy2(src, dst)
        written.append(rel)
        print(f"  wrote    {rel}")

    # A second --apply only writes what the first did not, so merge rather than
    # overwrite: otherwise --revert would forget everything the earlier run added.
    recorded = written
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as fh:
            previous = json.load(fh).get("added", [])
        recorded = previous + [rel for rel in written if rel not in previous]
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    with open(MANIFEST, "w") as fh:
        json.dump({"aiter_root": root, "added": recorded}, fh, indent=2)
    print(
        f"\nwrote {len(written)} files; manifest ({len(recorded)} recorded) -> {MANIFEST}"
    )
    print(f"revert with: python {os.path.abspath(__file__)} --revert")
    return 0


if __name__ == "__main__":
    sys.exit(main())
