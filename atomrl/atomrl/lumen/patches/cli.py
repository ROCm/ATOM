# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Apply registered SOURCE-phase patches to a Megatron-LM checkout."""

from __future__ import annotations

import argparse
import sys

from atomrl.lumen.patches.registry import PatchPhase, apply_patches, list_patches
from atomrl.lumen.patches.types import PatchResult


def parse_cli_tags(
    raw_tags: list[str] | None,
    *,
    tag_mode: str | None = None,
) -> tuple[set[str] | None, str]:
    """Parse ``--tag`` CLI values.

    * Comma-separated values (``--tag dsv4,rocm``) use OR matching (``any``).
    * Repeated flags (``--tag dsv4 --tag rocm``) use AND matching (``all``).
    * A single tag keeps AND semantics (equivalent to OR for one tag).
    """
    if not raw_tags:
        return None, "all"

    expanded: list[str] = []
    saw_comma = False
    for item in raw_tags:
        parts = [part.strip() for part in item.split(",") if part.strip()]
        if len(parts) > 1:
            saw_comma = True
        expanded.extend(parts)

    if tag_mode is not None:
        mode = tag_mode
    elif saw_comma:
        mode = "any"
    elif len(raw_tags) > 1:
        mode = "all"
    else:
        mode = "all"

    return set(expanded), mode


def _ensure_source_patches_registered() -> None:
    import atomrl.lumen.patches.source  # noqa: F401


def apply_megatron_source_patches(
    megatron_root: str,
    *,
    tags: set[str] | None = None,
    tag_mode: str = "all",
    default_only: bool = True,
    dry_run: bool = False,
) -> dict[str, PatchResult]:
    """Apply SOURCE patches to *megatron_root*."""
    _ensure_source_patches_registered()
    return apply_patches(
        PatchPhase.SOURCE,
        megatron_root=megatron_root,
        tags=tags,
        tag_mode=tag_mode,
        default_only=default_only,
        dry_run=dry_run,
    )


def print_patch_report(
    megatron_root: str, results: dict[str, PatchResult], *, dry_run: bool = False
) -> None:
    title = "Would patch" if dry_run else "Patched"
    print(f"{title} ROCm Megatron at {megatron_root}:")
    for name in sorted(results):
        result = results[name]
        if result.reason == "dry-run":
            status = "would apply"
        elif result.skipped:
            status = f"skipped ({result.reason})"
        elif not result.applied or result.return_value is False:
            status = "skipped"
        else:
            status = "PATCHED"
        print(f"  {status}: {name}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Apply Lumen SOURCE patches to a Megatron-LM checkout.",
    )
    parser.add_argument(
        "megatron_root",
        nargs="?",
        help="Path to Megatron-LM root",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List registered SOURCE patches and exit",
    )
    parser.add_argument(
        "--tag",
        action="append",
        dest="tags",
        default=None,
        help=(
            "Filter patches by tag. Comma-separated values use OR "
            "(e.g. --tag dsv4,rocm). Repeat --tag for AND "
            "(e.g. --tag dsv4 --tag rocm)."
        ),
    )
    parser.add_argument(
        "--tag-mode",
        choices=("all", "any"),
        default=None,
        help="Override tag matching: all=AND (default), any=OR.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Apply non-default patches too",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show which patches would be applied without modifying files",
    )
    args = parser.parse_args(argv)

    _ensure_source_patches_registered()

    tag_set, tag_mode = parse_cli_tags(args.tags, tag_mode=args.tag_mode)

    if args.list:
        for spec in list_patches(PatchPhase.SOURCE, tags=tag_set, tag_mode=tag_mode):
            gate = "enabled" if spec.enabled() else "disabled"
            print(
                f"{spec.name:40} [{gate}] default={spec.default} "
                f"tags={','.join(sorted(spec.tags)) or '-'}"
            )
            if spec.description:
                print(f"  {spec.description}")
        return 0

    if not args.megatron_root:
        parser.error("megatron_root is required unless --list is set")

    # Explicit --tag includes opt-in (default=False) patches; bare apply keeps defaults.
    default_only = not args.all and tag_set is None

    results = apply_megatron_source_patches(
        args.megatron_root,
        tags=tag_set,
        tag_mode=tag_mode,
        default_only=default_only,
        dry_run=args.dry_run,
    )
    print_patch_report(args.megatron_root, results, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
