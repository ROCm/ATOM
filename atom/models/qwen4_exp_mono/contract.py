# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Binary resource contract for every 256-CTA persistent Qwen kernel."""

from atom.mono.runtime.compile import launcher_resources

# Fresh-build scalar spill ceilings for the pinned gfx950 toolchain. Scalar
# spills lower to VGPR lanes; private scratch and VGPR spills remain forbidden.
SGPR_SPILL_LIMITS = {
    1: {
        "gdn": 75,
        "qsa_pre": 75,
        "qsa_direct": 119,
        "qsa_full": 81,
        # Fixed-LDS selection trees through the checkpoint's 256K capacity.
        "qsa_tree": 154,
    },
}


def sgpr_spill_limit(tokens: int, stage: str) -> int:
    try:
        return SGPR_SPILL_LIMITS[tokens][stage]
    except KeyError as exc:
        raise ValueError(
            f"Qwen mono has no resource contract for S={tokens} {stage}"
        ) from exc


# Accepted limits belong to the build contract, independent of the TP wrapper.
TP8_SPILL_LIMITS = {
    "tp8_finish": 88,
    "tp8_pull": 0,
    "tp8_gdn_k1": 0,
    "tp8_gdn_k2": 88,
    "tp8_qsa_k1": 0,
    "tp8_qsa_k2": 72,
}


def check_batch(build):
    for launcher, _ in build.kernels():
        check_launcher(launcher, max_sgpr_spill=0, max_lds=128 * 1024)


def check_tp8(build):
    for name, spill in TP8_SPILL_LIMITS.items():
        check_launcher(getattr(build, name), max_sgpr_spill=spill, max_lds=128 * 1024)


def check_launcher(
    launcher,
    *,
    max_sgpr_spill: int = 75,
    max_lds: int = 32 * 1024,
) -> list[dict]:
    resources = launcher_resources(launcher)
    if not resources:
        raise RuntimeError("Qwen mono launcher has no compiled binary")
    for entry in resources:
        if entry["scratch"]:
            raise RuntimeError(
                f"{entry['name']} uses {entry['scratch']} B scratch/lane"
            )
        if entry["vgpr_spill"] not in (None, 0):
            raise RuntimeError(f"{entry['name']} has vgpr_spill={entry['vgpr_spill']}")
        # Scalar-to-VGPR spills have stage-specific accepted ceilings; private
        # scratch and VGPR spills remain forbidden.
        if entry["sgpr_spill"] is not None and entry["sgpr_spill"] > max_sgpr_spill:
            raise RuntimeError(f"{entry['name']} has sgpr_spill={entry['sgpr_spill']}")
        if entry["lds"] > max_lds:
            raise RuntimeError(f"{entry['name']} uses {entry['lds']} B LDS")
    return resources
