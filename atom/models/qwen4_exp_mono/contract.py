# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Binary resource contract for every 256-CTA persistent Qwen kernel."""

from atom.mono.runtime.compile import launcher_resources

# Exact high-water marks from the accepted fresh-compile artifacts. Scalar
# spills lower to VGPR lanes and remain private-segment free; width builds are
# rejected if a compiler change raises any of these counts.
SGPR_SPILL_LIMITS = {
    1: {"gdn": 75, "qsa_pre": 75, "qsa_direct": 96, "qsa_full": 75},
    2: {"gdn": 200, "qsa_pre": 100, "qsa_direct": 158, "qsa_full": 184},
    4: {"gdn": 532, "qsa_pre": 235, "qsa_direct": 599, "qsa_full": 924},
}


def sgpr_spill_limit(tokens: int, stage: str) -> int:
    try:
        return SGPR_SPILL_LIMITS[tokens][stage]
    except KeyError as exc:
        raise ValueError(
            f"Qwen mono has no resource contract for S={tokens} {stage}"
        ) from exc


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
        # LLVM lowers these scalar spills to one VGPR's lanes; the accepted
        # kernel has no private segment. Device/LDS pointer tables removed most
        # of them but cost 4.6-7.7 us, so keep the measured faster contract.
        if entry["sgpr_spill"] is not None and entry["sgpr_spill"] > max_sgpr_spill:
            raise RuntimeError(f"{entry['name']} has sgpr_spill={entry['sgpr_spill']}")
        if entry["lds"] > max_lds:
            raise RuntimeError(f"{entry['name']} uses {entry['lds']} B LDS")
    return resources
