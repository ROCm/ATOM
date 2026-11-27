# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""ROCm/Megatron-LM platform SOURCE patches (not DSV4-model-specific).

These patches apply to the ROCm Megatron checkout regardless of model family.
Filter with ``--tag rocm`` (combine with ``--tag dsv4`` for the full DSV4 stack).
"""

from __future__ import annotations

import os

from atomrl.lumen.patches.registry import PatchPhase, register_patch


def patch_disable_batch_p2p_comm(megatron_root: str) -> bool:
    """Allow MEGATRON_NO_BATCH_P2P_COMM=1 to force batch_p2p_comm=False."""
    path = os.path.join(megatron_root, "megatron", "training", "arguments.py")
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        content = f.read()
    marker = "MEGATRON_NO_BATCH_P2P_COMM"
    if marker in content:
        return False
    old = "    kw_args['batch_p2p_comm'] = not args.overlap_p2p_comm\n"
    new = (
        "    kw_args['batch_p2p_comm'] = not args.overlap_p2p_comm\n"
        "    if os.environ.get('MEGATRON_NO_BATCH_P2P_COMM', '0') == '1':\n"
        "        kw_args['batch_p2p_comm'] = False\n"
    )
    if old not in content:
        return False
    with open(path, "w") as f:
        f.write(content.replace(old, new, 1))
    return True


def patch_cpu_offload_torch_gpu_adam(megatron_root: str) -> bool:
    """Use CPUAdam for GPU hybrid-offload partitions (MI325 gfx950 TE Adam workaround)."""
    path = os.path.join(megatron_root, "megatron", "core", "optimizer", "__init__.py")
    if not os.path.isfile(path):
        return False
    with open(path) as f:
        content = f.read()
    old = (
        "            if config.optimizer == 'adam':\n"
        "                gpu_optimizer_cls = Adam\n"
        "                cpu_optimizer_cls = CPUAdam\n"
    )
    new = (
        "            if config.optimizer == 'adam':\n"
        "                gpu_optimizer_cls = CPUAdam\n"
        "                cpu_optimizer_cls = CPUAdam\n"
    )
    if "gpu_optimizer_cls = CPUAdam" in content:
        return False
    if old not in content:
        return False
    with open(path, "w") as f:
        f.write(content.replace(old, new, 1))
    return True


register_patch(
    "disable_batch_p2p_comm",
    PatchPhase.SOURCE,
    description="MEGATRON_NO_BATCH_P2P_COMM=1 forces batch_p2p_comm=False",
    tags=frozenset({"rocm", "pipeline", "megatron"}),
)(patch_disable_batch_p2p_comm)

register_patch(
    "cpu_offload_torch_gpu_adam",
    PatchPhase.SOURCE,
    description="Use CPUAdam for GPU hybrid-offload partitions on MI325",
    tags=frozenset({"rocm", "optimizer", "megatron"}),
)(patch_cpu_offload_torch_gpu_adam)
