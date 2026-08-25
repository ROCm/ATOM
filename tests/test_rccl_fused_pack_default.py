# SPDX-License-Identifier: MIT

from pathlib import Path

RCCL_SOURCES = (
    Path("atom/model_ops/fused_moe/rccl_batched_experts.py"),
    Path("atom/model_ops/fused_moe/rccl_prepare_finalize.py"),
)


def test_rccl_fused_pack_has_no_runtime_toggle():
    for source in RCCL_SOURCES:
        assert "ATOM_RCCL_FUSED_PACK" not in source.read_text()
