# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU gate: MTP draft dense-only geometry must not crash paged prefill.

GPU kernel coverage lives in ``test_prefill_indices_paged.py``. This module
stays aiter-free so Pre Checkin can catch the regression that asserted
CSA+HCA on pools that only serve dense (Pro MTP3 draft-extend).
"""

from pathlib import Path

from atom.model_ops.attentions.pool_layout.v4_pool_geometry import (
    CSA_RATIO,
    DENSE_RATIO,
    HCA_RATIO,
    UnifiedPoolGeometry,
)

PREFILL_INDICES_SRC = (
    Path(__file__).parents[1] / "atom/model_ops/v4_kernels/paged_prefill_indices.py"
).read_text()


def test_prefill_indices_source_gates_absent_csa_hca():
    assert "HAS_CSA" in PREFILL_INDICES_SRC
    assert "HAS_HCA" in PREFILL_INDICES_SRC
    assert "has_csa = CSA_RATIO in served" in PREFILL_INDICES_SRC
    assert "has_hca = HCA_RATIO in served" in PREFILL_INDICES_SRC
    assert (
        "writes the CSA and HCA prefix buffers unconditionally"
        not in PREFILL_INDICES_SRC
    )


def test_dense_only_geometry_serves_no_csa_hca():
    geometry = UnifiedPoolGeometry(
        [DENSE_RATIO],
        num_blocks=4,
        num_slots=2,
        ring_slots=8,
        block_size=256,
    )
    served = {ratio: geometry.window_params(ratio) for ratio in geometry.classes}
    assert list(served) == [DENSE_RATIO]
    assert CSA_RATIO not in served
    assert HCA_RATIO not in served
    # Draft-extend prefill must tolerate this served set instead of asserting.
    assert CSA_RATIO not in geometry.classes
    assert HCA_RATIO not in geometry.classes
