import pytest

from atom.models.kimi_k3.mono.kernels.execution_plan import (
    Gfx1250KimiBuildPlan,
)
from atom.models.kimi_k3.mono.kernels.stages.k2_finish_persistent_gfx1250 import (
    FINISH_PACKED_FRONT_ABI,
    build_packed_finish,
)


def _plan(rows):
    return Gfx1250KimiBuildPlan(
        threads=256,
        blocks=256,
        lds_bytes=320 * 1024,
        tokens=rows,
        nblocks=4,
    )


@pytest.mark.parametrize("rows", (1, 16, 17, 32, 33, 48))
def test_persistent_finish_is_one_launch_with_static_resident_grid(rows):
    candidate = build_packed_finish(_plan(rows))
    assert candidate.grid == (256,)
    assert candidate.block == (256,)
    assert candidate.launch_count == 1
    assert candidate.required_lds_bytes == 113_664
    assert candidate.schedule.rows == rows
    candidate.schedule.prove_coverage()


def test_persistent_finish_abi_is_packed_and_rank_local():
    candidate = build_packed_finish(_plan(48))
    assert candidate.abi == FINISH_PACKED_FRONT_ABI
    assert candidate.abi[0] == "front_gathered"
    assert "routed_intermediate_fp4" in candidate.abi
    assert "routed_intermediate_scales" in candidate.abi
    assert not any(
        forbidden in name
        for name in candidate.abi
        for forbidden in ("peer", "ipc", "xgmi", "collective")
    )
