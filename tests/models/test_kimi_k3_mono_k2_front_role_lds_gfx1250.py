from atom.models.kimi_k3.mono.kernels.execution_plan import (
    Gfx1250KimiBuildPlan,
)
from atom.models.kimi_k3.mono.kernels.stages.k2_front_role_lds_gfx1250 import (
    ATTENTION_CTAS,
    ROLE_LDS_FRONT_ABI,
    build_role_lds_front,
)


def _plan(rows):
    return Gfx1250KimiBuildPlan(
        threads=256,
        blocks=256,
        lds_bytes=320 * 1024,
        tokens=rows,
        nblocks=4,
    )


def test_role_lds_front_abi_has_no_full_attnres_workspace():
    candidate = build_role_lds_front(_plan(1))
    assert candidate.abi == ROLE_LDS_FRONT_ABI
    assert "attnres_values" not in candidate.abi
    assert "attnres_mixed" not in candidate.abi
    assert "attnres_partials" not in candidate.abi
    assert "attnres_mean_squares" not in candidate.abi
    assert candidate.abi[3:5] == ("blk_sm", "blk_sr")
    assert candidate.grid == (256,)
    assert candidate.block == (256,)
    assert candidate.launch_count == 1
    assert candidate.required_lds_bytes == 237_824
    assert ATTENTION_CTAS == 56


def test_role_lds_front_scales_to_s48_without_more_launches():
    s1 = build_role_lds_front(_plan(1))
    s48 = build_role_lds_front(_plan(48))
    assert s1.launch_count == s48.launch_count == 1
    assert s48.traffic.net_removed_global_bytes == (
        48 * s1.traffic.net_removed_global_bytes
    )
    assert s48.scratch_bytes > s1.scratch_bytes


def test_role_lds_front_specializes_block_write_prefix_reset():
    normal = build_role_lds_front(_plan(16))
    reset = build_role_lds_front(_plan(16), reset=True)
    assert normal.name.endswith("_p1_r0")
    assert reset.name.endswith("_p1_r1")
    assert normal.abi == reset.abi
