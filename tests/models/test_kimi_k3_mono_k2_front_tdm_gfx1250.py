import pytest

from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import k2_front_tdm_gfx1250 as front
from atom.models.kimi_k3.mono.kernels.stages.k2_finish_persistent_gfx1250 import (
    build_packed_finish,
)
from atom.models.kimi_k3.mono.kernels.stages.gemv_layout import tdm_lds_bytes


def _plan(rows):
    return Gfx1250KimiBuildPlan(
        threads=256, blocks=256, lds_bytes=320 * 1024, tokens=rows, nblocks=4
    )


def test_task_pairs_never_straddle_two_weights():
    assert front.ROUTER_TASKS % 2 == 0
    assert (front.ROUTER_TASKS + front.LATENT_TASKS) % 2 == 0
    assert front.SHARED_TASKS % 2 == 0
    assert front.PAIRS == 188


@pytest.mark.parametrize("rows", (1, 16, 17, 40, 48))
def test_replicated_front_is_one_launch_of_one_cta_per_task_pair(rows):
    kernel = front.build_replicated_front(_plan(rows))
    assert kernel.grid == (front.PAIRS,)
    assert kernel.block == (256,)
    assert kernel.launch_count == 1
    assert kernel.name.endswith(f"_s{rows}")
    assert kernel.abi.names[0] == "x"
    assert "router_logits" in kernel.abi.names
    assert tdm_lds_bytes(rows, 2) <= 320 * 1024


def test_replicated_front_feeds_only_the_grouped_moe_finish():
    with pytest.raises(ValueError, match="grouped-MoE"):
        build_packed_finish(_plan(16), replicated_front=True)


@pytest.mark.parametrize("rows", (1, 16, 40, 48))
def test_replicated_finalize_covers_every_hidden_row_once(rows):
    from atom.models.kimi_k3.mono.kernels.stages import k2_finalize_tdm_gfx1250 as fin

    kernel = fin.build_replicated_finalize(_plan(rows))
    assert kernel.grid == (fin.PAIRS,) == (7168 // 32,)
    assert kernel.abi.names == ("routed_normed", "shared", "up_weight", "output")
