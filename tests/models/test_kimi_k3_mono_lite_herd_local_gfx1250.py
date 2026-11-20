import numpy as np
import pytest

from atom.models.kimi_k3.mono.kernels.execution_plan import Gfx1250KimiBuildPlan
from atom.models.kimi_k3.mono.kernels.stages import k2_finish_persistent_gfx1250 as fin
from atom.models.kimi_k3.mono.kernels.stages import moe_gfx1250

KP1 = 17
SWITCHES = (
    "ATOM_KIMI_MONO_HERD_OWN_LOCAL",
    "ATOM_KIMI_MONO_LITE_ROUTE_XCD",
    "ATOM_KIMI_MONO_LITE_SD_WIDE",
)


def _plan(rows):
    return Gfx1250KimiBuildPlan(
        threads=256, blocks=256, lds_bytes=320 * 1024, tokens=rows, nblocks=4
    )


def _finish(rows, **kwargs):
    herd = kwargs.pop("herd", rows >= 16)
    return fin.build_packed_finish(
        _plan(rows),
        gu_interleave=True,
        include_routed=False,
        replicated_front=True,
        herd=herd,
        fp8_shared_down=True,
        precompute_herd=herd,
        parallel_herd=kwargs.pop("parallel_herd", herd),
        **kwargs,
    )


@pytest.fixture
def switches(monkeypatch):
    """Only the named switches (name suffixes) on."""

    def on(*names):
        for name in SWITCHES:
            monkeypatch.setenv(name, "1" if any(name.endswith(n) for n in names) else "0")

    return on


@pytest.mark.parametrize("rows", (16, 24, 32, 40, 64))
def test_an_own_local_finish_is_a_build_of_its_own(switches, rows):
    switches()
    base = _finish(rows)
    switches("OWN_LOCAL")
    local = _finish(rows)
    assert local.name == base.name + "_hol1"
    assert local.abi == base.abi
    assert local.grid == base.grid
    assert local.scratch_bytes == base.scratch_bytes


def test_only_a_parallel_herd_finish_without_a_poll_limit_keeps_its_own(switches):
    switches("OWN_LOCAL")
    assert "_hol" not in _finish(40, parallel_herd=False).name
    assert "_hol" not in _finish(40, herd_poll_limit=1024).name
    assert "_hol" not in _finish(8).name


@pytest.mark.parametrize("rows", (16, 24, 32, 40, 64))
def test_the_routing_ctas_take_the_first_xcd_only_up_to_32_rows(switches, rows):
    switches("OWN_LOCAL", "ROUTE_XCD")
    assert ("_rx" in _finish(rows).name) == (rows <= 32)
    switches("ROUTE_XCD")
    assert "_rx" not in _finish(rows).name


@pytest.mark.parametrize("rows", range(1, 33))
def test_the_first_xcd_routing_ctas_leave_the_shared_down_its_order(rows):
    routing, shared = fin.routing_ctas(rows, 256, True)
    assert routing == [8 * k for k in range(rows)]
    assert sorted(routing + shared) == list(range(256))
    assert len(shared) == 256 - rows == len(set(shared))
    assert fin.routing_ctas(rows, 256, False) == (
        list(range(256 - rows, 256)),
        list(range(256 - rows)),
    )


@pytest.mark.parametrize("rows", (8, 16, 40, 64))
def test_a_wide_store_finish_is_a_build_of_its_own(switches, rows):
    switches()
    base = _finish(rows)
    switches("SD_WIDE")
    wide = _finish(rows)
    assert wide.name == base.name + "_sw"
    assert wide.scratch_bytes == base.scratch_bytes


def _herd_reference(ids, values, pops, rsf):
    """``_herd_drop_wave``'s routes: the drop the least (popularity, value,
    id), the kept sum in candidate order, places by id among the kept."""
    drop = min(range(KP1), key=lambda j: (pops[j], values[j], ids[j]))
    kept_sum = np.float32(0.0)
    for k in range(KP1):
        kept_sum = np.float32(kept_sum + (values[k] if k != drop else np.float32(0)))
    scale = np.float32(rsf / max(kept_sum, np.float32(1e-20)))
    kept = [j for j in range(KP1) if j != drop]
    places = {j: sum(ids[i] < ids[j] for i in kept) for j in kept}
    return drop, places, {j: np.float32(values[j] * scale) for j in kept}


def _herd_local(ids, values, pops, rsf):
    """``stage_herd_token_local``'s: every drop's scale and the ranks first,
    then the least (popularity << 5 | rank by value and id)."""
    scales, by_value, by_id = [], [], []
    for lane in range(KP1):
        kept_sum = np.float32(0.0)
        for k in range(KP1):
            kept_sum = np.float32(kept_sum + (values[k] if k != lane else np.float32(0)))
        scales.append(np.float32(rsf / max(kept_sum, np.float32(1e-20))))
        by_value.append(
            sum(
                (values[k] < values[lane])
                or (values[k] == values[lane] and ids[k] < ids[lane])
                for k in range(KP1)
            )
        )
        by_id.append(sum(ids[k] < ids[lane] for k in range(KP1)))
    keys = [(pops[j] << 5) | by_value[j] for j in range(KP1)]
    drop = keys.index(min(keys))
    kept = [j for j in range(KP1) if j != drop]
    places = {j: by_id[j] - (ids[drop] < ids[j]) for j in kept}
    return drop, places, {j: np.float32(values[j] * scales[drop]) for j in kept}


@pytest.mark.parametrize("seed", range(64))
def test_the_local_herd_drop_is_the_wave_drop(seed):
    rng = np.random.default_rng(seed)
    ids = [int(i) for i in rng.choice(896, KP1, replace=False)]
    # ties in value and popularity, as a batch of a few rows makes them
    values = [np.float32(v) for v in rng.choice(rng.random(6, dtype=np.float32), KP1)]
    pops = [int(p) for p in rng.integers(1, 4 if seed % 2 else 64, KP1)]
    rsf = np.float32(rng.choice([1.0, 2.827]))
    ref, local = _herd_reference(ids, values, pops, rsf), _herd_local(ids, values, pops, rsf)
    assert ref[0] == local[0]
    assert ref[1] == local[1]
    assert {j: w.tobytes() for j, w in ref[2].items()} == {
        j: w.tobytes() for j, w in local[2].items()
    }
    assert sorted(local[1].values()) == list(range(KP1 - 1))


def test_the_popularity_key_holds_a_batch_of_64_rows():
    # a token's 17 ids are distinct: an expert's popularity is at most the rows
    assert (64 << 5 | (KP1 - 1)) < 0x7FFFFFFF
    assert moe_gfx1250.TOPK + 1 == KP1
