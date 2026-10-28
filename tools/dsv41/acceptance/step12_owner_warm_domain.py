"""Keep593 warm requests and require every bounded packed-owner graph capacity."""

import copy
import json

CAPACITIES = (16, 32, 64, 96, 128, 192)
CANDIDATE_WARM = (
    ("initial_four_owners", 4, 4, 128),
    ("fill_replenish_drain", 512, 256, 1024),
    ("tail64_full_proposal", 64, 64, 128),
    ("tail12_full_proposal", 12, 12, 128),
    ("last_request", 1, 1, 128),
)


def expected_key(template, capacity):
    """Expected views of the same full native binding; never a production key."""
    result = copy.deepcopy(template)
    old_capacity = result[0]
    assert old_capacity in CAPACITIES and result[2] == 6
    assert result[1][1:4] == [[4 * old_capacity], [1], "torch.int32"]
    assert result[3] == [[f"v41_score_bands_{i}", [[0, old_capacity]]] for i in (1, 2)]
    assert [layer for layer, _ in result[4]] == [1, 14]
    result[0] = capacity
    result[1][1] = [4 * capacity]
    result[3] = [[f"v41_score_bands_{i}", [[0, capacity]]] for i in (1, 2)]
    for _, binding in result[4]:
        assert binding[1] == [1, old_capacity, 6144]
        assert binding[2] == [384 * 6144, 6144, 1]
        assert binding[3] == "torch.bfloat16"
        binding[1] = [1, capacity, 6144]
    return result


def check_domain(report, *, complete=True):
    assert report["stable_inputs"] and report["missing_key"] is None
    assert report["coverage"]["misses_after_seal"] == 0
    keys = report["keys"]
    assert keys
    template = keys[0]
    expected = {json.dumps(expected_key(template, capacity)) for capacity in CAPACITIES}
    actual = {json.dumps(key) for key in keys}
    assert (
        len(actual) == len(keys) and actual <= expected
    ), "Unexpected complete owner graph binding"
    capacities = sorted(key[0] for key in keys)
    if complete and actual != expected:
        raise RuntimeError(
            f"Missing owner graph capacities before formal: {sorted(set(CAPACITIES) - set(capacities))}"
        )
    if not complete and 16 not in capacities:
        raise RuntimeError("Initial four-owner warm requests did not create capacity16")
    return {
        "rank": report["rank"],
        "capacities": capacities,
        "full_bindings_verified": True,
    }
