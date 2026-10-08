# SPDX-License-Identifier: MIT
from copy import deepcopy

import pytest

from atom.spec_decode.calibration import validate_calibration


def profile():
    return {
        "version": 1,
        "identity": {"gpu": "test", "model": "test"},
        "draft_width": 5,
        "max_num_seqs": 4,
        "sts_temperatures": [1.0] * 5,
        "sps_table": [100 / (i + 1) for i in range(25)],
    }


def test_profile_covers_smaller_runtime_batches():
    data = profile()
    validate_calibration(data, data["identity"], width=5, max_batch=2)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("identity", {"gpu": "different"}),
        ("draft_width", 4),
        ("max_num_seqs", 2),
        ("sts_temperatures", [float("nan")] * 5),
        ("sts_temperatures", [-1.0] * 5),
        ("sps_table", [0.0] * 25),
        ("sps_table", [1.0] * 12),
        ("sps_table", list(range(1, 26))),
    ],
)
def test_incompatible_or_invalid_profile_is_rejected(field, value):
    data = profile()
    identity = deepcopy(data["identity"])
    data[field] = value
    with pytest.raises(ValueError):
        validate_calibration(data, identity, width=5, max_batch=4)
