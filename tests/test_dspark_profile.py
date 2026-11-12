# SPDX-License-Identifier: MIT
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from atom.spec_decode.calibration import calibration_identity, validate_calibration


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


def identity_of(tmp_path, monkeypatch, **kw):
    for name in ("config.json", "model.safetensors.index.json"):
        (tmp_path / name).write_text("{}")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "gpu")
    config = SimpleNamespace(
        model=str(tmp_path),
        hf_config=SimpleNamespace(model_type="m"),
        tensor_parallel_size=4,
        kv_cache_dtype="fp8",
        index_cache_dtype="fp8",
        enforce_eager=False,
        # as `Config` resolves it: the target's, unless the draft owns a pool
        speculative_config=SimpleNamespace(draft_kv_dtype="fp8"),
    )
    for key, value in kw.items():
        setattr(config, key, value)
    return calibration_identity(config, 0)


def test_a_draft_dtype_beside_the_targets_changes_the_identity(tmp_path, monkeypatch):
    def with_draft(dtype):
        spec = SimpleNamespace(draft_kv_dtype=dtype)
        return identity_of(tmp_path, monkeypatch, speculative_config=spec)

    # the target's own dtype leaves it as it was, so older profiles still match
    assert with_draft("fp8") == identity_of(tmp_path, monkeypatch)
    assert "draft_kv_cache_dtype" not in with_draft("fp8")
    assert with_draft("bf16") != with_draft("fp8")
