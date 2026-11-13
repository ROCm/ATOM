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


def test_mxfp4_kv_does_not_share_an_fp8_identity(tmp_path, monkeypatch):
    fp8 = identity_of(tmp_path, monkeypatch)
    assert "draft_attention_dtype" not in fp8
    kv_mx = identity_of(tmp_path, monkeypatch, kv_cache_dtype="fp4")
    index_mx = identity_of(tmp_path, monkeypatch, index_cache_dtype="fp4")
    assert len({str(fp8), str(kv_mx), str(index_mx)}) == 3

    data = {**profile(), "identity": fp8}
    with pytest.raises(ValueError):
        validate_calibration(data, kv_mx, width=5, max_batch=4)


@pytest.mark.parametrize("target_dtype", ["fp8", "fp4"])
def test_v41_calibration_requires_matching_draft_precision(
    tmp_path, monkeypatch, target_dtype
):
    def identity(draft_dtype):
        return identity_of(
            tmp_path,
            monkeypatch,
            hf_config=SimpleNamespace(model_type="deepseek_v41_text"),
            kv_cache_dtype=target_dtype,
            speculative_config=SimpleNamespace(draft_kv_dtype=target_dtype),
            dspark=SimpleNamespace(draft_kv_dtype=draft_dtype),
        )

    bf16, fp8 = identity("auto"), identity("fp8")
    assert bf16["draft_attention_dtype"] == "bf16"
    assert fp8["draft_attention_dtype"] == "fp8"
    legacy = {k: v for k, v in bf16.items() if k != "draft_attention_dtype"}
    for current, other in ((bf16, fp8), (fp8, bf16)):
        validate_calibration(
            {**profile(), "identity": current}, current, width=5, max_batch=4
        )
        for incompatible in (other, legacy):
            with pytest.raises(ValueError, match="model/hardware/runtime"):
                validate_calibration(
                    {**profile(), "identity": incompatible},
                    current,
                    width=5,
                    max_batch=4,
                )
