# SPDX-License-Identifier: MIT
from copy import deepcopy

import pytest

from atom.spec_decode.calibration import (
    _PLAIN_TP,
    deployment_identity,
    validate_calibration,
)


def profile():
    return {
        "version": 1,
        "identity": {
            "gpu": "test",
            "model": "test",
            "graph_mode": "TR_HYBRID",
            "compilation_level": 0,
        },
        "draft_width": 5,
        "max_num_seqs": 4,
        "sts_temperatures": [1.0] * 5,
        "sps_table": [100 / (i + 1) for i in range(25)],
    }


def test_profile_covers_smaller_runtime_batches():
    data = profile()
    validate_calibration(data, data["identity"], width=5, max_batch=2)


def test_profile_identity_binds_graph_mode_and_compilation_level(tmp_path, monkeypatch):
    from types import SimpleNamespace as NS

    import torch

    from atom.config import CUDAGraphMode

    for name in ["config.json", "model.safetensors.index.json"]:
        (tmp_path / name).write_text("{}")
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "test-gpu")
    config = NS(
        model=str(tmp_path),
        hf_config=NS(model_type="deepseek_v41_text"),
        tensor_parallel_size=2,
        kv_cache_dtype="bf16",
        index_cache_dtype="fp4",
        enforce_eager=False,
        compilation_config=NS(level=0, cudagraph_mode=CUDAGraphMode.TR_HYBRID),
    )
    identity = deployment_identity(config, "cpu")
    data = profile()
    data["identity"] = identity
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    with pytest.raises(ValueError, match="does not match"):
        validate_calibration(
            data, deployment_identity(config, "cpu"), width=5, max_batch=4
        )
    config.compilation_config.cudagraph_mode = CUDAGraphMode.TR_HYBRID
    config.compilation_config.level = 3
    with pytest.raises(ValueError, match="does not match"):
        validate_calibration(
            data, deployment_identity(config, "cpu"), width=5, max_batch=4
        )


def test_legacy_tp_profile_cannot_be_reused_for_dp_or_ep():
    data = profile()
    defaults = {
        "data_parallel_size": 1,
        "dp_attention": False,
        "expert_parallel": False,
        "moe_all2all_backend": "auto",
        "moe_ep_flatten_tp_across_dp": False,
    }
    identity = {**data["identity"], "parallelism": defaults}
    validate_calibration(data, identity, width=5, max_batch=4)
    for field, value in [
        ("data_parallel_size", 2),
        ("dp_attention", True),
        ("expert_parallel", True),
        ("moe_all2all_backend", "rccl"),
        ("moe_ep_flatten_tp_across_dp", True),
    ]:
        changed = {**identity, "parallelism": {**defaults, field: value}}
        with pytest.raises(ValueError, match="plain TP"):
            validate_calibration(data, changed, width=5, max_batch=4)
        measured = {**data, "identity": changed}
        validate_calibration(measured, changed, width=5, max_batch=4)


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("identity", {"gpu": "different"}),
        ("draft_width", 4),
        ("max_num_seqs", 2),
        ("sts_temperatures", [float("nan")] * 5),
        ("sps_table", [0.0] * 25),
        ("sps_table", list(range(1, 26))),
    ],
)
def test_incompatible_or_invalid_profile_is_rejected(field, value):
    data = profile()
    identity = deepcopy(data["identity"])
    data[field] = value
    with pytest.raises(ValueError):
        validate_calibration(data, identity, width=5, max_batch=4)


def test_a_profile_in_mains_format_still_loads():
    """main's profiles carry neither parallelism nor graph fields: they load
    on plain TP as they do on main, and a field they do carry still binds."""
    data = profile()
    identity = {**data["identity"], "parallelism": dict(_PLAIN_TP)}
    legacy = {
        **data,
        "identity": {
            k: v
            for k, v in data["identity"].items()
            if k not in ("graph_mode", "compilation_level")
        },
    }
    validate_calibration(legacy, identity, width=5, max_batch=4)
    validate_calibration(
        legacy, {**identity, "graph_mode": "FULL"}, width=5, max_batch=4
    )
    with pytest.raises(ValueError, match="does not match"):
        validate_calibration(
            data, {**identity, "graph_mode": "FULL"}, width=5, max_batch=4
        )


def test_identity_resolves_a_hub_id_to_its_local_snapshot(tmp_path, monkeypatch):
    """`--model org/name` is what Config.model keeps; the identity hashes the
    snapshot's files, not a relative path that does not exist."""
    from types import SimpleNamespace as NS

    import torch

    from atom.config import CUDAGraphMode

    for name in ("config.json", "model.safetensors.index.json"):
        (tmp_path / name).write_text("{}")
    monkeypatch.setattr(
        "atom.model_loader.weight_utils.local_model_dir",
        lambda model: str(tmp_path) if model == "org/name" else model,
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "test-gpu")
    config = NS(
        model="org/name",
        hf_config=NS(model_type="deepseek_v41_text"),
        tensor_parallel_size=1,
        kv_cache_dtype="bf16",
        index_cache_dtype="fp4",
        enforce_eager=False,
        compilation_config=NS(level=0, cudagraph_mode=CUDAGraphMode.TR_HYBRID),
    )
    identity = deployment_identity(config, 0)
    assert set(identity["model_files"]) == {
        "config.json",
        "model.safetensors.index.json",
    }
