# SPDX-License-Identifier: MIT
from copy import deepcopy

import pytest

from atom.spec_decode.calibration import calibration_identity, validate_calibration


def test_single_node_worker_rank_retains_the_full_node_dp_slice():
    from atom.spec_decode.calibration import single_node_parallel_config

    for rank in range(4):
        parallel, tp, local = single_node_parallel_config(4, 2, rank)
        assert parallel.data_parallel_size_local == 2
        assert parallel.data_parallel_rank == rank // 2
        assert parallel.data_parallel_rank_local * tp + local == rank
    with pytest.raises(ValueError, match="rank layout"):
        single_node_parallel_config(3, 2, 0)


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
    identity = calibration_identity(config, "cpu")
    data = profile()
    data["identity"] = identity
    config.compilation_config.cudagraph_mode = CUDAGraphMode.FULL
    with pytest.raises(ValueError, match="does not match"):
        validate_calibration(
            data, calibration_identity(config, "cpu"), width=5, max_batch=4
        )
    config.compilation_config.cudagraph_mode = CUDAGraphMode.TR_HYBRID
    config.compilation_config.level = 3
    with pytest.raises(ValueError, match="does not match"):
        validate_calibration(
            data, calibration_identity(config, "cpu"), width=5, max_batch=4
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
        with pytest.raises(ValueError, match="does not match"):
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
