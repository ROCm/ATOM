# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import math

import pytest
import torch
from atomrl.engine.inference import weight_integrity


def test_tensor_integrity_reports_first_nonfinite_tensor() -> None:
    report = weight_integrity.tensor_integrity(
        "layers.2.weight_scale_inv",
        torch.tensor([1.0, float("nan"), float("inf"), -2.0]),
    )

    assert report["name"] == "layers.2.weight_scale_inv"
    assert report["numel"] == 4
    assert report["finite_count"] == 2
    assert report["nan_count"] == 1
    assert report["inf_count"] == 1
    assert report["finite_min"] == -2.0
    assert report["finite_max"] == 1.0
    assert report["all_finite"] is False


def test_tensor_integrity_supports_float8() -> None:
    value = torch.tensor([1.0, -2.0], dtype=torch.float32).to(torch.float8_e4m3fnuz)

    report = weight_integrity.tensor_integrity("fp8_weight", value)

    assert report["all_finite"] is True
    assert report["finite_min"] == -2.0
    assert report["finite_max"] == 1.0


def test_tensor_integrity_detects_float8_fnuz_nan_encoding() -> None:
    value = torch.tensor([1.0, float("nan")], dtype=torch.float32).to(
        torch.float8_e4m3fnuz
    )

    report = weight_integrity.tensor_integrity("fp8_weight", value)

    assert report["nan_count"] == 1
    assert report["all_finite"] is False


def test_tensor_integrity_reports_unmaterialized_meta_tensor() -> None:
    report = weight_integrity.tensor_integrity(
        "pending_weight",
        torch.empty(3, device="meta"),
    )

    assert report["materialized"] is False
    assert report["all_finite"] is False
    assert report["checksum"] is None


def test_scan_named_tensors_stops_at_first_bad_tensor() -> None:
    report = weight_integrity.scan_named_tensors(
        [
            ("good", torch.ones(2)),
            ("bad", torch.tensor([math.nan])),
            ("unvisited", torch.tensor([math.inf])),
        ],
        stop_on_first_bad=True,
    )

    assert report["tensor_count"] == 2
    assert report["first_bad"]["name"] == "bad"
    assert report["all_finite"] is False


def test_sampled_tensor_integrity_preserves_original_shape() -> None:
    report = weight_integrity.sampled_tensor_integrity(
        "large",
        torch.arange(20, dtype=torch.float32).reshape(4, 5),
        max_samples=4,
    )

    assert report["shape"] == [4, 5]
    assert report["numel"] == 20
    assert report["sample_count"] == 4
    assert report["all_finite"] is True


def test_scan_fp8_scales_rejects_nonpositive_values() -> None:
    report = weight_integrity.scan_fp8_scales(
        [
            ("layer.weight", torch.ones(2)),
            ("layer.weight_scale_inv", torch.tensor([0.5, 0.0, -1.0])),
        ]
    )

    assert report["scale_tensor_count"] == 1
    assert report["nonpositive_count"] == 2
    assert report["all_valid"] is False
    assert report["first_bad"] == "layer.weight_scale_inv"


def test_require_finite_stream_names_the_bad_stage_and_tensor() -> None:
    stream = weight_integrity.require_finite_stream(
        [
            ("good", torch.ones(1)),
            ("bad_scale", torch.tensor([math.nan])),
        ],
        stage="rdma_recv",
    )

    assert next(stream)[0] == "good"
    with pytest.raises(
        FloatingPointError,
        match="stage=rdma_recv tensor=bad_scale",
    ):
        next(stream)
