# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""ATOM fused-MoE weight-sync coverage checks.

Run: python -m atomrl.tests.test_atom_moe_weight_sync
"""

import os

import torch
from atomrl.engine.inference.atom_moe_weight_sync import assert_bucket_fully_applied

E, I, H = 4, 6, 8


def _meta(shape, offset, dtype=torch.bfloat16):
    numel = torch.Size(shape).numel()
    return {
        "shape": tuple(shape),
        "dtype": str(dtype),
        "offset": offset,
        "nbytes": numel * dtype.itemsize,
    }


def _moe_bucket():
    """One layer's fused experts plus the neighbours that must be left alone."""
    w13 = _meta((E, 2 * I, H), 128)
    w2 = _meta((E, H, I), 128 + w13["nbytes"])
    return {
        "model.layers.0.mlp.experts.gate_up_proj": w13,
        "model.layers.0.mlp.experts.down_proj": w2,
        "model.layers.0.mlp.gate.weight": _meta((E, H), 0),
        "model.layers.0.self_attn.q_proj.weight": _meta((H, H), 64),
    }


def test_coverage_accepts_a_fully_applied_bucket():
    bucket = _moe_bucket()
    full = len(bucket)
    # Scalar result (one runner) and list result (several) both count.
    assert_bucket_fully_applied([{"cmd": "x", "result": full}] * 8, bucket)
    assert_bucket_fully_applied([{"cmd": "x", "result": [full, full]}], bucket)
    # No usable counts is a warning, not a failure: nothing was observed.
    assert_bucket_fully_applied([], bucket)
    assert_bucket_fully_applied([{"cmd": "x", "result": None}], bucket)


def test_coverage_rejects_a_partially_applied_bucket():
    bucket = _moe_bucket()
    full = len(bucket)
    responses = [{"cmd": "x", "result": full}] * 7 + [{"cmd": "x", "result": full - 2}]
    try:
        assert_bucket_fully_applied(responses, bucket)
    except RuntimeError as exc:
        message = str(exc)
        assert f"{full - 2}/{full}" in message
        assert "1 of 8 engine(s)" in message
    else:
        raise AssertionError(
            "a bucket ATOM only partly applied is the silent-stale-weights bug"
        )

    # True is an int in Python; a boolean ack must not be read as a count of 1.
    assert_bucket_fully_applied([{"cmd": "x", "result": True}], bucket)


def test_coverage_tolerates_a_short_count_when_shards_accumulate():
    """An FP8 rollout under-reports and it is not a fault.

    ATOM accumulates the shards of a fused parameter and requantizes on the
    last one, so q/k/v_proj count as one update and the two before it as none.
    A bucket ending mid-group then reports fewer updates than it holds. This is
    what deadlocked example 4 -- the receiver raised, its socket closed, and the
    sender sat in recv() forever.
    """
    bucket = _moe_bucket()
    short = [{"cmd": "x", "result": len(bucket) - 11}] * 8

    assert_bucket_fully_applied(short, bucket, exact=False)

    try:
        assert_bucket_fully_applied(short, bucket, exact=True)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a BF16 rollout must still be held to exact counts")


def test_coverage_modes_are_configurable():
    bucket = _moe_bucket()
    short = [{"cmd": "x", "result": 0}]
    previous = os.environ.get("ATOMRL_WEIGHT_SYNC_CHECK")
    try:
        for mode in ("warn", "off", "OFF"):
            os.environ["ATOMRL_WEIGHT_SYNC_CHECK"] = mode
            assert_bucket_fully_applied(short, bucket)
    finally:
        if previous is None:
            os.environ.pop("ATOMRL_WEIGHT_SYNC_CHECK", None)
        else:
            os.environ["ATOMRL_WEIGHT_SYNC_CHECK"] = previous


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} ok")
    print("all ATOM fused-MoE weight sync tests passed")
