# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Version stamping on the ZMQ CUDA-IPC weight sync control plane.

CPU-only: the sender's constructor does not touch CUDA and the code under test
is pure dict manipulation, so the actual roundtrip (which needs a GPU and a
live vLLM model) is out of scope here.

``check_bucket_version`` is tested directly rather than through
``BucketedWeightReceiver`` because ATOM's hand-rolled receive loops call the
same function -- both IPC receivers go through this one code path.

Run: python -m atomrl.tests.test_bucketed_weight_version
"""

from atomrl.engine.inference.bucketed_weight_transfer import (
    BucketedWeightSender,
    check_bucket_version,
)

HANDLE = "inproc://lumen-test-weight-version"


def _sender(version):
    return BucketedWeightSender(zmq_handle=HANDLE, bucket_size_mb=1, version=version)


def test_control_message_carries_the_version():
    message = _sender(7)._control({"w": {}}, is_last=True)
    assert message == {"bucket_meta": {"w": {}}, "is_last": True, "version": 7}


def test_control_message_omits_version_when_unset():
    # ATOM's hand-rolled receiver reads only bucket_meta/is_last; senders that
    # were given no version must keep producing exactly the old message.
    message = _sender(None)._control({}, is_last=False)
    assert message == {"bucket_meta": {}, "is_last": False}


def test_matching_version_is_accepted():
    check_bucket_version({"version": 7}, 7)


def test_repeated_version_is_accepted():
    # _sync_weights_ipc can run several times within one global_step, so the
    # check is equality and not strict monotonicity.
    for _ in range(3):
        check_bucket_version({"version": 7}, 7)


def test_mismatched_version_raises():
    try:
        check_bucket_version({"version": 6}, 7)
    except RuntimeError as exc:
        assert "expected 7" in str(exc) and "got 6" in str(exc)
    else:
        raise AssertionError("a stale bucket must not be accepted")


def test_missing_version_raises_when_one_is_expected():
    # An unstamped bucket means the sender is not the one we handshook with.
    try:
        check_bucket_version({}, 7)
    except RuntimeError as exc:
        assert "got None" in str(exc)
    else:
        raise AssertionError("an unstamped bucket must not be accepted")


def test_no_expectation_disables_the_check():
    check_bucket_version({}, None)
    check_bucket_version({"version": 999}, None)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} ok")
    print("all bucketed weight version tests passed")
