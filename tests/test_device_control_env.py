# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`set_device_control_env_var` must stay inert, and these tests say why.

Making it set a real `HIP_VISIBLE_DEVICES` looks like an obvious fix -- the
function is named for it and currently sets a variable nothing reads. It was
tried and reverted, because device placement already has an owner:
`ModelRunner._setup_device_and_distributed` computes an ABSOLUTE index,
`local_dp_rank * tp_size + rank`, and selects `cuda:{that}`.

A visible-device mask renumbers the child's devices, so the two offsets
compound. With `-dp 4 -tp 2`, DP rank 1 gets the mask "2,3" -- two visible
GPUs -- and then asks for `cuda:2`, which no longer exists. Startup dies on
the `local_device_rank >= torch.cuda.device_count()` check. Every multi-GPU DP
run breaks.

Either the mask or the absolute index owns placement, never both. These tests
fail if someone re-applies the "obvious" fix without also converting
`_setup_device_and_distributed` to mask-relative indexing.
"""

import os
from types import SimpleNamespace

from atom.utils import set_device_control_env_var

PLACEHOLDER = "VLLM_DEVICE_CONTROL_ENV_VAR_PLACEHOLDER"


def _config(tp_size):
    return SimpleNamespace(tensor_parallel_size=tp_size)


class TestStaysInert:
    def test_does_not_set_a_real_visible_device_mask(self):
        """The regression guard: a mask here double-offsets the child's device."""
        before = os.environ.get("HIP_VISIBLE_DEVICES")
        with set_device_control_env_var(_config(2), 1):
            assert os.environ.get("HIP_VISIBLE_DEVICES") == before, (
                "setting HIP_VISIBLE_DEVICES here conflicts with the absolute "
                "cuda:{local_dp_rank*tp+rank} index ModelRunner selects; "
                "-dp 4 -tp 2 would mask DP rank 1 to 2 GPUs then ask for cuda:2"
            )

    def test_does_not_set_cuda_visible_devices_either(self):
        before = os.environ.get("CUDA_VISIBLE_DEVICES")
        with set_device_control_env_var(_config(4), 0):
            assert os.environ.get("CUDA_VISIBLE_DEVICES") == before


class TestContextManagerHygiene:
    def test_restores_a_preexisting_value_on_exit(self, monkeypatch):
        monkeypatch.setenv(PLACEHOLDER, "sentinel")
        with set_device_control_env_var(_config(1), 0):
            pass
        assert (
            os.environ[PLACEHOLDER] == "sentinel"
        ), "the context manager must not leak its value past the with-block"

    def test_leaves_no_variable_behind_when_none_existed(self, monkeypatch):
        monkeypatch.delenv(PLACEHOLDER, raising=False)
        with set_device_control_env_var(_config(2), 0):
            pass
        assert PLACEHOLDER not in os.environ
