# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""CPU checks of the decode-step host trims: `FastLaunch` reuses the compiled
kernel only for a matching specialization and falls back otherwise;
`hip_copy.copy_async` declines what a plain memcpy cannot do."""

from types import SimpleNamespace

import pytest
import torch

from atom.model_ops.triton_fast_launch import FastLaunch
from atom.utils import hip_copy


class _Compiled:
    """A compiled kernel stand-in: `compiled[grid](*args)` launches."""

    def __init__(self, log):
        self.log = log

    def __getitem__(self, grid):
        return lambda *args: self.log.append((grid, args))


class _Kernel:
    """A JITFunction stand-in: `fn[grid](...)` compiles and returns a launcher."""

    def __init__(self):
        self.params = [
            SimpleNamespace(name="x", is_constexpr=False, do_not_specialize=False),
            SimpleNamespace(name="n", is_constexpr=False, do_not_specialize=True),
            SimpleNamespace(name="BLOCK", is_constexpr=True, do_not_specialize=False),
        ]
        self.slow, self.fast = [], []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.slow.append((grid, args, kwargs))
            return _Compiled(self.fast)

        return launch


def test_fast_launch_hits_only_on_same_specialization():
    k = _Kernel()
    launch = FastLaunch(k)
    x = torch.zeros(64, dtype=torch.float32)
    launch((4,), x, 7, BLOCK=16)
    launch((8,), x, 9, BLOCK=16)  # same key: n is do_not_specialize
    assert len(k.slow) == 1
    assert k.fast == [((8, 1, 1), (x, 9, 16))]
    launch((8,), x.to(torch.float16), 9, BLOCK=16)  # dtype changes the key
    launch((8,), x, 9, BLOCK=32)  # constexpr changes the key
    assert len(k.slow) == 3 and len(k.fast) == 1


def test_fast_launch_unknown_argument_takes_the_slow_path():
    k = _Kernel()
    launch = FastLaunch(k)
    launch((1,), object(), 1, BLOCK=16)
    launch((1,), object(), 1, BLOCK=16)
    assert len(k.slow) == 2 and not k.fast


@pytest.mark.parametrize(
    "dst,src",
    [
        # host -> host: not a D2H, whatever the shapes
        (torch.zeros(8, dtype=torch.int32), torch.zeros(8, dtype=torch.int32)),
        (torch.zeros(8, dtype=torch.int32), torch.zeros(8, dtype=torch.int64)),
        (torch.zeros(8, dtype=torch.int32), torch.zeros(9, dtype=torch.int32)),
        (
            torch.zeros(8, 2, dtype=torch.int32).t(),
            torch.zeros(2, 8, dtype=torch.int32),
        ),
    ],
)
def test_hip_copy_declines_what_memcpy_cannot_do(dst, src):
    assert hip_copy.copy_async(dst, src, hip_copy.D2H) is False
