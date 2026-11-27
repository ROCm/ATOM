# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

import torch
from torch import nn


class TestLumenGraphedCallable:

    def test_construction(self):
        from atomrl.lumen.utils.hip_graphs import LumenGraphedCallable

        def fn(x):
            return x * 2

        sample = torch.randn(4, 4)
        # Won't capture without CUDA, but should not crash
        gc = LumenGraphedCallable(fn, (sample,))
        assert gc is not None

    def test_call_without_cuda(self):
        from atomrl.lumen.utils.hip_graphs import LumenGraphedCallable

        def fn(x):
            return x * 2

        sample = torch.randn(4, 4)
        gc = LumenGraphedCallable(fn, (sample,))
        result = gc(torch.ones(4, 4))
        expected = torch.ones(4, 4) * 2
        torch.testing.assert_close(result, expected)

    def test_reset(self):
        from atomrl.lumen.utils.hip_graphs import LumenGraphedCallable

        def fn(x):
            return x + 1

        gc = LumenGraphedCallable(fn, (torch.randn(4, 4),))
        gc.reset()
        assert gc._graph is None


class TestLumenGraphedModule:

    def test_construction(self):
        from atomrl.lumen.utils.hip_graphs import LumenGraphedModule

        module = nn.Linear(8, 4)
        gm = LumenGraphedModule(module, enabled=False)
        assert gm.module is module

    def test_forward_disabled(self):
        from atomrl.lumen.utils.hip_graphs import LumenGraphedModule

        module = nn.Linear(8, 4)
        gm = LumenGraphedModule(module, enabled=False)
        x = torch.randn(2, 8)
        out = gm(x)
        assert out.shape == (2, 4)

    def test_release_graph(self):
        from atomrl.lumen.utils.hip_graphs import LumenGraphedModule

        module = nn.Linear(8, 4)
        gm = LumenGraphedModule(module, enabled=False)
        gm.release_graph()
        assert gm._graphed is None


class TestMakeGraphedCallables:

    def test_multiple_callables(self):
        from atomrl.lumen.utils.hip_graphs import lumen_make_graphed_callables

        def fn1(x):
            return x * 2

        def fn2(x):
            return x + 1

        args1 = (torch.randn(4, 4),)
        args2 = (torch.randn(4, 4),)
        result = lumen_make_graphed_callables([fn1, fn2], [args1, args2])
        assert len(result) == 2
