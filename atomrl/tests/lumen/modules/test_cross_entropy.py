# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

from unittest import mock

import torch
from atomrl.lumen.modules.cross_entropy import lumen_parallel_cross_entropy


class TestLumenParallelCrossEntropy:
    def test_passes_correct_args(self):
        logits = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, 128, (2, 32), device="cuda")
        tp_group = object()

        with (
            mock.patch("atomrl.lumen.modules.cross_entropy._parallel_ce") as mock_ce,
            mock.patch(
                "atomrl.lumen.modules.cross_entropy._use_sdma_from_args",
                return_value=False,
            ),
        ):
            lumen_parallel_cross_entropy(
                logits, labels, tp_group, is_cg_capturable=False
            )
            mock_ce.assert_called_once_with(
                logits,
                labels,
                0.0,
                False,
                tp_group,
                -100,
                False,
                False,
                0,
            )

    def test_is_cg_capturable_forwarded(self):
        logits = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, 128, (2, 32), device="cuda")
        tp_group = object()

        with (
            mock.patch("atomrl.lumen.modules.cross_entropy._parallel_ce") as mock_ce,
            mock.patch(
                "atomrl.lumen.modules.cross_entropy._use_sdma_from_args",
                return_value=False,
            ),
        ):
            lumen_parallel_cross_entropy(
                logits, labels, tp_group, is_cg_capturable=True
            )
            mock_ce.assert_called_once_with(
                logits,
                labels,
                0.0,
                False,
                tp_group,
                -100,
                True,
                False,
                0,
            )

    def test_use_sdma_integration(self):
        logits = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, 128, (2, 32), device="cuda")
        tp_group = object()

        with (
            mock.patch("atomrl.lumen.modules.cross_entropy._parallel_ce") as mock_ce,
            mock.patch(
                "atomrl.lumen.modules.cross_entropy._use_sdma_from_args",
                return_value=True,
            ),
        ):
            lumen_parallel_cross_entropy(
                logits, labels, tp_group, is_cg_capturable=False
            )
            mock_ce.assert_called_once()
            call_args = mock_ce.call_args[0]
            assert call_args[-2] is True
