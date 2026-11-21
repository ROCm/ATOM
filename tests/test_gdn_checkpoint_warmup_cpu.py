# SPDX-License-Identifier: MIT
"""Warmup preserves the self-copy while compiling the decode-image fold."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from aiter_stub import atom_without_aiter


@pytest.mark.parametrize("replayssm", [False, True])
@pytest.mark.parametrize("image_bytes", [None, 0, 16])
def test_gdn_checkpoint_warmup_preserves_self_copy(monkeypatch, replayssm, image_bytes):
    with atom_without_aiter() as load:
        mod = load("atom.model_ops.attentions.gdn_attn")
        base = load("atom.model_ops.attentions.backends")
        copy = load("atom.model_ops.attentions.pool_layout.paged_state_copy")
        plan = copy.plan_segmented_copy([6, 10], [8, 8], image_bytes or 0)
        rows = np.empty((plan.num_spans, 3), dtype=np.int64)
        b = object.__new__(mod.GDNAttentionMetadataBuilder)
        b.replayssm = replayssm
        b.model_runner = SimpleNamespace(
            state_runtime=SimpleNamespace(
                checkpoint_spec=(
                    None
                    if image_bytes is None
                    else SimpleNamespace(units_per_checkpoint=2)
                )
            ),
            mamba_v_cache=torch.empty(1),
        )
        b._checkpoint_copy_plan = lambda: plan
        b._checkpoint_slot_bases = lambda: np.array([[100, 200]], dtype=np.int64)
        b._page_unit_stream_sizes = lambda _units: [8, 8]
        b._page_unit_bases = lambda _units: np.array([[1000, 2000]], dtype=np.int64)
        b._checkpoint_staging = lambda: SimpleNamespace(
            rows=lambda _slot: rows,
            upload=lambda _slot, count: rows[:count].copy(),
        )
        calls = []

        def launch(descriptor, actual_plan):
            assert actual_plan is plan
            calls.append("copy")
            np.testing.assert_array_equal(descriptor[:, 0], [100, 200, 202])
            np.testing.assert_array_equal(descriptor[:, 1], descriptor[:, 0])
            np.testing.assert_array_equal(descriptor[:, 2], [6, 2, 8])

        def materialize(src, dst, steps):
            calls.append("fold")
            assert src.dtype == dst.dtype == steps.dtype == torch.int64
            assert (
                src.device
                == dst.device
                == steps.device
                == b.model_runner.mamba_v_cache.device
            )
            assert src.tolist() == dst.tolist() == [0]
            assert steps.tolist() == [-1]

        monkeypatch.setattr(base, "launch_copy_descriptor", launch)
        monkeypatch.setattr(mod, "launch_copy_descriptor", launch)
        b._materialize = materialize
        b.warmup_per_req_cache()

        expected = ["copy"] if image_bytes else []
        if image_bytes and replayssm:
            expected.append("fold")
        assert calls == expected
