"""PP verify: each row's drafts land right after its anchor in `input_ids`."""

from types import SimpleNamespace

import numpy as np
import pytest

# On the module, not on "aiter" -- see tests/test_verify_ctx_lens.py.
runner_module = pytest.importorskip(
    "atom.model_engine.model_runner",
    reason="requires AITER to import model_runner",
    exc_type=ImportError,
)


def test_drafts_fill_only_their_rows_columns():
    # Rows of 3, 1 (no drafts) and 2 tokens; spec rows padded to width 2.
    lens = np.array([3, 1, 2], dtype=np.int32)
    cu = np.array([0, 3, 4, 6], dtype=np.int32)
    anchors = np.array([100, 0, 0, 101, 102, 0], dtype=np.int32)
    spec = np.array([[10, 11], [-1, -1], [20, -1]], dtype=np.int32)
    batch = SimpleNamespace(
        scheduled_tokens=anchors,
        total_tokens_num_prefill=0,
        total_tokens_num_decode=6,
        total_seqs_num_prefill=0,
        num_spec_step=2,
        scheduled_spec_decode_tokens=spec,
        num_rejected=None,
        num_bonus=None,
    )
    staged = SimpleNamespace(np=np.full(8, -7, dtype=np.int32))
    processor = SimpleNamespace(
        is_deferred_out=False,
        is_pipeline_parallel=True,
        is_pd_producer=False,
        is_spec_run=True,
        input_ids=staged,
        runner=SimpleNamespace(
            attn_metadata_builder=SimpleNamespace(decode_spans=lambda b: (3, lens, cu))
        ),
        _publish_input_ids=lambda n, group: staged.np[:n].copy(),
    )
    ids, _ = runner_module.tokenIDProcessor._stage_input_ids(processor, batch, 3, None)
    np.testing.assert_array_equal(ids, [100, 10, 11, 101, 102, 20])
