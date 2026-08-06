# SPDX-License-Identifier: MIT
"""Why the Kimi-K3 DSpark draft's MLA decode has to stay in persistent mode."""

import pytest
import torch

pytest.importorskip("aiter")

gpu_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a GPU for the aiter MLA kernel"
)

# The draft blocks 7 tokens at a time and, at TP8, pads its 12 local heads to
# the 16 the MLA kernels require.
DRAFT_BLOCK_WIDTH = 7
DRAFT_PADDED_HEADS = 16


@gpu_only
def test_the_draft_block_has_no_non_persistent_fp8_kernel():
    """The draft shares the engine's fp8 pool, which pins it to persistent mode.

    aiter's non-persistent split-KV table is keyed by ``nhead * max_seqlen_q``
    and has no fp8 entry for the draft's 16x7 -- and asm_mla.cu likewise rejects
    a gqa_ratio=16 fp8 decode wider than 4 outside persistent mode. So the
    metadata builder deciding to skip the persistent work descriptors is not a
    slower fallback for this draft, it is a hard failure, which is why
    `_build_decode` asserts its capacity rather than degrading.
    """
    from aiter import dtypes
    from aiter.mla import get_meta_param

    kwargs = dict(
        num_kv_splits=None,
        bs=1,
        total_kv=1024,
        nhead=DRAFT_PADDED_HEADS,
        dtype=dtypes.fp8,
    )
    # The target verifies 1 + 7 tokens, which the table does cover.
    get_meta_param(max_seqlen_q=DRAFT_BLOCK_WIDTH + 1, **kwargs)
    with pytest.raises(KeyError):
        get_meta_param(max_seqlen_q=DRAFT_BLOCK_WIDTH, **kwargs)
