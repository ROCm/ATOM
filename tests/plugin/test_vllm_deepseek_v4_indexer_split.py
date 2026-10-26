# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""DeepSeek-V4 indexer split on prefill-classified steps with no prefill rows.

A prompt of 128*m + 1 tokens re-sent after a full prefix hit, a fresh prompt of
8192*k + 1 tokens, or a 1-token prompt has a last prefill step of one token. The
bridge classifies that step as prefill, ``_populate_indexer`` peels every
1-token row onto the paged decode path, and nothing is left for the dense path.
Calling the dense path anyway gathered one committed row through an empty block
table on the GPU (memory access fault).
"""

import numpy as np
import pytest
import torch

pytest.importorskip("aiter", reason="the V4 model module reaches AITER ops")

from atom.plugin.vllm import deepseek_v4_bridge as bridge


def test_prefilling_row_of_decode_length_ends_the_decode_group():
    q1 = np.ones(64, dtype=np.int32)
    # 63 real decodes then the 1-token extend: like native, the extend is a
    # prefill row, so the dense group is not empty.
    pref = torch.zeros(64, dtype=torch.bool)
    pref[63] = True
    assert bridge._indexer_decode_group(q1, 1, pref) == (63, 1)
    # The extend alone.
    assert bridge._indexer_decode_group(
        np.ones(1, dtype=np.int32), 1, torch.tensor([True])
    ) == (0, 1)
    # Extend first in the decode-length run: everything goes dense.
    pref = torch.zeros(4, dtype=torch.bool)
    pref[0] = True
    assert bridge._indexer_decode_group(np.ones(4, dtype=np.int32), 1, pref) == (0, 1)
    # Unchanged when nothing is prefilling: run of equal decode lengths, then a
    # longer prefill row.
    lens = np.array([1, 1, 1, 37], dtype=np.int32)
    assert bridge._indexer_decode_group(
        lens, 1, torch.tensor([False, False, False, True])
    ) == (3, 1)
    assert bridge._indexer_decode_group(
        np.array([4, 4, 1, 4], dtype=np.int32), 4, None
    ) == (2, 4)
    assert bridge._indexer_decode_group(np.array([37, 1], dtype=np.int32), 1, None) == (
        0,
        1,
    )
