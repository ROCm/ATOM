# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""DraftGraph's persistent outputs (CPU): views of registered storage come back without a clone, every other
tensor is cloned, staged-input aliases are returned as before."""

import torch

from atom.spec_decode.draft_graph import _owned


def test_owned_persistent_and_clone():
    buf = torch.zeros(4, 8, dtype=torch.int32)
    other = torch.ones(4, 7, dtype=torch.int32)
    staged = torch.arange(4, dtype=torch.int32)
    persistent = (buf.untyped_storage().data_ptr(),)
    ids = buf[:1, 1:]
    out = _owned((ids, other, staged, None), (staged,), persistent)
    assert (
        out[0].data_ptr() == ids.data_ptr()
    )  # a view of the registered storage: no clone
    assert out[1].data_ptr() != other.data_ptr() and torch.equal(
        out[1], other
    )  # cloned
    assert out[2].data_ptr() == staged.data_ptr()  # a staged input: returned as is
    assert out[3] is None
    # without the registration the same view is cloned (rF9 behaviour)
    out2 = _owned((ids,), ())
    assert out2[0].data_ptr() != ids.data_ptr() and torch.equal(out2[0], ids)
