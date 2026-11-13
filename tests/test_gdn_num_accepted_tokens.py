# SPDX-License-Identifier: MIT
"""`prepare_num_accepted_tokens` publishes the row in one copy, not one per seq.

Each element assignment from a host scalar into a device tensor is a blocking
H2D copy; the path used to issue one per request on every spec-decode step.
These run on CPU: they pin the values to what the per-element path wrote, the
destination to its original storage (captured graphs read it by address), and
the write count to one.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode


def _import_gdn_attn():
    """Import the builder module on a host with no visible GPU.

    AITER's Triton helpers read the GPU arch at import (FP8 dtype choice), and
    with no device the probe fails. Nothing here launches a kernel, so lend it
    an arch for the import and put the probe back afterwards.
    """
    arch_info = pytest.importorskip("aiter.ops.triton.utils._triton.arch_info")
    lent = arch_info._CACHED_ARCH is None
    if lent:
        arch_info._CACHED_ARCH = "gfx950"
    try:
        return pytest.importorskip("atom.model_ops.attentions.gdn_attn")
    finally:
        if lent:
            arch_info._CACHED_ARCH = None


gdn_attn = _import_gdn_attn()

MAX_BS = 8


def make_builder(num_bonus):
    builder = object.__new__(gdn_attn.GDNAttentionMetadataBuilder)
    builder.num_accepted_tokens = torch.ones(MAX_BS, dtype=torch.int32)
    builder.model_runner = SimpleNamespace(
        tokenID_processor=SimpleNamespace(num_bonus=num_bonus)
    )
    return builder


def per_element_reference(dst, num_bonus):
    """The path this replaced, kept as the oracle."""
    dst.fill_(1)
    if num_bonus is None:
        return
    for idx, nb in enumerate(num_bonus):
        dst[idx] = nb + 1


class InPlaceWrites(TorchDispatchMode):
    def __init__(self, storage_ptr):
        super().__init__()
        self.storage_ptr = storage_ptr
        self.writes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = func._schema.name
        target = args[0] if args else None
        if (
            name.endswith("_")
            and isinstance(target, torch.Tensor)
            and target.untyped_storage().data_ptr() == self.storage_ptr
        ):
            self.writes.append(name)
        return func(*args, **(kwargs or {}))


# A larger batch, a smaller one after it (stale tail must go back to 1),
# a full one, an empty one, a list as the tests elsewhere pass, and None.
STEPS = [
    np.array([3, 0, 2, 1, 3], dtype=np.int32),
    np.array([2], dtype=np.int32),
    np.array([0, 1, 2, 3, 3, 2, 1, 0], dtype=np.int32),
    np.array([], dtype=np.int32),
    [1, 3],
    None,
]


def test_values_match_the_per_element_path_across_batches():
    builder = make_builder(None)
    ptr = builder.num_accepted_tokens.data_ptr()
    oracle = torch.ones(MAX_BS, dtype=torch.int32)
    for num_bonus in STEPS:
        builder.model_runner.tokenID_processor.num_bonus = num_bonus
        builder.prepare_num_accepted_tokens(batch=None)
        per_element_reference(oracle, num_bonus)
        assert builder.num_accepted_tokens.tolist() == oracle.tolist(), num_bonus
        # Captured graphs and replayssm_commit read this tensor by address.
        assert builder.num_accepted_tokens.data_ptr() == ptr


@pytest.mark.parametrize("bs", [1, 3, 6, MAX_BS])
def test_one_write_regardless_of_batch_size(bs):
    builder = make_builder(np.arange(bs, dtype=np.int32) % 4)
    mode = InPlaceWrites(builder.num_accepted_tokens.untyped_storage().data_ptr())
    with mode:
        builder.prepare_num_accepted_tokens(batch=None)
    assert mode.writes == ["aten::copy_"]
