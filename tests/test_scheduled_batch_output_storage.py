# SPDX-License-Identifier: MIT
"""Queued results must not retain tensor-backed NumPy storage."""

import gc
import pickle
import weakref

import numpy as np
import pytest
import torch

from atom.model_engine.scheduler import ScheduledBatchOutput


@pytest.mark.parametrize("field", ["draft_token_ids", "num_rejected", "num_bonus"])
@pytest.mark.parametrize("strided", [False, True])
def test_tensor_owner_released_before_transport_preserving_values(field, strided):
    tensor = torch.arange(12, dtype=torch.int32).reshape(3, 4)
    view = tensor.numpy()
    if strided:
        view = view[:, ::2]
    root = view
    while isinstance(root.base, np.ndarray):
        root = root.base
    owner = weakref.ref(root.base)
    expected = pickle.dumps(view)
    kwargs = {
        "req_ids": [7],
        "token_ids": [(9,)],
        "draft_token_ids": None,
        "num_rejected": None,
        "num_bonus": None,
    }
    kwargs[field] = view
    output = ScheduledBatchOutput(**kwargs)
    del kwargs, root, view, tensor
    gc.collect()
    assert owner() is None
    assert pickle.dumps(getattr(output, field)) == expected
    assert getattr(output, field).dtype == np.int32


def test_owned_arrays_and_empty_outputs_keep_their_storage():
    array = np.arange(4, dtype=np.int32)
    output = ScheduledBatchOutput([7], [(9,)], array, None, None)
    assert output.num_rejected is array
    assert output.num_bonus is None and output.draft_token_ids is None
    assert output.get_idx(7) == 0 and output.get_idx(8) is None
