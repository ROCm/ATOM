# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""`shuffle_weights` must not move the parameter it shuffles.

A decode CUDA graph captures the address of every weight it reads. An online
weight update writes through `param.data.copy_` and then reshuffles, so the
graph stays valid -- unless the reshuffle rebinds `tensor.data` to aiter's
return value, which hands the parameter a new address while the graph still
holds the old one. The 3D branch always wrote through the existing storage;
the 2D branch rebound.

Values alone cannot see this. These assert on `data_ptr()`.
"""

import pytest
import torch
from torch import nn

if not torch.cuda.is_available():
    pytest.skip("aiter's shuffle_weight needs a ROCm device", allow_module_level=True)

from aiter.ops.shuffle import shuffle_weight

from atom.model_ops.utils import shuffle_weights

DEVICE = torch.device("cuda")


def _param(*shape, dtype=torch.bfloat16):
    return nn.Parameter(
        torch.randn(*shape, dtype=torch.float32, device=DEVICE).to(dtype),
        requires_grad=False,
    )


def test_a_2d_weight_keeps_its_address():
    weight = _param(64, 64)
    # Hold the original tensor, so the caching allocator cannot hand its block
    # back for the shuffle's output and make a rebind look stable.
    original = weight.data

    shuffle_weights(weight)

    assert weight.data_ptr() == original.data_ptr()


def test_a_3d_weight_keeps_its_address():
    weight = _param(4, 64, 64)
    original = weight.data

    shuffle_weights(weight)

    assert weight.data_ptr() == original.data_ptr()


@pytest.mark.parametrize("shape", [(64, 64), (4, 64, 64)])
def test_the_shuffled_bytes_are_aiter_s(shape):
    """Preserving the address must not change what lands in it."""
    weight = _param(*shape)
    original = weight.data.clone()

    shuffle_weights(weight)

    if len(shape) == 2:
        expected = shuffle_weight(original, layout=(16, 16))
    else:
        expected = torch.stack(
            [shuffle_weight(original[i], layout=(16, 16)) for i in range(shape[0])]
        )
    assert torch.equal(weight.data, expected)


def test_shuffling_marks_the_parameter():
    weight = _param(64, 64)

    shuffle_weights(weight)

    assert weight.is_shuffled is True


def test_a_1d_weight_is_refused():
    with pytest.raises(ValueError, match="dim to be 2 or 3"):
        shuffle_weights(_param(64))


def test_a_plain_tensor_is_refused():
    with pytest.raises(TypeError, match="Parameter"):
        shuffle_weights(torch.zeros(64, 64, device=DEVICE))
