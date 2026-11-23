# SPDX-License-Identifier: MIT
"""Real sparse MLA reads remain valid beside raw recurrent-state images."""

import struct

import pytest
import torch
from test_state_checkpoint import ckpt_config, make_block_manager

from atom.model_engine.page_unit_checkpoint import PagedStateCheckpointSpec
from atom.model_engine.state_runtime import StateRuntime, StateTransfer

pytestmark = pytest.mark.skipif(
    not torch.version.hip or not torch.cuda.is_available(), reason="requires a ROCm GPU"
)


@pytest.mark.parametrize("length", [17, 65, 129])
@pytest.mark.parametrize("mapped_null", [False, True])
def test_raw_state_does_not_change_sparse_padding_reads(length, mapped_null):
    from aiter import dtypes
    from aiter.mla import mla_decode_fwd

    from atom.model_ops.attention_mla import triton_gather_kv_indices_sparse

    block, dim = 64, 576
    unit_bytes = block * dim
    spec = PagedStateCheckpointSpec(
        unit_bytes, 3 * unit_bytes, "null-page-gpu", image_bytes=3 * unit_bytes
    )
    bm = make_block_manager(
        ckpt_config(enable_prefix_caching=False, num_kvcache_blocks=8),
        state_runtime=StateRuntime(
            transfer=StateTransfer.copy(spec.layout_id), checkpoint_spec=spec
        ),
    )
    units = bm.paged_state_checkpoints.reserve_transfer_units("image")
    pages = []
    for _ in range((length + block - 1) // block):
        page = bm.kv.pop()
        bm.kv.allocate(page)
        pages.append(page)
    torch.manual_seed(19)
    q = (torch.randn(1, 128, dim, device="cuda") * 0.1).to(dtypes.fp8)
    q[:, 96:] = 0  # Kimi's active heads followed by the assembly width's padding.
    logical = (torch.randn(length, dim, device="cuda") * 0.1).to(dtypes.fp8)
    cache = torch.zeros(8 * block, 1, 1, dim, device="cuda", dtype=dtypes.fp8)
    # A finite FP32 state whose bytes contain FP8 NaN codes.
    value = struct.unpack("<f", struct.pack("<I", 0x3F007FC0))[0]
    state = torch.full((spec.image_bytes // 4,), value, device="cuda")
    assert torch.isfinite(state).all()
    byte_pages = cache.view(torch.uint8).view(8, unit_bytes)
    for offset, page in enumerate(units):
        byte_pages[page].copy_(
            state.view(torch.uint8)[offset * unit_bytes : (offset + 1) * unit_bytes]
        )
    slots = torch.tensor(
        [pages[i // block] * block + i % block for i in range(length)],
        device="cuda",
        dtype=torch.int32,
    )
    cache.view(torch.uint8).view(-1, dim)[slots.long()] = logical.view(torch.uint8)
    ptr = torch.tensor([0, length], device="cuda", dtype=torch.int32)
    indices = torch.zeros(2048, device="cuda", dtype=torch.int32)
    if mapped_null:
        selected = torch.arange(2048, device="cuda", dtype=torch.int32)[None, :]
        selected[:, length - 1 :] = -1
        indices = triton_gather_kv_indices_sparse(
            ptr,
            torch.zeros(1, device="cuda", dtype=torch.int32),
            selected,
            slots,
            ptr,
            out=indices,
        )
    else:
        indices[:length] = slots
    qptr = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    last = torch.ones(1, device="cuda", dtype=torch.int32)
    scale = torch.ones(1, device="cuda")

    def attention():
        out = torch.empty(1, 128, 512, device="cuda", dtype=torch.bfloat16)
        mla_decode_fwd(
            q,
            cache,
            out,
            qptr,
            ptr,
            indices,
            last,
            1,
            sm_scale=192**-0.5,
            num_kv_splits=16,
            q_scale=scale,
            kv_scale=scale,
        )
        return out[:, :96].float()

    actual = attention()
    # The clean-native reference removes only the potential padding poison;
    # it never launches a draft writer or changes logical Q/KV or CSR data.
    cache[0].zero_()
    clean = attention()
    assert clean.abs().max() > 1e-3
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, clean, rtol=0, atol=0)
