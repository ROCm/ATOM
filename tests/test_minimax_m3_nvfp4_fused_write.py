# SPDX-License-Identifier: MIT
"""`SparseMHAPagedAttentionImpl.rope_cache` with the NVFP4 caches (gfx950).

aiter's fused norm / RoPE / KV-write op quantizes K, V and the index key into NVFP4 pages
itself. The pages are compared byte for byte with what the Triton oracle
(``nvfp4_write_oracle``) makes of the bf16 rows the same op writes in "auto" mode, on
garbage-filled pages, at scattered slots with a skipped one; the main q must be what the
bf16 write gives, the index q its e4m3 cast. The kernel's own numerics are aiter's
``op_tests/test_fused_qknorm_nvfp4.py``.
"""

import pytest
import torch

from atom.model_ops.minimax_m3.nvfp4_pages import INDEX_PAGE_BYTES, kv_page_bytes

aiter = pytest.importorskip("aiter")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or "gfx950" not in torch.cuda.get_device_properties(0).gcnArchName
    or not hasattr(aiter, "fused_qknorm_idxrqknorm"),
    reason="needs gfx950 and aiter.fused_qknorm_idxrqknorm",
)

from nvfp4_write_oracle import nvfp4_insert_index, nvfp4_insert_kv
from test_minimax_m3_mxfp4_write import (
    NKV,
    NQ,
    PAGES,
    inputs,
    make_impl,
    reference,
    run,
)

D, NIQ = 128, 1


def nvfp4_impl(skip_index: bool = False):
    impl = make_impl(False, "nvfp4")
    impl.kv_cache_dtype = "nvfp4"
    impl.skip_index_topk = skip_index
    # distinct global scales: a swapped K / V scale cannot pass
    impl.nvfp4_k_scale = torch.tensor([0.75], device="cuda")
    impl.nvfp4_v_scale = torch.tensor([1.5], device="cuda")
    return impl


def garbage(*shape, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda", generator=g)


@pytest.mark.parametrize("skip_index", [False, True])
def test_pages_equal_the_oracle_byte_for_byte(skip_index):
    T = 37
    qkv, position, slots = inputs(T, 5)
    rk, rv, ri, q_bf16, _ = reference(qkv, position, slots)
    # the reference's rows are per slot; the oracle takes them per token
    safe = slots.clamp(min=0)
    k_rows = rk.view(-1, NKV, D)[safe]
    v_rows = rv.view(-1, NKV, D)[safe]
    i_rows = ri.view(-1, D)[safe]

    impl = nvfp4_impl(skip_index)
    kp = garbage(PAGES, kv_page_bytes(NKV), seed=1)
    ip = garbage(PAGES, INDEX_PAGE_BYTES, seed=2)
    want_kp, want_ip = kp.clone(), ip.clone()
    q, _, _, kc, vc, ks, vs = run(impl, qkv, position, slots, kp, kp, ip)
    assert kc is kp and vc is kp and ks is None and vs is None

    nvfp4_insert_kv(
        k_rows, v_rows, want_kp, slots, impl.nvfp4_k_scale, impl.nvfp4_v_scale
    )
    if skip_index:
        assert impl._index_q is None
        assert torch.equal(ip, want_ip)  # a top-k reusing layer writes no index key
    else:
        nvfp4_insert_index(i_rows, want_ip, slots)
        assert torch.equal(ip, want_ip)
        assert impl._index_q.dtype == torch.float8_e4m3fn
    assert torch.equal(kp, want_kp)
    assert torch.equal(q.view(torch.uint8), q_bf16.view(-1, NQ, D).view(torch.uint8))
