"""Validate the vendored QSA Triton kernels at Qwen3.8-Flash-Next's actual geometry.

The AITER PR these come from was benchmarked at head_dim 128 / GQA group 5 /
8 index heads. This checkpoint is head_dim 256 / group 12 / 4 index heads, a
shape nobody upstream has exercised, so these tests are the only evidence the
portable Triton path is correct for it.
"""

import pytest
import torch

from tests.qwen3_8_flash_next.parity_harness import init_single_rank

INDEX_HEADS, INDEX_DIM = 4, 128
Q_HEADS, KV_HEADS, HEAD_DIM = 24, 2, 256
COMPRESS_RATIO = 4
PAGE_SIZE = 16
DTYPE = torch.bfloat16


@pytest.fixture(scope="module", autouse=True)
def _cuda():
    init_single_rank()


def _paged_cache(num_pages, heads, dim):
    return (torch.randn(num_pages, PAGE_SIZE, heads, dim) * 0.3).cuda().to(DTYPE)


def test_compressed_scoring_matches_torch_reference():
    from atom.model_ops.qwen3_8_flash_next.qsa_ops import qsa_paged_mqa_logits

    torch.manual_seed(0)
    num_reqs, pages_per_req, tokens = 3, 4, 9
    num_pages = num_reqs * pages_per_req
    k_cache = _paged_cache(num_pages, 1, INDEX_DIM)
    page_table = (
        torch.arange(num_pages, dtype=torch.int32).reshape(num_reqs, pages_per_req)
    ).cuda()
    q = (torch.randn(tokens, INDEX_HEADS, INDEX_DIM) * 0.3).cuda().to(DTYPE)
    token_to_request = torch.tensor(
        [0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.int32
    ).cuda()
    query_positions = torch.tensor(
        [10, 20, 33, 5, 40, 60, 0, 7, 15], dtype=torch.int32
    ).cuda()
    context_lens = torch.tensor([48, 64, 32], dtype=torch.int32).cuda()

    logits, visible = qsa_paged_mqa_logits(
        q,
        k_cache,
        page_table,
        token_to_request,
        query_positions,
        context_lens,
        COMPRESS_RATIO,
    )

    columns = pages_per_req * PAGE_SIZE
    flat_keys = k_cache.reshape(num_pages * PAGE_SIZE, INDEX_DIM).float()
    for token in range(tokens):
        req = int(token_to_request[token])
        expected_visible = max(
            0,
            min(
                (int(query_positions[token]) + 1) // COMPRESS_RATIO,
                int(context_lens[req]) // COMPRESS_RATIO,
            ),
        )
        assert int(visible[token]) == expected_visible
        for column in range(columns):
            got = logits[token, column].item()
            if column >= expected_visible:
                assert got == float("-inf")
                continue
            page = int(page_table[req, column // PAGE_SIZE])
            key = flat_keys[page * PAGE_SIZE + column % PAGE_SIZE]
            want = (torch.relu(q[token].float() @ key).sum() / (INDEX_DIM**0.5)).item()
            assert abs(got - want) < 2e-2, (token, column, got, want)


def test_sparse_gqa_matches_torch_reference():
    from atom.model_ops.qwen3_8_flash_next.qsa_ops import qsa_sparse_paged_gqa

    torch.manual_seed(1)
    num_reqs, pages_per_req, tokens, width = 2, 5, 6, 37
    num_pages = num_reqs * pages_per_req
    k_cache = _paged_cache(num_pages, KV_HEADS, HEAD_DIM)
    v_cache = _paged_cache(num_pages, KV_HEADS, HEAD_DIM)
    block_table = (
        torch.arange(num_pages, dtype=torch.int32).reshape(num_reqs, pages_per_req)
    ).cuda()
    q = (torch.randn(tokens, Q_HEADS, HEAD_DIM) * 0.3).cuda().to(DTYPE)
    token_to_request = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.int32).cuda()

    capacity = pages_per_req * PAGE_SIZE
    logical = torch.randint(0, capacity, (tokens, width), dtype=torch.int32).cuda()
    # Exercise the -1 padding path on a couple of rows.
    logical[1, width // 2 :] = -1
    logical[4, -3:] = -1

    out = qsa_sparse_paged_gqa(
        q, k_cache, v_cache, logical, block_table, token_to_request
    )

    scale = HEAD_DIM**-0.5
    group = Q_HEADS // KV_HEADS
    for token in range(tokens):
        req = int(token_to_request[token])
        ids = [int(i) for i in logical[token].tolist() if i >= 0]
        pages = [block_table[req, i // PAGE_SIZE] for i in ids]
        offsets = [i % PAGE_SIZE for i in ids]
        for head in range(Q_HEADS):
            kv_head = head // group
            keys = torch.stack(
                [k_cache[p, o, kv_head].float() for p, o in zip(pages, offsets)]
            )
            values = torch.stack(
                [v_cache[p, o, kv_head].float() for p, o in zip(pages, offsets)]
            )
            scores = (q[token, head].float() @ keys.T) * scale
            want = torch.softmax(scores, dim=-1) @ values
            got = out[token, head].float()
            assert torch.allclose(got, want, rtol=6e-2, atol=6e-2), (
                token,
                head,
                (got - want).abs().max().item(),
            )


def test_expansion_widths_and_tail():
    from atom.model_ops.qwen3_8_flash_next.qsa_ops import qsa_expand_block_indices

    token_topk, block_topk = 32, 8
    groups = torch.arange(block_topk, dtype=torch.int32).repeat(2, 1).cuda()
    query_positions = torch.tensor([40, 41], dtype=torch.int32).cuda()
    context_lens = torch.tensor([64], dtype=torch.int32).cuda()
    token_to_request = torch.zeros(2, dtype=torch.int32).cuda()

    out = qsa_expand_block_indices(
        groups,
        query_positions,
        context_lens,
        token_to_request,
        COMPRESS_RATIO,
        token_topk,
    )
    assert out.shape == (2, token_topk + COMPRESS_RATIO - 1)
    # Every emitted id is either padding or a real, causally visible position.
    for row in range(2):
        ids = out[row].tolist()
        real = [i for i in ids if i >= 0]
        assert real, "expansion produced no tokens"
        assert max(real) <= int(query_positions[row])
