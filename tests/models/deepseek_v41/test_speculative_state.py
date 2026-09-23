# SPDX-License-Identifier: MIT
"""Every accepted prefix must recover history, window rows and compressor state."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("aiter", reason="the paged cache reaches AITER")

from atom.model_ops.attentions.deepseek_v41.cache import PagedAttentionCache
from atom.model_ops.attentions.deepseek_v41.checkpoints import StateCopies
from atom.model_ops.attentions.deepseek_v41.metadata import RequestSpan
from atom.model_ops.attentions.pool_layout.v41_pool_geometry import V41PoolGeometry


def encoded_rows(positions, layer, geometry, device):
    width = geometry.window_row_bytes if geometry.packed else geometry.head_dim
    values = (
        torch.as_tensor(positions, device=device)[:, None]
        + layer * 71
        + torch.arange(width, device=device)
    ) % 251
    return values.to(torch.uint8 if geometry.packed else torch.bfloat16)


def write_window(cache, layer, step, rows):
    if not rows.is_cuda:
        for span in step.requests:
            positions = (
                torch.arange(span.position, span.end) % cache.geometry.ring_slots
            )
            cache.state.view("window")[layer, span.slot, positions] = rows[
                span.token_slice
            ]
    elif cache.packed:
        dim = cache.geometry.head_dim
        cache.write_window(
            layer, (rows[:, :dim].contiguous(), rows[:, dim:].contiguous()), step
        )
    else:
        cache.write_window(layer, rows.unsqueeze(0), step)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("position", [127, 128, 265])
@pytest.mark.parametrize("accepted", range(1, 7))
def test_every_prefix_survives_ring_wrap_and_ragged_request_order(
    single_rank, device, packed, position, accepted
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("ROCm GPU required")
    geometry = V41PoolGeometry(
        2,
        ((0, 2), (1, 1)),
        32,
        128,
        128,
        32,
        packed=packed,
        speculative_tokens=5,
        # The only class the window-only specs below ask for.
        layer_ratios=(0,),
    )
    cache = PagedAttentionCache(geometry, 40, 5, device)
    spans = (
        RequestSpan(11, position, 0, 6, 4, tuple(range(20))),
        RequestSpan(22, position - 4, 6, 3, 1, tuple(range(20, 40))),
    )
    history = torch.tensor([41, -1, 43], device=device)
    for span in spans:
        cache.cursor[span.slot, 0] = span.position
        cache.cursor[span.slot, 1:] = history
        kept = torch.arange(
            max(0, span.position - geometry.ring_slots), span.position, device=device
        )
        for layer in range(2):
            cache.state.view("window")[layer, span.slot, kept % geometry.ring_slots] = (
                encoded_rows(kept, layer, geometry, device)
            )
    untouched = cache.state_bytes[0].clone()
    old_cursors = cache.cursor.clone()
    step = cache.begin_step(spans, tentative=True)
    cache.prepare_state(step)
    expected_histories = []
    accepted_lengths = (accepted, min(accepted, 3))
    for i, span in enumerate(spans):
        ids = [7, -1, 19, 20, 21, 22][: span.length]
        cache.pending.stage_history(span, ids)
        expected_histories.append((history.tolist() + ids[: accepted_lengths[i]])[-3:])
    for layer in range(2):
        write_window(
            cache, layer, step, encoded_rows(step.positions, layer, geometry, device)
        )
    assert torch.equal(cache.cursor, old_cursors)
    with pytest.raises(RuntimeError, match="Commit the accepted prefix"):
        cache.begin_step(spans)
    copies = StateCopies.__new__(StateCopies)
    copies.cache = cache
    with pytest.raises(RuntimeError, match="Commit the accepted prefix"):
        copies.entry(4)
    cache.commit_tentative(step, torch.tensor(accepted_lengths, device=device))
    assert torch.equal(cache.state_bytes[0], untouched)
    for i, (span, count) in enumerate(zip(spans, accepted_lengths)):
        end = span.position + count
        assert cache.cursor[span.slot].tolist() == [end] + expected_histories[i]
        positions = torch.arange(max(0, end - geometry.window_size), end, device=device)
        for layer in range(2):
            actual = cache.state.view("window")[
                layer, span.slot, positions % geometry.ring_slots
            ]
            assert torch.equal(actual, encoded_rows(positions, layer, geometry, device))
    # Reorder requests after acceptance. Addressing must use each noncontiguous
    # STATE slot, and logical visibility must stay at 128, not the 133-row ring.
    next_spans = tuple(
        RequestSpan(
            span.request_id, span.position + count, i * 2, 2, span.slot, span.block_ids
        )
        for i, (span, count) in enumerate(reversed(list(zip(spans, accepted_lengths))))
    )
    next_step = cache.begin_step(next_spans)
    cache.prepare_state(next_step)
    if device == "cuda":
        for layer in range(2):
            spec = SimpleNamespace(layer_id=layer, ratio=0, kv_owner=None)
            prefix, ptr, extend, eptr = cache.attention_indices(spec, next_step)
            prefix, ptr, extend, eptr = [x.cpu() for x in (prefix, ptr, extend, eptr)]
            for span in next_spans:
                for offset in range(2):
                    t = span.offset + offset
                    first = max(0, span.position + offset - geometry.window_size + 1)
                    window = geometry.window(layer, cache.num_pages)
                    expected = [
                        window.index(span.slot, p) for p in range(first, span.position)
                    ]
                    if packed:
                        expected = [(address << 1) | 1 for address in expected]
                    assert prefix[ptr[t] : ptr[t + 1]].tolist() == expected
                    assert extend[eptr[t] : eptr[t + 1]].tolist() == list(
                        range(span.offset, t + 1)
                    )


def test_tentative_state_refuses_missing_prefixes_and_a_stale_step():
    """The two things `commit` still refuses.

    An out-of-range accepted length is no longer one of them: that bound cost
    four launches a step and is commented out in `commit`, so a length past
    the staged span now writes an earlier round's cursor in silence. The
    check is still spelled out there for whoever suspects it.
    """
    geometry = V41PoolGeometry(1, ((0, 2),), 32, 128, 128, 32, speculative_tokens=5)
    cache = PagedAttentionCache(geometry, 1, 1, "cpu")
    cache.cursor[0, 0] = 3
    span = RequestSpan(1, 3, 0, 6, 0, (0,))
    stale = cache.begin_step((span,))
    step = cache.begin_step((span,), tentative=True)
    cache.prepare_state(step)
    with pytest.raises(RuntimeError, match="missing"):
        cache.commit_tentative(step, [1])
    cache.pending.stage_history(span, [1, 2, 3, 4, 5, 6])
    # The accepted prefix belongs to one forward. Another step's lengths would
    # index this one's staged cursors and commit a row nobody verified.
    with pytest.raises(RuntimeError, match="not prepared for this step"):
        cache.commit_tentative(stale, [1])
    cache.commit_tentative(step, [1])
    assert cache.cursor[0, 0] == 4


def test_commit_moves_the_scheduled_cursors_and_no_padding_requests():
    """A padded verify step is wider than its batch.

    The padding rows of `slots` carry 0, which is a real STATE slot and in
    general somebody else's, so anything that pairs a slot with per-request
    host data has to stop at `scheduled_bs`. The width here is deliberately
    not the batch: at `running_bs == scheduled_bs` a slice by either is the
    same slice and proves nothing.
    """
    geometry = V41PoolGeometry(1, ((0, 2),), 32, 128, 128, 32, speculative_tokens=5)
    cache = PagedAttentionCache(geometry, 1, 4, "cpu")
    for slot in range(4):
        cache.cursor[slot, 0] = 3
    spans = (RequestSpan(1, 3, 0, 2, 3, (0,)), RequestSpan(2, 3, 2, 2, 1, (0,)))
    step = cache.begin_step(spans, tentative=True, running_bs=3, running_tokens=6)
    assert step.scheduled_bs == 2 and step.slots.tolist() == [3, 1, 0]
    cache.prepare_state(step)
    for span in spans:
        cache.pending.stage_history(span, [7, 9])
    cache.commit_tentative(step, [2, 1])
    assert cache.cursor[3, 0] == 5 and cache.cursor[1, 0] == 4
    # The slot the padding named, untouched: it belongs to nobody in this step.
    assert cache.cursor[0, 0] == 3 and cache.cursor[2, 0] == 3


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("accepted", [1, 3, 5])
def test_a_rejected_round_leaves_the_next_one_as_if_it_never_drafted(
    single_rank, accepted
):
    """The claim `compress_ring_slots` makes: slack instead of a rollback.

    Nothing rewinds the compressor after a rejection -- the discarded rows stay
    in the ring, and the next round is supposed to be unable to see them
    because they sit past the `K_pool` window it reads. So run the same accepted
    prefix twice, once behind six drafted tokens and once alone, and require the
    next round's latent to be the same bits. An arm that only drafted what it
    kept is the only oracle here: there is no closed form for what the ring
    should hold.
    """
    from atom.model_ops.deepseek_v41.compressor import Compressor
    from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding

    geometry = V41PoolGeometry(1, ((0, 2),), 32, 128, 128, 32, speculative_tokens=5)
    drafted, position, blocks = 6, 40, tuple(range(8))
    torch.manual_seed(409)
    with torch.device("cuda"):
        compressor = Compressor(64, 128, 2, 1e-6)
        rope = RotaryEmbedding(64, 4096, base=10000)
        for parameter in compressor.parameters():
            parameter.data.normal_(0, 0.05)
        hidden = torch.randn(1, drafted, 64, dtype=torch.bfloat16)
        following = torch.randn(1, 3, 64, dtype=torch.bfloat16)

    def round_one(length):
        cache = PagedAttentionCache(geometry, 20, 1, "cuda")
        cache.cursor[0, 0] = position
        span = RequestSpan(7, position, 0, length, 0, blocks)
        step = cache.begin_step((span,), tentative=True)
        cache.prepare_state(step)
        cache.compress(
            0, compressor, *compressor.project(hidden[:, :length]), step, rope
        )
        cache.pending.stage_history(span, [-1] * length)
        cache.commit_tentative(step, [accepted])
        return cache

    def round_two(cache):
        span = RequestSpan(7, position + accepted, 0, 3, 0, blocks)
        step = cache.begin_step((span,))
        cache.prepare_state(step)
        latent = cache.compress(
            0, compressor, *compressor.project(following), step, rope
        )
        cache.advance_cursor(step, [[-1, -1, -1]])
        return latent, step.plans[2].key_rope_positions_gpu

    drafting, honest = round_one(drafted), round_one(accepted)
    # Armed: the rings really do differ, so the equality below is a statement
    # about what the next round reads, not about two identical caches.
    assert not torch.equal(*[c.compress_state(0)[0] for c in (drafting, honest)])
    expected, expected_positions = round_two(honest)
    actual, actual_positions = round_two(drafting)
    # Positive control: an accepted prefix that crosses no boundary would make
    # both arms `None` and the comparison vacuous.
    assert expected is not None and expected.shape[1]
    assert torch.equal(actual_positions, expected_positions)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("packed", [False, True])
def test_block_context_read_decodes_only_the_selected_request_windows(packed):
    from atom.model_ops.blockscale import quantize_fp8

    geometry = V41PoolGeometry(
        2,
        ((0, 2),),
        32,
        128,
        512,
        32,
        packed=packed,
        speculative_tokens=5,
        layer_ratios=(0,),
    )
    cache = PagedAttentionCache(geometry, 24, 4, "cuda")
    spans = (
        RequestSpan(1, 0, 0, 145, 3, tuple(range(10))),
        RequestSpan(2, 0, 145, 142, 1, tuple(range(10, 20))),
    )
    step = cache.begin_step(spans)
    cache.prepare_state(step)
    torch.manual_seed(95)
    values = torch.randn(1, 287, 512, device="cuda", dtype=torch.bfloat16)
    expected_rows = quantize_fp8(values, dequantize=True)[0]
    cache.write_window(
        1, quantize_fp8(values) if packed else expected_rows.unsqueeze(0), step
    )
    cache.advance_cursor(step, [[1, 2, 3], [4, 5, 6]])
    actual = cache.read_window(1, torch.tensor([1, 3, 1], device="cuda"))
    expected = torch.empty_like(actual)
    for i, span in enumerate((spans[1], spans[0], spans[1])):
        for p in range(span.end - geometry.ring_slots, span.end):
            expected[i, p % geometry.ring_slots] = expected_rows[span.offset + p]
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("packed", [False, True])
def test_verify_decode_kernel_is_causal_after_writing_the_whole_block(packed):
    from atom.model_ops.attentions.deepseek_v41.packed_attention import packed_decode
    from atom.model_ops.attentions.deepseek_v41.packed_rows import pack_rows
    from atom.model_ops.blockscale import quantize_fp8
    from atom.model_ops.v4_kernels import sparse_attn_v4_paged_decode

    torch.manual_seed(863)
    geometry = V41PoolGeometry(
        1,
        ((0, 2),),
        32,
        4,
        512,
        32,
        packed=packed,
        speculative_tokens=5,
        layer_ratios=(0,),
    )
    cache = PagedAttentionCache(geometry, 1, 1, "cuda")
    raw = torch.randn(13, 512, device="cuda", dtype=torch.bfloat16)
    quantized = quantize_fp8(raw)
    keys = quantize_fp8(raw, dequantize=True)
    stored = pack_rows(*quantized) if packed else keys
    cache.state.view("window")[0, 0, :7] = stored[:7]
    cache.cursor[0, 0] = 7
    step = cache.begin_step((RequestSpan(1, 7, 0, 6, 0, (0,)),), tentative=True)
    cache.prepare_state(step)
    assert step.decode
    cache.write_window(
        0,
        tuple(v[7:].contiguous() for v in quantized) if packed else keys[None, 7:],
        step,
    )
    spec = SimpleNamespace(layer_id=0, ratio=0, kv_owner=None)
    indices, ptr, _, _ = cache.attention_indices(spec, step)
    query = torch.randn(6, 8, 512, device="cuda", dtype=torch.bfloat16)
    sink = torch.randn(8, device="cuda")
    function = packed_decode if packed else sparse_attn_v4_paged_decode
    output = function(query, cache.pool, indices, ptr, sink, 512**-0.5)
    expected = []
    for t, position in enumerate(range(7, 13)):
        visible = keys[position - 3 : position + 1].float()
        scores = query[t].float() @ visible.T * 512**-0.5
        probability = torch.cat((scores, sink[:, None]), -1).softmax(-1)[:, :-1]
        expected.append(probability @ visible)
    expected = torch.stack(expected)
    assert (output.float() - expected).norm() / expected.norm() < 0.004
    # All rows were physically written; the final future row still cannot
    # affect any preceding query's attention result.
    cache.state.view("window")[0, 0, 12 % geometry.ring_slots].zero_()
    changed = function(query, cache.pool, indices, ptr, sink, 512**-0.5)
    assert torch.equal(changed[:-1], output[:-1])
    assert not torch.equal(changed[-1], output[-1])
