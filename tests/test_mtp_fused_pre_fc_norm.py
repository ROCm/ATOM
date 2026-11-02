# SPDX-License-Identifier: MIT
"""ATOM_MTP_FUSED_EMBED_PRE_FC_NORM: the Qwen3.5 MTP draft's embedding gather,
both pre-fc Gemma norms and the concat in one Triton launch, against what the
default path computes (F.embedding, aiter's no-quant Gemma norm twice,
torch.cat) -- bit for bit."""

import pytest
import torch

pre_fc = pytest.importorskip(
    "atom.model_ops.mtp_pre_fc_norm",
    reason="needs triton and the aiter torch guard",
    exc_type=ImportError,
)


def _rows(n, h=2048, device="cpu", dtype=torch.bfloat16):
    return torch.zeros(n, h, dtype=dtype, device=device)


def test_the_shape_gate_takes_only_aiters_2048_wide_bf16_rows():
    ids = torch.zeros(4, dtype=torch.int32)
    emb = _rows(16)
    ok = pre_fc.mtp_embed_pre_fc_norms_supported
    assert ok(ids, emb, _rows(4))
    assert not ok(ids, emb, _rows(3)), "one id a row"
    assert not ok(ids, _rows(16, 4096), _rows(4, 4096)), "another width"
    assert not ok(ids, emb, _rows(4, dtype=torch.float16)), "another dtype"
    assert not ok(ids, emb, _rows(4, 4096)[:, ::2]), "strided rows"
    assert not ok(ids.view(2, 2), emb, _rows(4)), "flat ids"


@pytest.fixture(scope="module")
def aiter_norm():
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    mod = pytest.importorskip("aiter.ops.fused_qk_rmsnorm_group_quant")
    return mod.fused_qk_rmsnorm_group_quant


def _run(backend, ids, emb, hidden, we, wh, eps):
    """One implementation of the launch, forced (`mtp_embed_pre_fc_norms` picks
    aiter's gather where the installed aiter has it, else Triton)."""
    if backend == "aiter" and not pre_fc.aiter_gathers_rows():
        pytest.skip("this aiter's norm takes no q_row_index")
    out = torch.empty(
        ids.shape[0], 2 * hidden.shape[1], dtype=hidden.dtype, device="cuda"
    )
    launch = pre_fc._launch_aiter if backend == "aiter" else pre_fc._launch_triton
    launch(ids, emb, hidden, we, eps, wh, eps, out)
    return out


def _reference(norm, ids, emb, hidden, we, wh, eps):
    e = torch.nn.functional.embedding(ids, emb)
    halves = []
    for x, w in ((e, we), (hidden, wh)):
        out = torch.empty_like(x)
        norm(q=x, q_weight=w, q_epsilon=eps, q_out_unquantized=out, gemma_norm=True)
        halves.append(out)
    return torch.cat(halves, dim=-1)


@pytest.mark.parametrize("backend", ["triton", "aiter"])
@pytest.mark.parametrize("rows", [1, 8, 33, 512, 1025, 4099])
def test_bitwise_equal_to_the_gather_two_norms_and_concat(aiter_norm, rows, backend):
    g = torch.Generator(device="cuda").manual_seed(rows)
    vocab, h = 4096, 2048
    emb = (torch.randn(vocab, h, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    emb[::7] *= 40  # rows of very different scale
    we = (torch.randn(h, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    wh = (torch.randn(h, device="cuda", generator=g) * 0.5).to(torch.bfloat16)
    ids = torch.randint(
        0, vocab, (rows,), device="cuda", generator=g, dtype=torch.int32
    )
    hidden = (
        torch.randn(rows, h, device="cuda", generator=g)
        * torch.rand(rows, 1, device="cuda", generator=g)
        * 4
    ).to(torch.bfloat16)
    hidden[:, ::97] *= 30  # outlier channels
    ref = _reference(aiter_norm, ids, emb, hidden, we, wh, 1e-6)
    out = _run(backend, ids, emb, hidden, we, wh, 1e-6)
    assert out.shape == (rows, 2 * h) and out.dtype == torch.bfloat16
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))


@pytest.mark.parametrize("backend", ["triton", "aiter"])
def test_an_id_outside_the_table_embeds_as_zeros(aiter_norm, backend):
    """The masked lookup's rule (`ReplicatedEmbedding`): a transient -1
    placeholder reads no memory and normalizes to zeros."""
    vocab, h = 64, 2048
    emb = torch.randn(vocab, h, device="cuda").to(torch.bfloat16)
    w = torch.zeros(h, device="cuda", dtype=torch.bfloat16)
    ids = torch.tensor([-1, 3, vocab], device="cuda", dtype=torch.int32)
    hidden = torch.randn(3, h, device="cuda").to(torch.bfloat16)
    out = _run(backend, ids, emb, hidden, w, w, 1e-6)
    assert out[0, :h].abs().max().item() == 0
    assert out[2, :h].abs().max().item() == 0
    ref = _reference(aiter_norm, ids[1:2], emb, hidden[1:2], w, w, 1e-6)
    assert torch.equal(out[1:2].view(torch.int16), ref.view(torch.int16))


def test_records_into_a_graph_and_replays_new_inputs(aiter_norm):
    vocab, h, rows = 512, 2048, 128
    emb = torch.randn(vocab, h, device="cuda").to(torch.bfloat16)
    we = torch.randn(h, device="cuda").to(torch.bfloat16)
    wh = torch.randn(h, device="cuda").to(torch.bfloat16)
    ids = torch.zeros(rows, device="cuda", dtype=torch.int32)
    hidden = torch.zeros(rows, h, device="cuda", dtype=torch.bfloat16)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        pre_fc.mtp_embed_pre_fc_norms(ids, emb, hidden, we, 1e-6, wh, 1e-6)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=s):
            out = pre_fc.mtp_embed_pre_fc_norms(ids, emb, hidden, we, 1e-6, wh, 1e-6)
    ids.copy_(torch.randint(0, vocab, (rows,), device="cuda", dtype=torch.int32))
    hidden.copy_(torch.randn(rows, h, device="cuda").to(torch.bfloat16))
    graph.replay()
    torch.cuda.synchronize()
    ref = _reference(aiter_norm, ids, emb, hidden, we, wh, 1e-6)
    assert torch.equal(out.view(torch.int16), ref.view(torch.int16))
