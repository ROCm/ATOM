# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Qwen3.8-Flash-Next sequence-parallel HC streams, on CPU with simulated TP ranks."""

import threading
from concurrent.futures import ThreadPoolExecutor
from types import MethodType, SimpleNamespace

import pytest
import torch

pytest.importorskip("triton")
pytest.importorskip("aiter.ops.enum", exc_type=ImportError)

from atom.models import qwen4_exp, qwen4_exp_mtp


class _Collectives:
    """In-process TP group: one thread per rank, collectives meet at a barrier."""

    def __init__(self, world: int) -> None:
        self.world = world
        self.barrier = threading.Barrier(world, timeout=30)
        self.slots = [None] * world

    def _exchange(self, rank: int, x: torch.Tensor) -> list[torch.Tensor]:
        self.slots[rank] = x
        self.barrier.wait()
        inputs = list(self.slots)
        self.barrier.wait()
        return inputs

    def group(self, rank: int):
        def all_gather(x, use_custom=False, dim=0):
            assert dim == 0
            return torch.cat(self._exchange(rank, x))

        def reduce_scatter_tensor(x, use_custom=False, dim=0):
            assert dim == 0 and x.shape[0] % self.world == 0
            total = torch.stack(self._exchange(rank, x)).sum(0)
            return total.chunk(self.world)[rank].clone()

        return SimpleNamespace(
            all_gather=all_gather, reduce_scatter_tensor=reduce_scatter_tensor
        )


@pytest.fixture
def ranks(monkeypatch):
    """`ranks(world, fn)` runs `fn(rank)` on `world` simulated TP ranks; outside
    `ranks` the TP world size is 1."""
    local = threading.local()
    monkeypatch.setattr(qwen4_exp, "get_tp_group", lambda: local.group)
    monkeypatch.setattr(qwen4_exp, "get_tensor_model_parallel_rank", lambda: local.rank)
    monkeypatch.setattr(
        qwen4_exp,
        "get_tensor_model_parallel_world_size",
        lambda: getattr(local, "world", 1),
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    def run(world, fn):
        collectives = _Collectives(world)

        def on_rank(rank):
            local.rank, local.world = rank, world
            local.group = collectives.group(rank)
            return fn(rank)

        with ThreadPoolExecutor(world) as pool:
            return list(pool.map(on_rank, range(world)))

    return run


@pytest.mark.parametrize("world,tokens", [(2, 8), (4, 8), (4, 10), (4, 1027)])
def test_shard_gather_round_trip_drops_padding(ranks, world, tokens):
    full = torch.randn(tokens, 6)

    def on_rank(rank):
        sp = qwen4_exp._SequenceParallel(tokens, world, rank)
        local = sp.shard(full)
        assert local.shape[0] == -(-tokens // world)
        start = rank * local.shape[0]
        real = max(0, min(local.shape[0], tokens - start))
        assert torch.equal(local[:real], full[start : start + real])
        if real < local.shape[0]:
            # Only the local rows are copied, not the whole padded tensor.
            nbytes = local.untyped_storage().nbytes()
            assert nbytes == local.numel() * local.element_size()
        assert not local[real:].any(), "padded rows must be zero"
        return sp.gather(local)

    for gathered in ranks(world, on_rank):
        assert torch.equal(gathered, full)


@pytest.mark.parametrize("world,tokens", [(2, 8), (4, 10), (4, 1027)])
def test_scatter_sums_partials_into_local_rows(ranks, world, tokens):
    partials = torch.randn(world, tokens, 5, dtype=torch.float64)
    total = partials.sum(0)

    def on_rank(rank):
        sp = qwen4_exp._SequenceParallel(tokens, world, rank)
        summed = sp.scatter(partials[rank])
        start = rank * sp.chunk
        real = max(0, min(sp.chunk, tokens - start))
        torch.testing.assert_close(summed[:real], total[start : start + real])
        assert not summed[real:].any()
        return sp.gather(summed)

    for gathered in ranks(world, on_rank):
        torch.testing.assert_close(gathered, total)


class _Layer:
    """Per-token stand-in for a fused decoder layer; the PLE layer needs every
    token of the request (its causal convolution is a prefix sum here)."""

    def __init__(self, idx: int, ple: bool, calls: list) -> None:
        self.idx, self.ple, self.calls = idx, (object() if ple else None), calls

    def __call__(self, positions, hidden, input_ids, pending, sp=None):
        self.calls.append((self.idx, hidden.shape[0], sp is not None))
        if pending is not None:
            hidden = hidden + pending[0] * pending[1]
        if self.ple is not None:
            assert sp is None and hidden.shape[0] == input_ids.shape[0]
            hidden = hidden + input_ids.double().cumsum(0)[:, None]
        hidden = torch.tanh(hidden * (self.idx + 1.5))
        return hidden, (hidden.sum(1, keepdim=True), hidden[:, :1])


def _model(num_layers, ple_layer, return_hc_state, embed, hc=4):
    calls = []
    layers = [_Layer(i, i == ple_layer, calls) for i in range(num_layers)]
    hidden = embed.shape[1]

    def apply_pending(h, pending):
        return h + pending[0] * pending[1]

    model = SimpleNamespace(
        config=SimpleNamespace(hidden_size=hidden),
        hc_count=hc,
        return_hc_state=return_hc_state,
        get_input_embeddings=lambda ids: embed[ids],
        start_layer=0,
        end_layer=num_layers,
        layers=layers,
        # As `Qwen4ExpModel.__init__`: the layer after the last PLE layer.
        sp_start_layer=ple_layer + 1,
        hyper_connection_mixer=SimpleNamespace(
            apply_pending=apply_pending,
            mix_fused=lambda h, p: (apply_pending(h, p)[:, :hidden] * 0.5, None),
        ),
    )
    model._sequence_parallel = MethodType(
        qwen4_exp.Qwen4ExpModel._sequence_parallel, model
    )
    return model, calls


@pytest.mark.parametrize("return_hc_state", [False, True])
@pytest.mark.parametrize(
    "world,tokens,ple_layer",
    # Above HC_QUANT_ROWS_MAX (1024), the smallest forwards that shard.
    [
        (2, 1026, 1),
        (4, 1027, 1),
        # PLE in the last layer: nothing is left to shard, SP must stay off.
        (4, 1032, 3),
    ],
)
def test_model_forward_matches_unsharded(
    ranks, monkeypatch, world, tokens, ple_layer, return_hc_state
):
    monkeypatch.setenv("ATOM_QWEN4_EXP_SP_MIN_TOKENS", "4")
    num_layers = 4
    torch.manual_seed(0)
    input_ids = torch.randint(0, 64, (tokens,))
    positions = torch.arange(tokens)
    embed = torch.randn(64, 3, dtype=torch.float64)

    reference, _ = _model(num_layers, ple_layer, return_hc_state, embed)
    expected = qwen4_exp.Qwen4ExpModel.forward(reference, input_ids, positions)

    def on_rank(rank):
        model, calls = _model(num_layers, ple_layer, return_hc_state, embed)
        out = qwen4_exp.Qwen4ExpModel.forward(model, input_ids, positions)
        return out, calls

    chunk = -(-tokens // world)
    for out, calls in ranks(world, on_rank):
        torch.testing.assert_close(out, expected)
        for idx, rows, sharded in calls:
            assert sharded == (idx > ple_layer)
            assert rows == (chunk if sharded else tokens)


@pytest.mark.parametrize(
    "env,tokens,target_shards,drafter_shards",
    [
        # The target's fused-HC-quant boundary: only the drafter shards.
        ("1024", 1024, False, True),
        ("1024", 1025, True, True),
        # A lower threshold cannot take the target below the boundary.
        ("512", 1024, False, True),
        ("512", 700, False, True),
        ("512", 511, False, False),
        # A higher one applies to both.
        ("2048", 1025, False, False),
        ("2048", 2048, True, True),
        ("0", 4096, False, False),  # disabled
    ],
)
def test_sp_gates(ranks, monkeypatch, env, tokens, target_shards, drafter_shards):
    """The target model shards only above HC_QUANT_ROWS_MAX (its attention HC
    fuses the FP8 quant up to there); the drafter's unfused HC never
    quantizes, so it follows the env threshold alone."""
    monkeypatch.setenv("ATOM_QWEN4_EXP_SP_MIN_TOKENS", env)
    assert qwen4_exp.HC_QUANT_ROWS_MAX == 1024
    target = SimpleNamespace(sp_start_layer=2, end_layer=48)
    target_gate = MethodType(qwen4_exp.Qwen4ExpModel._sequence_parallel, target)
    drafter_gate = qwen4_exp._SequenceParallel.maybe
    results = ranks(2, lambda rank: (target_gate(tokens), drafter_gate(tokens)))
    expected = (target_shards, drafter_shards)
    assert [(t is not None, d is not None) for t, d in results] == [expected] * 2
    # TP=1 never shards.
    assert target_gate(tokens) is None and drafter_gate(tokens) is None


class _DraftLayer:
    """The drafter's QSA layer around stand-in sub-layers; `__call__` runs the
    real `Qwen4ExpDecoderLayer._forward_unfused`. Attention mixes tokens (a
    causal prefix sum) and MoE is per-token; under TP each returns this
    rank's share of the sum, as the real ones do before the all-reduce or
    reduce-scatter."""

    layer_type = "full_attention"
    ple = None

    def __init__(self, weights, rank, world, tokens, seen):
        self.tp_size = world
        share = (rank + 1) / (world * (world + 1) / 2)
        hidden = weights["mlp"].shape[0]
        hc = weights["mix"].shape[0]

        def mix(streams):
            gated = streams.view(-1, hc, hidden) * weights["mix"][None, :, None]
            return gated.sum(1), streams

        def combine(out, residual):
            return residual + (
                out[:, None, :] * weights["inject"][None, :, None]
            ).flatten(1)

        def attention(positions, mixed, rows=None):
            # Token mixing: every rank must see all rows.
            seen.append(("attn", mixed.shape[0], rows is not None))
            assert mixed.shape[0] == tokens
            return (
                share
                * torch.cumsum(mixed, 0)
                / torch.arange(1, tokens + 1, dtype=mixed.dtype)[:, None]
            )

        def mlp(mixed, reduce_results=True):
            seen.append(("mlp", mixed.shape[0], not reduce_results))
            assert mixed.shape[0] == tokens
            return share * torch.tanh(mixed @ weights["mlp"])

        self.attn_hyper_connection = SimpleNamespace(mix=mix, combine=combine)
        self.mlp_hyper_connection = SimpleNamespace(mix=mix, combine=combine)
        self.self_attn = attention
        self.mlp = mlp

    def __call__(self, positions, hidden_states, input_ids, sp=None):
        return qwen4_exp.Qwen4ExpDecoderLayer._forward_unfused(
            self, positions, hidden_states, input_ids, sp
        )


def _drafter(weights, rank, world, tokens, seen):
    def per_token_norm(x):
        # RMS norm with eps, so padded (zero) rows stay zero as in the model.
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

    return SimpleNamespace(
        hidden_size=weights["mlp"].shape[0],
        hc_count=weights["mix"].shape[0],
        embed_tokens=lambda ids: weights["embed"][ids],
        pre_fc_norm_embedding=per_token_norm,
        pre_fc_norm_hidden=per_token_norm,
        fc_embedding=lambda x: x @ weights["fc_embedding"],
        fc_hidden=lambda x: x @ weights["fc_hidden"],
        layers=[_DraftLayer(weights, rank, world, tokens, seen)],
    )


@pytest.mark.parametrize("world,tokens", [(2, 1024), (2, 1027), (4, 1026), (4, 1027)])
def test_drafter_forward_matches_unsharded(ranks, monkeypatch, world, tokens):
    """The drafter's prefill extend shards the embeddings and the target HC
    state, runs the pre-fc norms, fc projections and its unfused HC on the
    shard, and gathers the streams at the end: the result must equal the
    replicated forward, from the threshold itself (1024 tokens) and for
    uneven splits."""
    monkeypatch.setenv("ATOM_QWEN4_EXP_SP_MIN_TOKENS", "1024")
    hidden, hc = 3, 4
    gen = torch.Generator().manual_seed(5)

    def rand(*shape):
        return torch.randn(*shape, generator=gen, dtype=torch.float64)

    weights = {
        "embed": rand(64, hidden),
        "fc_embedding": rand(hidden, hidden),
        "fc_hidden": rand(hidden, hidden),
        "mix": rand(hc),
        "inject": rand(hc),
        "mlp": rand(hidden, hidden),
    }
    input_ids = torch.randint(0, 64, (tokens,), generator=gen)
    positions = torch.arange(tokens)
    target_state = rand(tokens, hc, hidden)
    forward = qwen4_exp_mtp.Qwen4ExpMultiTokenPredictor.forward

    reference_seen = []
    reference = _drafter(weights, 0, 1, tokens, reference_seen)
    expected = forward(reference, input_ids, positions, target_state)
    assert all(not sharded for _, _, sharded in reference_seen)

    def on_rank(rank):
        seen = []
        model = _drafter(weights, rank, world, tokens, seen)
        return forward(model, input_ids, positions, target_state), seen

    for out, seen in ranks(world, on_rank):
        assert out.shape == (tokens, hc, hidden)
        torch.testing.assert_close(out, expected)
        # Both sub-layers ran on all rows and took the sharded path.
        assert seen == [("attn", tokens, True), ("mlp", tokens, True)]


def _indexer_layer(qsa_attention, device, pages, block, ratio):
    """A QSA layer with the real indexer norm / RoPE / caches; the packed
    projection output is supplied directly as `hidden`."""
    from atom.model_ops.layernorm import GemmaRMSNorm

    config = SimpleNamespace(
        head_dim=256,
        max_position_embeddings=8192,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000000.0,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
    )
    indexer = qsa_attention.Qwen4ExpIndexer.__new__(qsa_attention.Qwen4ExpIndexer)
    torch.nn.Module.__init__(indexer)
    indexer.index_n_heads, indexer.index_kv_heads, indexer.index_head_dim = 4, 1, 128
    indexer.token_topk, indexer.compress_ratio = 2048, ratio
    indexer.rotary_emb = qsa_attention.build_qwen4_exp_rope(config, 128).to(device)
    indexer.q_layernorm = GemmaRMSNorm(128).to(device=device, dtype=torch.bfloat16)
    indexer.k_layernorm = GemmaRMSNorm(128).to(device=device, dtype=torch.bfloat16)
    indexer.index_qk_proj = lambda hidden, otype: hidden
    return SimpleNamespace(
        indexer=indexer,
        raw_key_cache=torch.zeros(
            pages, block, 1, 128, device=device, dtype=torch.bfloat16
        ),
        compressed_key_cache=torch.zeros(
            pages, block // ratio, 1, 128, device=device, dtype=torch.bfloat16
        ),
        compressed_key_scale=None,
        rope_position_cache=None,
    )


def _clone_caches(layer):
    return SimpleNamespace(
        **{
            **vars(layer),
            "raw_key_cache": layer.raw_key_cache.clone(),
            "compressed_key_cache": layer.compressed_key_cache.clone(),
        }
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")
@pytest.mark.parametrize("is_prefill", [False, True], ids=["triton", "flydsl"])
@pytest.mark.parametrize("world", [2, 4, 8])
@torch.inference_mode()
def test_select_tokens_row_sharded_matches_replicated(
    ranks, monkeypatch, world, is_prefill
):
    """The real `Qwen4ExpAttention._select_tokens(rows=...)`: each rank scores
    only its own query rows of a prefixed + fresh 1500-row prefill batch and
    all-gathers the groups; every rank must end with the replicated
    selection and the same index caches. `is_prefill` turns on the FlyDSL
    scorer serving prefill uses (gfx942)."""
    from atom.model_ops.qwen4_exp import qsa_attention
    from atom.model_ops.qwen4_exp.ops.qsa import qsa_compressed_slots

    if is_prefill:
        pytest.importorskip("flydsl")
        from aiter.jit.utils.chip_info import get_gfx_runtime

        if get_gfx_runtime() != "gfx942":
            pytest.skip("the FlyDSL index scorer is built for gfx942")
    monkeypatch.setattr(
        qsa_attention,
        "get_forward_context",
        lambda: SimpleNamespace(
            context=SimpleNamespace(is_draft=False, is_prefill=is_prefill)
        ),
    )
    device, block, ratio = "cuda", 64, 4
    gen = torch.Generator(device=device).manual_seed(17)
    seq_lens = (4200, 300)
    pages_per_req = [-(-s // block) for s in seq_lens]
    width = max(pages_per_req)
    pool = torch.randperm(sum(pages_per_req) + width, device=device, generator=gen)
    tables = pool[: 2 * width].view(2, width).to(torch.int32)
    base = _indexer_layer(qsa_attention, device, pool.numel(), block, ratio)

    def batch(chunks):
        """chunks: (request, first position, count) -> hidden, positions, qsa."""
        logical = torch.cat(
            [torch.arange(s, s + n, device=device) for _, s, n in chunks]
        )
        requests = torch.cat(
            [
                torch.full((n,), r, dtype=torch.int32, device=device)
                for r, _, n in chunks
            ]
        )
        slots = tables[requests.long(), logical // block].long() * block + (
            logical % block
        )
        compressed = torch.empty_like(slots)
        qsa_compressed_slots(slots, logical, ratio, compressed)
        lengths = torch.zeros(2, dtype=torch.int32, device=device)
        for r, s, n in chunks:
            lengths[r] = s + n
        qsa = SimpleNamespace(
            block_tables=tables,
            token_to_req=requests,
            logical_positions=logical,
            slot_mapping=slots,
            compressed_slot_mapping=compressed,
            seq_lens=lengths,
            max_seq_len=int(lengths.max()),
        )
        hidden = torch.randn(
            logical.numel(), 640, device=device, dtype=torch.bfloat16, generator=gen
        )
        return hidden, logical[None].expand(3, -1), qsa

    select = qsa_attention.Qwen4ExpAttention._select_tokens
    # Prefix of request 0, replicated, fills the caches.
    select(base, *batch([(0, 0, 3000)]))
    # Prefixed chunk of request 0 plus a fresh request 1: 1500 rows.
    hidden, positions, qsa = batch([(0, 3000, 1200), (1, 0, 300)])
    reference = _clone_caches(base)
    expected = select(reference, hidden, positions, qsa)

    def on_rank(rank):
        layer = _clone_caches(base)
        rows = qwen4_exp._SequenceParallel(hidden.shape[0], world, rank)
        selected = select(layer, hidden, positions, qsa, rows=rows)
        torch.cuda.synchronize()
        return selected, layer

    for selected, layer in ranks(world, on_rank):
        assert torch.equal(selected, expected)
        for name in ("raw_key_cache", "compressed_key_cache"):
            assert torch.equal(getattr(layer, name), getattr(reference, name))
