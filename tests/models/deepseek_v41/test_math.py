# SPDX-License-Identifier: MIT
"""Differential module checks against unchanged pinned upstream model methods."""

import os
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from atom.model_ops.deepseek_v41.mhc import (
    SinglePassHCState,
    apply_sublayer,
    predict_mixes,
)
from atom.model_ops.deepseek_v41.moe import Expert, Router, weighted_swiglu
from atom.model_ops.engram import CompressedTokenizer, EngramConfig, NgramHashMapping
from atom.model_ops.engram_layer import EngramOp

from .reference import load_reference


@pytest.fixture
def reference():
    directory = os.environ.get("ATOM_DSV41_REFERENCE")
    if not directory:
        pytest.skip("Set ATOM_DSV41_REFERENCE for pinned model methods")
    with load_reference(directory) as module:
        yield module


@contextmanager
def bf16_default():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def test_single_pass_mhc_uses_incoming_mix_and_final_ffn_mix(reference):
    torch.manual_seed(211)
    ref = reference.Block.__new__(reference.Block)
    nn.Module.__init__(ref)
    ref.norm_eps, ref.hc_eps, ref.hc_sinkhorn_iters, ref.hc_mult = 1e-20, 1e-6, 20, 4

    class Projection(nn.Linear):
        def forward(self, x, *_):
            return super().forward(x)

    with bf16_default():
        ref.attn, ref.ffn = Projection(32, 32, bias=False), Projection(
            32, 32, bias=False
        )
        ref.attn_norm, ref.ffn_norm = reference.RMSNorm(32, 1e-20), reference.RMSNorm(
            32, 1e-20
        )
    for sublayer in ("attn", "ffn"):
        setattr(ref, f"hc_{sublayer}_fn", nn.Parameter(torch.randn(24, 128) * 0.05))
        setattr(ref, f"hc_{sublayer}_base", nn.Parameter(torch.randn(24) * 0.2))
        setattr(ref, f"hc_{sublayer}_scale", nn.Parameter(torch.randn(3) * 0.1))
    residual = torch.randn(2, 5, 4, 32, dtype=torch.bfloat16)
    incoming = torch.rand(2, 5, 4)
    expected, expected_mix = ref(residual, 0, incoming, None)
    state = SinglePassHCState(residual, incoming)
    for name in ("attn", "ffn"):
        operation, norm = getattr(ref, name), getattr(ref, name + "_norm")
        state = apply_sublayer(
            state,
            nn.Sequential(norm, operation),
            getattr(ref, f"hc_{name}_fn"),
            getattr(ref, f"hc_{name}_scale"),
            getattr(ref, f"hc_{name}_base"),
        )
    assert torch.equal(state.residual, expected)
    torch.testing.assert_close(state.pre_mix, expected_mix, rtol=1e-6, atol=1e-7)
    assert torch.equal(state.collapse(), ref.hc_pre(expected, expected_mix))
    initial = SinglePassHCState.from_embeddings(residual[:, :, 0], 4)
    assert torch.equal(initial.collapse(), residual[:, :, 0])
    assert torch.equal(initial.pre_mix[..., 0], torch.ones(2, 5))


def test_mhc_coefficient_norm_epsilon_is_not_sinkhorn_epsilon(reference):
    torch.manual_seed(2)
    x = (torch.randn(1, 2, 4, 32) * 1e-8).bfloat16()
    fn, scale, base = torch.randn(24, 128), torch.ones(3), torch.zeros(24)
    stub = SimpleNamespace(norm_eps=1e-20, hc_mult=4, hc_sinkhorn_iters=20, hc_eps=1e-6)
    expected = reference.Block.hc_mixes(stub, x, fn, scale, base)
    actual = predict_mixes(x, fn, scale, base)
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=1e-6, atol=1e-7)
    with pytest.raises(ValueError, match="FP32"):
        predict_mixes(x, fn.bfloat16(), scale, base)


def test_router_image_bias_only_changes_selection(reference):
    torch.manual_seed(22)
    args = reference.ModelArgs(
        dim=32,
        n_routed_experts=8,
        n_activated_experts=3,
        vision_n_layers=1,
        route_scale=1.5,
    )
    with bf16_default():
        expected_router = reference.Gate(0, args)
    target = Router(32, 8, 3)
    for name, parameter in target.named_parameters():
        data = torch.randn_like(parameter)
        parameter.data.copy_(data)
        getattr(expected_router, name).data.copy_(data)
    hidden = torch.randn(7, 32, dtype=torch.bfloat16)
    images = torch.tensor([False, True, True, False, False, True, False])
    actual_w, actual_i = target(hidden, images)
    expected_w, expected_i = expected_router(hidden, images)
    assert torch.equal(actual_i, expected_i)
    torch.testing.assert_close(actual_w, expected_w, rtol=1e-6, atol=1e-7)
    # Uniform offsets preserve selection and must never enter normalization.
    target.bias.data.add_(100)
    target.bias_vl.data.add_(100)
    shifted_w, shifted_i = target(hidden, images)
    assert torch.equal(actual_i, shifted_i)
    assert torch.equal(actual_w, shifted_w)


def test_weighted_swiglu_reference_order_and_asymmetric_clamp(reference):
    class FixedProjection(nn.Module):
        def __init__(self, output):
            super().__init__()
            self.output = output

        def forward(self, _):
            return self.output

    ref = reference.Expert.__new__(reference.Expert)
    nn.Module.__init__(ref)
    gate = torch.tensor([[-30.0, -10.0, -3.0, 0.0, 4.0, 30.0]], dtype=torch.bfloat16)
    up = torch.tensor([[30.0, -30.0, 4.0, 2.0, -4.0, -15.0]], dtype=torch.bfloat16)
    ref.w1, ref.w3, ref.w2, ref.swiglu_limit = (
        FixedProjection(gate),
        FixedProjection(up),
        nn.Identity(),
        10.0,
    )
    weights = torch.tensor([[0.137]])
    expected = ref(gate, weights)
    actual = weighted_swiglu(gate, up, weights)
    assert torch.equal(actual, expected)
    assert actual[0, 0] != 0  # negative gate is not clamped at -10
    expert = Expert(ref.w1, ref.w2, ref.w3)
    assert torch.equal(expert(gate, weights), expected)


def _reference_engram(reference, target):
    ref = reference.Engram.__new__(reference.Engram)
    nn.Module.__init__(ref)
    ref.dim, ref.hc_mult, ref.eps, ref.clamp_value = (
        target.hidden_size,
        target.hc_mult,
        target.norm_eps,
        1e-6,
    )
    ref.embed = nn.Identity()
    ref.wkv = deepcopy(target.wkv)
    ref.q_weight = nn.Parameter(target.q_weight.detach().clone())
    ref.k_weight = nn.Parameter(target.k_weight.detach().clone())
    return ref


def test_engram_fp32_gate_residual_and_image_mask(reference):
    torch.manual_seed(192)
    with bf16_default():
        target = EngramOp(1, hidden_size=32, engram_hidden_size=64, hc_mult=4)
    target.q_weight.data.normal_(mean=1, std=0.1)
    target.k_weight.data.normal_(mean=1, std=0.1)
    target.process_weights_after_loading()
    ref = _reference_engram(reference, target)
    hidden = torch.randn(2, 7, 4, 32, dtype=torch.bfloat16)
    embeddings = torch.randn(2, 7, 2, 32, dtype=torch.bfloat16)
    mask = torch.tensor(
        [
            [True, True, False, False, True, False, True],
            [False, True, True, True, True, True, False],
        ]
    )
    expected = ref(hidden, embeddings, mask)
    actual = target(hidden, embeddings.flatten(-2), mask)
    assert torch.equal(actual, expected)
    assert torch.equal(actual[~mask], hidden[~mask])


def test_zero_engram_dot_preserves_signed_sqrt_contract(reference):
    target = EngramOp(1, hidden_size=32, engram_hidden_size=32, hc_mult=4)
    target.wkv.weight.data.zero_()
    target.wkv.weight.data[-32:] = 1
    reference_op = _reference_engram(reference, target)
    hidden = torch.zeros(1, 1, 4, 32)
    embeddings = torch.ones(1, 1, 1, 32)
    assert torch.equal(
        target(hidden, embeddings.flatten(-2)), reference_op(hidden, embeddings)
    )


def test_real_tokenizer_history_images_and_accepted_prefix(reference):
    from transformers import AutoTokenizer

    directory = os.environ["ATOM_DSV41_REFERENCE"]
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True)
    from atom.config import get_hf_config

    config = get_hf_config(directory)
    engram_fields = {
        name: getattr(config, name)
        for name in (
            "engram_layer_ids",
            "engram_num_embeddings",
            "engram_max_ngram_size",
            "engram_vocab_size",
            "engram_n_heads",
            "engram_head_dim",
            "engram_compressed_vocab_size",
        )
    }
    args = reference.ModelArgs(
        max_batch_size=2,
        max_seq_len=32,
        engram_pad_id=config.engram_pad_token_id,
        **engram_fields,
    )
    layout = reference.EngramLayout.from_args(args)
    source = reference.NgramHashState(args, layout, tokenizer)
    config = EngramConfig(
        layer_ids=layout.layer_ids,
        num_embeddings=layout.num_embeddings,
        max_ngram_size=layout.max_ngram_size,
        vocab_size=args.engram_vocab_size,
        n_heads=layout.n_heads,
        head_dim=layout.head_dim,
        pad_token_id=args.engram_pad_id,
        compressed_vocab_size=args.engram_compressed_vocab_size,
    )
    target = NgramHashMapping(
        config,
        CompressedTokenizer(tokenizer, expected_size=config.compressed_vocab_size),
    )
    assert np.array_equal(target.tokenizer.lookup_table, source.token_map.numpy())
    tokens = np.array(
        [
            [0, 371, 9, 102, 129264, 129264, 129264, 20, 19, 72, 41, 3],
            [0, 71, 9, 102, 21, 22, 13, 129264, 129264, 129264, 41, 3],
        ],
        dtype=np.int64,
    )
    mask = tokens != 129264
    expected = source(torch.from_numpy(tokens), 0, torch.from_numpy(mask)).numpy()
    from atom.model_engine.engram_runtime import EngramPrefetcher, EngramRequest

    class RowIds:
        def gather(self, rows):
            return torch.from_numpy(rows.copy())

    prefetcher = EngramPrefetcher(
        target, {layer: RowIds() for layer in layout.layer_ids}
    )
    histories = None
    position = 0
    for count in (1, 2, 3, 1, 5):
        chunk, live = (
            tokens[:, position : position + count],
            mask[:, position : position + count],
        )
        result = target.hash_all_layers(chunk, history=histories, token_mask=live)
        for layer_index, layer in enumerate(layout.layer_ids):
            actual = target.to_row_indices(result[layer], layer)
            np.testing.assert_array_equal(
                actual, expected[:, position : position + count, layer_index]
            )
        requests = [
            EngramRequest(
                row,
                0,
                position,
                tuple(chunk[row]),
                (-1, -1, -1) if histories is None else tuple(histories[row]),
                tuple(live[row]),
            )
            for row in range(2)
        ]
        prefetcher.submit_compute(requests).result(timeout=30)
        fallback = prefetcher.compute(requests)
        for row, request in enumerate(requests):
            for layer_index, layer in enumerate(layout.layer_ids):
                official = expected[row, position : position + count, layer_index]
                np.testing.assert_array_equal(
                    prefetcher.cache.take(request, layer), official
                )
                np.testing.assert_array_equal(fallback[(request, layer)], official)
        compressed = target.compress_tokens(chunk, live)
        histories = target.advance_history(histories, compressed)
        position += count
    prefetcher.shutdown()
    # Rejected tails and image delimiters do not enter the next committed history.
    tentative = target.compress_tokens(tokens[:, :5], mask[:, :5])
    for accepted in range(6):
        tail = target.advance_history(None, tentative, np.array([accepted, accepted]))
        expected_tail = np.concatenate(
            (np.full((2, 3), -1, dtype=np.int64), tentative[:, :accepted]), axis=1
        )[:, -3:]
        np.testing.assert_array_equal(tail, expected_tail)
    with pytest.raises(ValueError, match="Accepted lengths"):
        target.advance_history(None, tentative, [6, 0])


@pytest.fixture
def projection_factory(monkeypatch):
    from aiter import QuantType

    from atom.config import QuantizationConfig
    from atom.model_ops import linear
    from atom.quant_spec import LayerQuantConfig

    monkeypatch.setattr(
        linear, "get_tp_group", lambda: SimpleNamespace(rank_in_group=0, world_size=1)
    )

    def make(k, n, fp4=False):
        config = QuantizationConfig()
        config.global_spec = LayerQuantConfig(
            quant_type=QuantType.per_1x32,
            quant_dtype=torch.float4_e2m1fn_x2 if fp4 else torch.float8_e4m3fn,
            weight_block_size=(1, 32) if fp4 else (32, 32),
            activation_dtype=torch.float8_e4m3fn,
        )
        return linear.ReplicatedLinear(k, n, quant_config=config).cuda()

    return make


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_w4a8_expert_against_reference(reference, projection_factory):
    from .oracle_kernels import fp4_act_quant

    torch.manual_seed(193)
    with bf16_default():
        ref = reference.Expert(64, 96, dtype=torch.float4_e2m1fn_x2, swiglu_limit=10.0)
        target = Expert(
            projection_factory(64, 96, True),
            projection_factory(96, 64, True),
            projection_factory(64, 96, True),
        )
        for name in ("w1", "w2", "w3"):
            source, destination = getattr(ref, name), getattr(target, name)
            n, packed_k = source.weight.shape
            weight, scale = fp4_act_quant(
                torch.randn(n, packed_k * 2, dtype=torch.bfloat16) * 0.25
            )
            source.weight.data.view(torch.uint8).copy_(weight.view(torch.uint8))
            source.scale.data.copy_(scale)
            destination.weight_loader(destination.weight, weight)
            destination.weight_loader(destination.weight_scale, scale)
            destination.process_weights_after_loading()
        x = torch.randn(9, 64, dtype=torch.bfloat16) * 3
        weights = torch.tensor(
            [0.01, 0.137, 0.3, 0.8, 1.0, 0.77, 0.15, 0.35, 0.99], dtype=torch.float32
        )[:, None]
        expected = ref(x, weights)
        actual = target(x.cuda(), weights.cuda()).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_native_engram_projection_and_gpu_gate(reference, projection_factory):
    torch.manual_seed(114)
    with bf16_default():
        projection = projection_factory(64, 160)
        target = EngramOp(
            1, hidden_size=32, engram_hidden_size=64, hc_mult=4, projection=projection
        ).cuda()
        source_projection = reference.Linear(64, 160, dtype=torch.float8_e4m3fn)
        weight = (torch.randn(160, 64, dtype=torch.float32) * 16).to(
            torch.float8_e4m3fn
        )
        scale = torch.exp2(torch.randint(-8, -4, (5, 2)).float()).to(
            torch.float8_e8m0fnu
        )
        q_weight = torch.randn(4, 32, dtype=torch.bfloat16) * 0.1 + 1
        k_weight = torch.randn(4, 32, dtype=torch.bfloat16) * 0.1 + 1
        target.load_checkpoint_weights(weight, k_weight, q_weight, scale)
        source_projection.weight.data.copy_(weight)
        source_projection.scale.data.copy_(scale)
        ref = reference.Engram.__new__(reference.Engram)
        nn.Module.__init__(ref)
        ref.dim, ref.hc_mult, ref.eps, ref.clamp_value = 32, 4, 1e-20, 1e-6
        ref.embed, ref.wkv = nn.Identity(), source_projection
        ref.q_weight, ref.k_weight = nn.Parameter(q_weight), nn.Parameter(k_weight)
        hidden = torch.randn(2, 5, 4, 32, dtype=torch.bfloat16)
        embeddings = torch.randn(2, 5, 2, 32, dtype=torch.bfloat16)
        mask = torch.tensor(
            [[True, False, True, True, False], [False, True, True, False, True]]
        )
        expected = ref(hidden, embeddings, mask)
        actual = target(hidden.cuda(), embeddings.flatten(-2).cuda(), mask.cuda()).cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert target.wkv.weight.dtype == torch.float8_e4m3fn
    assert target.gate_weight.dtype == torch.float32


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
def test_real_engram_weights_and_native_table_rows(reference, projection_factory):
    from atom.config import get_hf_config
    from atom.models.deepseek_v41.weights import (
        CheckpointReader,
        build_weight_manifest,
        checkpoint_schema,
    )

    directory = os.environ["ATOM_DSV41_REFERENCE"]
    config = get_hf_config(directory)
    schema = checkpoint_schema(config)
    manifest = {entry.source.name: entry for entry in build_weight_manifest(schema)}
    with CheckpointReader(directory, schema) as reader, bf16_default():
        table = reader.engram_tables(config)[1]
        rows = (np.arange(24) * 16000000 + 7)[None, None, :]
        embeddings = table.gather(rows, out_dtype=torch.bfloat16)
        projection = projection_factory(6144, 25600)
        target = EngramOp(1, projection=projection).cuda()
        tensors = {
            name: reader.read(manifest[f"layers.1.engram.{name}"])
            for name in ("wkv.weight", "wkv.scale", "k_weight", "q_weight")
        }
        target.load_checkpoint_weights(
            tensors["wkv.weight"],
            tensors["k_weight"],
            tensors["q_weight"],
            tensors["wkv.scale"],
        )
        source_projection = reference.Linear(6144, 25600, dtype=torch.float8_e4m3fn)
        source_projection.weight.data = tensors["wkv.weight"]
        source_projection.scale.data = tensors["wkv.scale"]
        ref = reference.Engram.__new__(reference.Engram)
        nn.Module.__init__(ref)
        ref.dim, ref.hc_mult, ref.eps, ref.clamp_value = 5120, 4, 1e-20, 1e-6
        ref.embed, ref.wkv = nn.Identity(), source_projection
        ref.q_weight = nn.Parameter(tensors["q_weight"])
        ref.k_weight = nn.Parameter(tensors["k_weight"])
        torch.manual_seed(426)
        hidden = torch.randn(1, 1, 4, 5120, dtype=torch.bfloat16)
        expected = ref(hidden, embeddings)
        actual = target(hidden.cuda(), embeddings.flatten(-2).cuda()).cpu()
        # At BF16 output precision, tolerate at most one ulp away from zero;
        # FP32 reduction order differs across the CPU and GPU implementations.
        torch.testing.assert_close(actual, expected, rtol=1 / 128, atol=2**-10)
