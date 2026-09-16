# SPDX-License-Identifier: MIT
"""CSA2 operator/ownership differential tests against pinned upstream methods."""

import pytest
import torch

from atom.model_ops.deepseek_v41.compressor import Compressor
from atom.model_ops.deepseek_v41.rotary import RotaryEmbedding


@pytest.mark.parametrize("original_length,base", [(0, 10000), (65536, 160000)])
def test_rope_matches_reference_interleaved_and_inverse(
    reference, original_length, base
):
    torch.manual_seed(723)
    target = RotaryEmbedding(
        64, 1024, base=base, original_length=original_length, factor=16
    )
    frequencies = reference.precompute_freqs_cis(
        64, 1024, original_length, base, 16, 32, 1
    )
    torch.testing.assert_close(target.frequencies, frequencies, rtol=0, atol=0)
    positions = torch.tensor([0, 1, 97, 511, 1023])
    for shape in ((2, 5, 512), (2, 5, 8, 512)):
        source = torch.randn(shape, dtype=torch.bfloat16)
        for inverse in (False, True):
            expected = source.clone()
            reference.apply_rotary_emb(
                expected[..., -64:], frequencies[positions], inverse
            )
            actual = target(source.clone(), positions, inverse=inverse)
            assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("ratio", [1, 2])
def test_compressor_all_chunk_boundaries_against_official_decode(
    reference, single_rank, ratio
):
    torch.manual_seed(317)
    args = reference.ModelArgs(
        dim=64, head_dim=64, compress_ratios=(ratio,), max_batch_size=2, max_seq_len=16
    )
    with reference.set_dtype(torch.bfloat16):
        source = reference.Compressor(args, 0)
        target = Compressor(64, 64, ratio, args.norm_eps).cuda()
        for name, parameter in target.named_parameters():
            value = (torch.randn(parameter.shape) * 0.1).bfloat16()
            if name == "norm.weight":
                value.fill_(1)
            parameter.data.copy_(value)
            dict(source.named_parameters())[name].data.copy_(value)
        target.process_weights_after_loading()
        hidden = torch.randn(2, 11, 64, dtype=torch.bfloat16)
        expected = []
        for position in range(hidden.shape[1]):
            latent = source(hidden[:, position : position + 1], position)
            if latent is not None:
                expected.append(latent)
        expected = torch.cat(expected, dim=1)
        for chunks in ((11,), (1, 3, 2, 5), (2, 1, 7, 1)):
            position, tail, actual = 0, None, []
            for length in chunks:
                latent, tail = target(
                    hidden[:, position : position + length].cuda(), position, tail
                )
                if latent is not None:
                    actual.append(latent.cpu())
                position += length
            torch.testing.assert_close(
                torch.cat(actual, dim=1), expected, rtol=1 / 128, atol=2**-10
            )
            assert (tail is None) == (ratio == 1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="ROCm GPU required")
@pytest.mark.parametrize("chunks", [(6,), (3, 1, 1, 1), (1, 3, 2), (2, 1, 3)])
def test_full_reuse_reindex_attention_prefill_and_decode(
    reference, single_rank, small_config, chunks, attention_contract
):
    from atom.model_ops.attentions.deepseek_v41_state import EagerAttentionCache
    from atom.models.deepseek_v41.attention import Attention
    from atom.models.deepseek_v41.config import build_attention_topology

    captured, check_attention = attention_contract
    torch.manual_seed(196)
    config = small_config
    config.index_topk = 16
    topology = build_attention_topology(config)
    args = reference.ModelArgs(
        dim=64,
        n_heads=config.num_attention_heads,
        head_dim=config.head_dim,
        q_lora_rank=32,
        o_groups=config.o_groups,
        o_lora_rank=32,
        window_size=4,
        index_n_heads=32,
        index_head_dim=32,
        index_topk=16,
        candidate_block_size=2,
        candidate_topk_blocks=4,
        compress_ratios=(0, 2, 2, 1, 1),
        kv_source_layers=(1, 3),
        index_source_layers=(1, 3, 4),
        candidate_source_layer=3,
        n_layers=5,
        max_batch_size=1,
        max_seq_len=32,
        rope_head_dim=32,
    )
    with reference.set_dtype(torch.bfloat16):
        targets = [Attention(config, spec).cuda() for spec in topology]
        sources = [reference.Attention(spec.layer_id, args) for spec in topology]
        for target, source in zip(targets, sources):
            source_params = dict(source.named_parameters())
            for name, parameter in target.named_parameters():
                source_name = name.replace(".weight_scale", ".scale")
                if parameter.dtype == torch.float8_e8m0fnu:
                    value = torch.full(parameter.shape, 2**-5).to(parameter.dtype)
                elif parameter.dtype == torch.float8_e4m3fn:
                    value = (torch.randn(parameter.shape) * 8).to(parameter.dtype)
                elif name.endswith("norm.weight"):
                    value = torch.ones(parameter.shape, dtype=parameter.dtype)
                else:
                    value = (torch.randn(parameter.shape) * 0.1).to(parameter.dtype)
                parameter.data.copy_(value)
                if not name.startswith("wo_a."):
                    source_params[source_name].data.copy_(value)
            # The model now allocates the checkpoint FP8 wo_a and dequantizes
            # it after loading; the official module stores that weight as BF16.
            target.process_weights_after_loading()
            source.wo_a.weight.data.copy_(target.wo_a.weight)
            if target.compressor is not None:
                target.compressor.process_weights_after_loading()
        window_rope = RotaryEmbedding(32, 32, base=args.rope_theta).cuda()
        global_rope = RotaryEmbedding(
            32,
            32,
            base=args.compress_rope_theta,
            original_length=args.original_seq_len,
            factor=args.rope_factor,
            beta_fast=args.beta_fast,
            beta_slow=args.beta_slow,
        ).cuda()
        hidden = [torch.randn(1, 6, 64, dtype=torch.bfloat16) for _ in topology]
        # The official single prefill is causal for every query. Compare chunked
        # target execution against it, including groups split at odd positions.
        # Keep all six positions here; top-k policy has separate contract tests.
        expected = [layer(values, 0) for layer, values in zip(sources, hidden)]
        cache = EagerAttentionCache(config, topology, 1, 32, "cuda")
        assert set(cache.main) == {1, 3} and set(cache.index) == {1, 3}
        position = 0
        for length in chunks:
            step = cache.begin_step(position, length, 1)
            for spec, target, values, output in zip(
                topology, targets, hidden, expected
            ):
                with check_attention(
                    [captured[spec.layer_id][:, position : position + length]]
                ):
                    actual = target(
                        values[:, position : position + length].cuda(),
                        cache,
                        step,
                        global_rope if spec.ratio else window_rope,
                    ).cpu()
                torch.testing.assert_close(
                    actual,
                    output[:, position : position + length],
                    rtol=1 / 64,
                    atol=2**-10,
                    msg=lambda message, position=position, layer=spec.layer_id: f"position={position}, layer={layer}: {message}",
                )
            cache.finish_step(step)
            position += length
        assert set(step.indices) == {1, 3, 4}
        assert set(step.candidates) == {3}
        assert cache.position == 6
