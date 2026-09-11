# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""CPU tests for DeepSeek-V4.1-Flash (``atom/models/deepseek_v41.py``).

No checkpoint and no GPU kernel: a tiny config drives a real model tree, and
every numeric claim is checked against a direct transcription of
``/data/DeepSeek-V4.1-Flash/inference/model.py`` written out in this file.

Two things are structural rather than numeric, and deliberately so. The MoE
gate and the SwiGLU clamp both end up inside AITER kernels that need a GPU, so
what is tested here is the reference formula on one side and the arguments the
model hands those kernels on the other -- which is the wiring that actually
regresses.
"""

import json
import math
import os
import socket
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

pytest.importorskip("aiter", reason="needs the AITER kernel library to import atom")

import atom.models.deepseek_v41 as v41
from atom.models.deepseek_v41 import (
    DeepseekV41Args,
    DeepseekV41Block,
    V41LayerMap,
    V41SharedRuntime,
    _fp4_roundtrip,
    _fp8_roundtrip,
    _rope_,
    _sparse_attn,
    _window_topk_idxs,
    select_candidate_blocks,
)
from atom.models.deepseek_v41_engram import NgramHashState

CHECKPOINT = "/data/DeepSeek-V4.1-Flash"
# The reference-parity fixture converted to an HF layout: 6 layers with the
# production structure, bf16, built by tools/dsv41/make_mini_ckpt.py.
MINI_CKPT = "/app/ATOM/tools/dsv41/mini_ckpt"

# --------------------------------------------------------------------------
# tiny config
# --------------------------------------------------------------------------

TINY_TEXT = dict(
    model_type="deepseek_v41_text",
    vocab_size=64,
    hidden_size=256,
    num_hidden_layers=6,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=64,
    qk_rope_head_dim=16,
    q_lora_rank=32,
    o_lora_rank=32,
    o_groups=2,
    hidden_act="silu",
    swiglu_limit=10.0,
    rms_norm_eps=1e-20,
    max_position_embeddings=512,
    rope_theta=10000,
    rope_scaling=dict(
        rope_type="yarn",
        factor=16,
        beta_fast=32,
        beta_slow=1,
        original_max_position_embeddings=64,
    ),
    moe_intermediate_size=64,
    n_routed_experts=8,
    n_shared_experts=1,
    num_experts_per_tok=2,
    scoring_func="sqrtsoftplus",
    topk_method="noaux_tc",
    norm_topk_prob=True,
    routed_scaling_factor=1.5,
    sliding_window=8,
    compress_ratios=[0, 0, 2, 2, 1, 1, 0],
    compress_rope_theta=160000,
    kv_source_layer_ids=[2, 4],
    index_source_layer_ids=[2, 4, 5],
    index_n_heads=2,
    index_head_dim=32,
    index_topk=8,
    candidate_source_layer_id=4,
    candidate_topk_blocks=4,
    candidate_block_size=2,
    hc_mult=4,
    hc_sinkhorn_iters=20,
    hc_eps=1e-6,
    engram_layer_ids=[1],
    engram_num_embeddings=[4096],
    engram_max_ngram_size=4,
    engram_vocab_size=256,
    engram_n_heads=2,
    engram_head_dim=32,
    engram_pad_token_id=2,
    engram_compressed_vocab_size=64,
    num_nextn_predict_layers=3,
)

TINY_FULL = dict(
    architectures=["DeepseekV41ForCausalLM"],
    model_type="deepseek_v41",
    dtype="bfloat16",
    bos_token_id=0,
    eos_token_id=1,
    pad_token_id=2,
    image_token_id=60,
    text_config=TINY_TEXT,
    vision_config=dict(
        model_type="deepseek_v41_vision",
        num_hidden_layers=2,
        hidden_size=32,
        num_attention_heads=2,
        intermediate_size=64,
        patch_size=14,
        rope_theta=10000,
        downsample_ratio=3,
        max_image_tokens=16,
        min_pixels=1024,
    ),
)


@pytest.fixture(scope="module")
def tiny_model_dir():
    with tempfile.TemporaryDirectory() as path:
        with open(os.path.join(path, "config.json"), "w") as handle:
            json.dump(TINY_FULL, handle)
        yield path


@pytest.fixture(scope="module")
def tiny_config(tiny_model_dir):
    from atom.config import Config

    return Config(
        model=tiny_model_dir,
        max_model_len=256,
        max_num_batched_tokens=256,
        enforce_eager=True,
    )


@pytest.fixture(scope="module")
def tp_group():
    """A world=1 TP group, which is all the parallel layers need to build."""
    import torch.distributed as dist

    try:
        from aiter.dist.parallel_state import (
            init_distributed_environment,
            initialize_model_parallel,
        )
    except ImportError as exc:  # pragma: no cover - AITER is present in CI
        pytest.skip(f"requires aiter: {exc}")
    if not dist.is_initialized():
        if not torch.cuda.is_available():
            pytest.skip("the parallel state needs a device to bind to")
        # A fixed port collides with a leftover store from an earlier run, so
        # let the OS hand out a free one.
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = str(port)
        torch.cuda.set_device(0)
        init_distributed_environment(
            world_size=1,
            rank=0,
            local_rank=0,
            distributed_init_method=f"tcp://127.0.0.1:{port}",
        )
        initialize_model_parallel(tensor_model_parallel_size=1)
    yield


@pytest.fixture
def tiny_model(tp_group, tiny_config):
    """The real model tree on CPU. Parameters only -- no kernel is launched.

    Function-scoped on purpose: several tests swap a sub-module out for a
    torch-only stand-in, and a shared tree would carry that into the next one.
    FusedMoE refuses a prefix it has already seen, so the per-forward registry
    is cleared before each build.
    """
    from atom.config import set_current_atom_config
    from atom.models.deepseek_v41 import DeepseekV41ForCausalLM

    set_current_atom_config(tiny_config)
    tiny_config.compilation_config.static_forward_context.clear()
    torch.manual_seed(0)
    model = DeepseekV41ForCausalLM(tiny_config)
    for param in model.parameters():
        if param.dtype in (torch.float32, torch.bfloat16, torch.float16):
            param.data.normal_(0, 0.05)
    return model


# --------------------------------------------------------------------------
# reference transcriptions (inference/model.py + inference/kernel.py)
# --------------------------------------------------------------------------


def ref_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:

        def corrected_dim(rotations):
            return (
                dim
                * math.log(original_seq_len / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), dim - 1)
        ramp = (
            (torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)
        ).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / factor * (1 - smooth) + freqs * smooth
    freqs = torch.outer(torch.arange(seqlen), freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def ref_rope(x, freqs_cis, inverse=False):
    """Adjacent element pairs as one complex number; `inverse` conjugates."""
    out = x.clone().float()
    flat = out.reshape(-1, out.shape[-1])
    freqs = freqs_cis.conj() if inverse else freqs_cis
    per_token = flat.shape[0] // freqs.shape[0]
    for row in range(flat.shape[0]):
        row_freqs = freqs[row // per_token]
        for pair in range(flat.shape[-1] // 2):
            real, imag = flat[row, 2 * pair].item(), flat[row, 2 * pair + 1].item()
            f = row_freqs[pair]
            value = complex(real, imag) * complex(f.real.item(), f.imag.item())
            flat[row, 2 * pair] = value.real
            flat[row, 2 * pair + 1] = value.imag
    return out.to(x.dtype)


def ref_rms(x, weight, eps):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (weight.float() * xf).to(x.dtype)


def ref_sparse_attn(q, kv, attn_sink, topk_idxs, scale):
    """One softmax per (token, head); the sink only in the denominator."""
    tokens, heads, dim = q.shape
    out = torch.zeros(tokens, heads, dim, dtype=torch.float32)
    for token in range(tokens):
        for head in range(heads):
            logits, rows = [], []
            for slot in topk_idxs[token].tolist():
                if slot < 0:
                    continue
                rows.append(kv[slot].float())
                logits.append(
                    float(q[token, head].detach().float() @ kv[slot].detach().float())
                    * scale
                )
            sink = float(attn_sink[head])
            peak = max(logits + [sink])
            weights = [math.exp(v - peak) for v in logits]
            denom = sum(weights) + math.exp(sink - peak)
            acc = torch.zeros(dim, dtype=torch.float32)
            for weight, row in zip(weights, rows, strict=True):
                acc += weight * row
            out[token, head] = acc / denom
    return out


def ref_compressor(x, wkv, wgate, norm_weight, eps, ratio, state, start_pos):
    """inference/model.py:Compressor.forward, on a [1, s, dim] batch."""
    if ratio == 1:
        return ref_rms(F.linear(x, wkv), norm_weight, eps)
    x = x.float()
    kv, score = F.linear(x, wkv.float()), F.linear(x, wgate.float())
    seqlen = x.shape[1]
    if start_pos == 0:
        should = seqlen >= ratio
        remainder = seqlen % ratio
        cutoff = seqlen - remainder
        if remainder:
            state["kv"][:, :remainder] = kv[:, cutoff:]
            state["score"][:, :remainder] = score[:, cutoff:]
            kv, score = kv[:, :cutoff], score[:, :cutoff]
        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio))
        kv = (kv * score.softmax(dim=2)).sum(dim=2)
    else:
        should = (start_pos + 1) % ratio == 0
        slot = start_pos % ratio
        state["kv"][:, slot] = kv.squeeze(1)
        state["score"][:, slot] = score.squeeze(1)
        if should:
            kv = (state["kv"] * state["score"].softmax(dim=1)).sum(dim=1, keepdim=True)
    if not should:
        return None
    return ref_rms(kv, norm_weight, eps)


def ref_select_candidate_blocks(logits, compress_lens, topk_blocks, block_size):
    width = logits.size(-1)
    scores = F.pad(logits, (0, -width % block_size), value=-torch.inf)
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(torch.arange(num_blocks) == last, torch.inf)
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    keep = torch.zeros_like(scores, dtype=torch.bool).scatter_(
        -1, top.indices, top.values > -torch.inf
    )
    return keep.repeat_interleave(block_size, dim=-1)[..., :width]


def ref_hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, iters, eps):
    """inference/kernel.py:hc_split_sinkhorn -- 20 column and 19 row passes."""
    pre = torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult]) + eps
    post = 2 * torch.sigmoid(
        mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1] + hc_base[hc_mult : 2 * hc_mult]
    )
    comb = mixes[..., 2 * hc_mult :].unflatten(-1, (hc_mult, hc_mult)) * hc_scale[
        2
    ] + hc_base[2 * hc_mult :].unflatten(-1, (hc_mult, hc_mult))
    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def ref_gate(x, weight, bias, topk, route_scale):
    """inference/model.py:Gate.forward for score_func='sqrtsoftplus'."""
    scores = F.softplus(F.linear(x.float(), weight.float())).sqrt()
    indices = (scores + bias).topk(topk, dim=-1)[1]
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights * route_scale, indices


def ref_swiglu(gate, up, limit):
    """inference/model.py:Expert.forward -- up clamped both sides, gate above."""
    up = torch.clamp(up, min=-limit, max=limit)
    gate = torch.clamp(gate, max=limit)
    return F.silu(gate) * up


def ref_engram(x, key, value, q_weight, k_weight, dim, eps):
    weight = q_weight.float() * k_weight.float()
    h = x.float()
    rstd = torch.rsqrt(h.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (h * weight * key).sum(-1) * rstd * dim**-0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
    return (h + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


class PlainLinear(nn.Module):
    """A torch Linear that borrows an ATOM Linear's weight, so CPU can run it."""

    def __init__(self, weight: torch.Tensor):
        super().__init__()
        self.weight = nn.Parameter(weight.detach().clone().float())

    def forward(self, x):
        return F.linear(x.float(), self.weight)


def plainify(module, *names):
    for name in names:
        setattr(module, name, PlainLinear(getattr(module, name).weight))


class ModuleFn(nn.Module):
    """A stand-in sub-module, because nn.Module refuses a bare lambda."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


# --------------------------------------------------------------------------
# layer maps and config
# --------------------------------------------------------------------------


def tiny_args():
    from transformers import AutoConfig

    hf = AutoConfig.for_model("deepseek_v3").from_dict(dict(TINY_TEXT))
    for key, value in TINY_TEXT.items():
        if not hasattr(hf, key):
            setattr(hf, key, value)
    return DeepseekV41Args.from_hf_config(hf)


def test_tiny_layer_map():
    layer_map = V41LayerMap.from_args(tiny_args())
    # The config list has a 7th entry for a layer the backbone does not have.
    assert layer_map.ratios == (0, 0, 2, 2, 1, 1)
    # `None`, not -1: the row space spells "no owner" that way and a -1 that
    # reaches an index expression selects the last layer instead of raising.
    assert layer_map.kv_owner == (None, None, 2, 2, 4, 4)
    assert layer_map.index_owner == (None, None, 2, 2, 4, 5)
    assert layer_map.candidate_source == 4


def test_layer_map_rejects_a_source_of_the_wrong_ratio():
    args = tiny_args()
    args.kv_source_layer_ids = (2,)
    args.index_source_layer_ids = (2,)
    with pytest.raises(ValueError, match="reads layer 2"):
        V41LayerMap.from_args(args)


def test_layer_map_rejects_a_kv_source_that_publishes_no_index_keys():
    args = tiny_args()
    args.index_source_layer_ids = (2, 5)
    with pytest.raises(ValueError, match="not index sources"):
        V41LayerMap.from_args(args)


@pytest.mark.skipif(
    not os.path.isdir(CHECKPOINT), reason="DeepSeek-V4.1-Flash config not present"
)
def test_released_config_gives_the_documented_layer_map():
    from atom.config import get_hf_config

    args = DeepseekV41Args.from_hf_config(get_hf_config(CHECKPOINT))
    assert (args.dim, args.n_layers, args.n_heads) == (5120, 40, 64)
    assert (args.head_dim, args.rope_head_dim) == (512, 64)
    assert (args.q_lora_rank, args.o_lora_rank, args.o_groups) == (1280, 1024, 8)
    assert (args.moe_inter_dim, args.n_routed_experts) == (2304, 384)
    assert (args.n_activated_experts, args.route_scale) == (6, 1.5)
    assert (args.index_n_heads, args.index_head_dim, args.index_topk) == (32, 128, 512)
    assert args.norm_eps == 1e-20 and args.swiglu_limit == 10.0
    assert args.weight_block_size == 32 and args.expert_dtype == "fp4"
    assert args.engram_layer_ids == (1, 14)
    # The config list carries three MTP layers past the backbone.
    assert len(args.compress_ratios) == 40
    assert args.compress_ratios[:2] == (0, 0)
    assert set(args.compress_ratios[2:20]) == {2}
    assert set(args.compress_ratios[20:]) == {1}

    layer_map = V41LayerMap.from_args(args)
    assert layer_map.kv_sources == (2, 8, 14, 20)
    assert layer_map.index_sources == (2, 8, 14, 20, 24, 28, 32, 36)
    assert layer_map.candidate_source == 20
    assert layer_map.kv_owner[19] == 14 and layer_map.kv_owner[39] == 20
    assert layer_map.index_owner[23] == 20 and layer_map.index_owner[39] == 36


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------


def test_architecture_is_registered():
    from atom.model_engine.model_runner import support_model_arch_dict

    assert (
        support_model_arch_dict["DeepseekV41ForCausalLM"]
        == "atom.models.deepseek_v41.DeepseekV41ForCausalLM"
    )


def test_config_uses_the_text_sub_config_and_the_v4_backend(tiny_config):
    from atom.model_engine.llm_engine import InputOutputProcessor
    from atom.utils.selector import Family, attn_family

    hf = tiny_config.hf_config
    assert hf.model_type == "deepseek_v41_text"
    assert hf.architectures == ["DeepseekV41ForCausalLM"]
    # A V4.1-only field the deepseek_v3 schema does not declare.
    assert list(hf.kv_source_layer_ids) == [2, 4]
    assert attn_family(hf) is Family.V4
    assert tiny_config.kv_cache_block_size == 256
    assert hf.model_type in InputOutputProcessor._per_req_cache_model_types()


# --------------------------------------------------------------------------
# module layout and parameter names
# --------------------------------------------------------------------------


def test_only_source_layers_own_a_compressor_or_an_indexer(tiny_model):
    layers = tiny_model.model.layers
    assert [i for i, m in enumerate(layers) if m.attn.compressor is not None] == [2, 4]
    assert [i for i, m in enumerate(layers) if m.attn.indexer is not None] == [2, 4, 5]
    assert [i for i, m in enumerate(layers) if m.engram is not None] == [1]
    # Only the ratio > 1 compressor has a gate; ratio 1 is a plain projection.
    assert hasattr(layers[2].attn.compressor, "wgate")
    assert not hasattr(layers[4].attn.compressor, "wgate")
    # Index keys come off the compressor latent, so only a kv source has wk.
    assert layers[2].attn.indexer.owns_k and layers[4].attn.indexer.owns_k
    assert not layers[5].attn.indexer.owns_k
    assert layers[4].attn.indexer.is_candidate_source
    assert layers[5].attn.indexer.uses_candidates
    assert not layers[2].attn.indexer.uses_candidates


def synthetic_checkpoint_names(args):
    """Every tensor the released checkpoint holds, at the tiny config's sizes."""
    names = ["embed.weight", "norm.weight", "head.weight"]
    names += ["image_start", "image_end", "image_newline"]
    names += ["vision.blocks.0.attn.weight", "aligner.proj.weight"]
    names += ["mtp.0.attn.wq_a.weight"]
    layer_map = V41LayerMap.from_args(args)
    for layer_id in range(args.n_layers):
        base = f"layers.{layer_id}"
        names += [f"{base}.attn_norm.weight", f"{base}.ffn_norm.weight"]
        names += [
            f"{base}.hc_attn_fn",
            f"{base}.hc_ffn_fn",
            f"{base}.hc_attn_base",
            f"{base}.hc_ffn_base",
            f"{base}.hc_attn_scale",
            f"{base}.hc_ffn_scale",
        ]
        names += [f"{base}.attn.attn_sink"]
        names += [f"{base}.attn.q_norm.weight", f"{base}.attn.kv_norm.weight"]
        for proj in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"):
            names += [f"{base}.attn.{proj}.weight", f"{base}.attn.{proj}.scale"]
        if layer_id in layer_map.kv_sources:
            names += [
                f"{base}.attn.compressor.wkv.weight",
                f"{base}.attn.compressor.norm.weight",
            ]
            if args.compress_ratios[layer_id] > 1:
                names += [f"{base}.attn.compressor.wgate.weight"]
            names += [
                f"{base}.attn.indexer.wk.weight",
                f"{base}.attn.indexer.k_norm.weight",
            ]
        if layer_id in layer_map.index_sources:
            names += [
                f"{base}.attn.indexer.wq_b.weight",
                f"{base}.attn.indexer.wq_b.scale",
                f"{base}.attn.indexer.weights_proj.weight",
            ]
        if layer_id in args.engram_layer_ids:
            names += [
                f"{base}.engram.embed.weight",
                f"{base}.engram.embed.scale",
                f"{base}.engram.wkv.weight",
                f"{base}.engram.wkv.scale",
                f"{base}.engram.q_weight",
                f"{base}.engram.k_weight",
            ]
        names += [
            f"{base}.ffn.gate.weight",
            f"{base}.ffn.gate.bias",
            f"{base}.ffn.gate.bias_vl",
        ]
        for proj in ("w1", "w2", "w3"):
            names += [f"{base}.ffn.shared_experts.{proj}.weight"]
            names += [f"{base}.ffn.shared_experts.{proj}.scale"]
            for expert in range(args.n_routed_experts):
                names += [f"{base}.ffn.experts.{expert}.{proj}.weight"]
                names += [f"{base}.ffn.experts.{expert}.{proj}.scale"]
    return names


def resolve_checkpoint_names(model, names):
    """Walk on-disk names to parameter names the way the loader does.

    Returns (claimed parameter names, [(disk name, rewritten) that hit nothing]).
    The rewriter and the expert mapping are the real ones; only the loop around
    them is local, because the loader's own loop also writes tensors.
    """
    from atom.model_loader.weight_names import CheckpointNameRewriter

    rewriter = CheckpointNameRewriter(
        weights_mapper=model.weights_mapper,
        weights_mapping=model.weights_mapping,
        skip_weight_prefixes=model.skip_weight_prefixes,
        num_hidden_layers=model.args.n_layers,
        n_routed_experts=model.args.n_routed_experts,
        fuse_shared_expert=lambda *_: not model.disable_fused_shared_loading,
        disable_fused_shared_loading=model.disable_fused_shared_loading,
    )
    expert_parts = {
        weight_part: param_part
        for param_part, weight_part, _, _ in model.get_expert_mapping()
    }
    packed = model.packed_modules_mapping
    params = set(dict(model.named_parameters()))

    claimed, dropped = set(), []
    for name in names:
        rewritten = rewriter.rewrite(name)
        if rewritten is None:
            continue
        for weight_part, param_part in expert_parts.items():
            if weight_part in rewritten:
                rewritten = rewritten.replace(weight_part, param_part)
                break
        else:
            for disk_part, (param_part, _) in packed.items():
                if disk_part in rewritten:
                    rewritten = rewritten.replace(disk_part, param_part)
                    break
        if rewritten in params:
            claimed.add(rewritten)
        else:
            dropped.append((name, rewritten))
    return claimed, dropped


def test_checkpoint_names_reach_the_parameters_the_model_declares(tiny_model):
    """Rewrite a synthetic checkpoint and check both directions of the map.

    Every model parameter must be claimed by some tensor, and every tensor must
    either be claimed or be one this text-only model means to drop.
    """
    claimed, dropped = resolve_checkpoint_names(
        tiny_model, synthetic_checkpoint_names(tiny_model.args)
    )
    # A scale companion has no home for a layer this config leaves unquantized.
    dropped = [(a, b) for a, b in dropped if not b.endswith("weight_scale")]
    assert dropped == [], dropped[:10]
    params = set(dict(tiny_model.named_parameters()))
    unclaimed = sorted(params - claimed)
    assert unclaimed == [], unclaimed[:10]


@pytest.mark.skipif(
    not os.path.isdir(MINI_CKPT), reason="mini parity checkpoint not built"
)
def test_every_mini_checkpoint_tensor_lands_on_a_parameter(tp_group):
    """The names in the parity checkpoint are the production names.

    The tensors are not actually written here: the loader's post-load hooks
    reach AITER (FusedMoE reshuffles its expert buffers), which a CPU run
    cannot do. What is checked is the mapping, in both directions.
    """
    from atom.config import Config, set_current_atom_config
    from atom.models.deepseek_v41 import DeepseekV41ForCausalLM

    config = Config(
        model=MINI_CKPT,
        max_model_len=256,
        max_num_batched_tokens=256,
        enforce_eager=True,
    )
    set_current_atom_config(config)
    config.compilation_config.static_forward_context.clear()
    model = DeepseekV41ForCausalLM(config)

    with open(os.path.join(MINI_CKPT, "model.safetensors.index.json")) as handle:
        names = sorted(json.load(handle)["weight_map"])
    assert len(names) > 200 and "layers.2.attn.compressor.wgate.weight" in names

    claimed, dropped = resolve_checkpoint_names(model, names)
    assert dropped == [], dropped[:10]
    params = set(dict(model.named_parameters()))
    unclaimed = sorted(params - claimed)
    assert unclaimed == [], unclaimed[:10]


def test_the_vl_routing_bias_and_the_vision_tower_are_dropped(tiny_model):
    from atom.model_loader.weight_names import CheckpointNameRewriter

    rewriter = CheckpointNameRewriter(
        weights_mapper=tiny_model.weights_mapper,
        weights_mapping=tiny_model.weights_mapping,
        skip_weight_prefixes=tiny_model.skip_weight_prefixes,
        num_hidden_layers=tiny_model.args.n_layers,
    )
    for name in (
        "layers.0.ffn.gate.bias_vl",
        "vision.blocks.0.attn.weight",
        "aligner.proj.weight",
        "mtp.0.attn.wq_a.weight",
        "image_start",
        "image_end",
        "image_newline",
    ):
        assert rewriter.rewrite(name) is None, name
    # And the ones next to them survive.
    assert (
        rewriter.rewrite("layers.0.ffn.gate.bias")
        == "model.layers.0.ffn.gate.e_score_correction_bias"
    )
    assert rewriter.rewrite("norm.weight") == "model.norm.weight"
    assert rewriter.rewrite("layers.0.attn_norm.weight") == (
        "model.layers.0.attn_norm.weight"
    )


# --------------------------------------------------------------------------
# RoPE
# --------------------------------------------------------------------------


def test_rope_tables_match_the_reference_yarn(tiny_model):
    """Ratio 0 uses theta 1e4 and no YaRN; the rest use 160000 with YaRN."""
    args = tiny_model.args
    positions = torch.arange(17)
    for layer_id, (base, original) in (
        (0, (args.rope_theta, 0)),
        (2, (args.compress_rope_theta, args.original_seq_len)),
        (4, (args.compress_rope_theta, args.original_seq_len)),
    ):
        rope = tiny_model.model.layers[layer_id].attn.rotary_emb
        assert rope.base == base and rope.original_seq_len == original
        expected = ref_freqs_cis(
            args.rope_head_dim,
            positions.numel(),
            original,
            base,
            args.rope_factor,
            args.beta_fast,
            args.beta_slow,
        )
        got = rope.freqs_for_positions(positions)
        # The cos/sin cache is bf16 on purpose; this checks the convention.
        torch.testing.assert_close(got.real, expected.real, atol=5e-3, rtol=0)
        torch.testing.assert_close(got.imag, expected.imag, atol=5e-3, rtol=0)


def test_rope_rotates_adjacent_pairs_and_inverts(tiny_model):
    rope = tiny_model.model.layers[2].attn.rotary_emb
    positions = torch.arange(5)
    freqs = rope.freqs_for_positions(positions)
    x = torch.randn(5, 3, 16)
    got = _rope_(x.clone(), freqs)
    torch.testing.assert_close(got, ref_rope(x, freqs), atol=1e-5, rtol=1e-5)
    # The conjugate undoes the rotation only as far as the bf16 cos/sin cache
    # allows: those pairs sit a bf16 epsilon off the unit circle.
    back = _rope_(got.clone(), freqs, inverse=True)
    torch.testing.assert_close(back, x, atol=2e-2, rtol=1e-2)


# --------------------------------------------------------------------------
# sparse attention primitives
# --------------------------------------------------------------------------


def test_sparse_attn_matches_a_per_query_reference():
    torch.manual_seed(1)
    q = torch.randn(4, 3, 8)
    kv = torch.randn(6, 8)
    sink = torch.randn(3)
    idxs = torch.tensor(
        [[0, -1, -1], [0, 1, -1], [1, 2, 3], [-1, -1, -1]], dtype=torch.int32
    )
    got = _sparse_attn(q, kv, sink, idxs, 8**-0.5)
    expected = ref_sparse_attn(q, kv, sink, idxs, 8**-0.5)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)
    # A row that can see nothing returns zero: the sink alone fills the denominator.
    assert torch.count_nonzero(got[3]) == 0


def test_the_sink_only_enters_the_denominator():
    q = torch.zeros(1, 1, 4)
    kv = torch.ones(1, 4)
    idxs = torch.zeros(1, 1, dtype=torch.int32)
    without = _sparse_attn(q, kv, torch.tensor([-30.0]), idxs, 1.0)
    withsink = _sparse_attn(q, kv, torch.tensor([0.0]), idxs, 1.0)
    torch.testing.assert_close(without, torch.ones(1, 1, 4), atol=1e-6, rtol=0)
    torch.testing.assert_close(withsink, torch.full((1, 1, 4), 0.5), atol=1e-6, rtol=0)


def test_window_indices_are_causal_and_ring_ordered():
    prefill = _window_topk_idxs(4, 6, 0, torch.device("cpu"))
    assert prefill.shape == (6, 4)
    assert prefill[0].tolist() == [0, -1, -1, -1]
    assert prefill[3].tolist() == [0, 1, 2, 3]
    assert prefill[5].tolist() == [2, 3, 4, 5]
    decode = _window_topk_idxs(4, 1, 6, torch.device("cpu"))
    assert sorted(decode[0].tolist()) == [0, 1, 2, 3]


def test_candidate_block_selection_matches_the_reference():
    torch.manual_seed(2)
    logits = torch.randn(5, 9)
    lens = torch.arange(1, 6).unsqueeze(-1)
    logits = logits.masked_fill(torch.arange(9) >= lens, -torch.inf)
    got = select_candidate_blocks(logits, lens, 2, 2)
    expected = ref_select_candidate_blocks(logits, lens, 2, 2)
    assert torch.equal(got, expected)
    # The block holding the newest reachable position is always kept.
    for row in range(5):
        assert got[row, int(lens[row]) - 1]


def test_fp4_roundtrip_lands_on_the_e2m1_grid():
    x = torch.randn(3, 64) * 4
    out = _fp4_roundtrip(x, 32)
    assert out.shape == x.shape
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    blocks = out.unflatten(-1, (-1, 32)).flatten(0, 1)
    for values in blocks:
        peak = values.abs().max()
        # Every value in a block is one grid magnitude times one shared scale;
        # which grid point the largest lands on is not fixed, so try each.
        assert any(
            bool(
                torch.isclose(
                    values.abs().unsqueeze(-1), grid * (peak / top), atol=1e-6
                )
                .any(-1)
                .all()
            )
            for top in grid[1:]
        )
    # A power-of-two scale rounded up puts the block peak in [3, 6] scale
    # units, so no value moves by more than one scale unit, i.e. peak / 3.
    peaks = x.unflatten(-1, (-1, 32)).abs().amax(-1, keepdim=True)
    bound = (peaks / 3.0 + 1e-6).expand(-1, -1, 32).reshape(x.shape)
    assert bool(((out - x).abs() <= bound).all())


# --------------------------------------------------------------------------
# compressor
# --------------------------------------------------------------------------


@pytest.mark.parametrize("ratio, layer_id", [(2, 2), (1, 4)])
def test_compressor_matches_the_reference(tiny_model, ratio, layer_id):
    torch.manual_seed(3)
    compressor = tiny_model.model.layers[layer_id].attn.compressor
    assert compressor.compress_ratio == ratio
    names = ["wkv"] + (["wgate"] if ratio > 1 else [])
    plainify(compressor, *names)
    wkv = compressor.wkv.weight
    wgate = compressor.wgate.weight if ratio > 1 else None
    norm_w, eps = compressor.norm.weight, compressor.eps

    # A prefill whose length is not a multiple of the ratio, then two decodes,
    # so the partial group has to survive across the calls.
    x = torch.randn(7, tiny_model.args.dim)
    state = {
        "kv": torch.zeros(1, ratio, compressor.head_dim),
        "score": torch.full((1, ratio, compressor.head_dim), -torch.inf),
    }
    if ratio > 1:
        compressor.kv_state.zero_()
        compressor.score_state.fill_(-torch.inf)

    got = compressor(x, 0)
    expected = ref_compressor(x.unsqueeze(0), wkv, wgate, norm_w, eps, ratio, state, 0)
    torch.testing.assert_close(got, expected.squeeze(0), atol=1e-5, rtol=1e-5)

    outputs = []
    for step in range(7, 10):
        got = compressor(x[:1], step)
        expected = ref_compressor(
            x[:1].unsqueeze(0), wkv, wgate, norm_w, eps, ratio, state, step
        )
        outputs.append(got is not None)
        if expected is None:
            assert got is None, step
        else:
            torch.testing.assert_close(got, expected.squeeze(0), atol=1e-5, rtol=1e-5)
    # Ratio 1 yields on every step; ratio 2 only when a group completes.
    assert outputs.count(True) == (3 if ratio == 1 else 2)


# --------------------------------------------------------------------------
# indexer
# --------------------------------------------------------------------------


def test_indexer_matches_the_reference_including_the_candidate_pool(tiny_model):
    torch.manual_seed(4)
    args = tiny_model.args
    layers = tiny_model.model.layers
    source, consumer = layers[4].attn.indexer, layers[5].attn.indexer
    plainify(source, "wq_b", "weights_proj", "wk")
    plainify(consumer, "wq_b", "weights_proj")
    shared = V41SharedRuntime()

    num_tokens, ratio = 12, source.compress_ratio
    positions = torch.arange(num_tokens)
    x = torch.randn(num_tokens, args.dim)
    qr = torch.randn(num_tokens, args.q_lora_rank)
    latent = torch.randn(num_tokens // ratio, args.head_dim)
    source.k_cache = None
    shared.index_k = source.publish_keys(latent.clone(), 0)

    def reference(indexer, candidates):
        q = F.linear(qr, indexer.wq_b.weight).view(
            num_tokens, indexer.n_heads, indexer.index_head_dim
        )
        freqs = indexer.rotary_emb.freqs_for_positions(positions)
        rd = indexer.rope_head_dim
        q = torch.cat([q[..., :-rd], ref_rope(q[..., -rd:], freqs)], dim=-1)
        if v41._V41_INDEXER_MXFP4:
            q = _fp4_roundtrip(q)
        index_k = shared.index_k[: num_tokens // ratio]
        weights = F.linear(x, indexer.weights_proj.weight) * indexer._weights_scale
        score = torch.einsum("shd,td->sht", q, index_k)
        score = (score.relu() * weights.unsqueeze(-1)).sum(dim=1)
        lens = (torch.arange(1, num_tokens + 1) // ratio).unsqueeze(-1)
        score = score.masked_fill(torch.arange(num_tokens // ratio) >= lens, -torch.inf)
        if candidates is None:
            pool = ref_select_candidate_blocks(
                score, lens, indexer.candidate_topk_blocks, indexer.candidate_block_size
            )
        else:
            pool = None
            score = score.masked_fill(~candidates, -torch.inf)
        topk = min(indexer.index_topk, num_tokens // ratio)
        idxs = score.topk(topk, dim=-1, sorted=False).indices.sort(dim=-1).values
        return torch.where(idxs < lens, idxs + 3, -1).int(), pool

    got = source(x, qr, positions, 0, 3, shared)
    expected, pool = reference(source, None)
    assert torch.equal(got, expected)
    assert shared.candidates is not None and torch.equal(shared.candidates, pool)

    got = consumer(x, qr, positions, 0, 3, shared)
    expected, _ = reference(consumer, shared.candidates)
    assert torch.equal(got, expected)


def test_indexer_indices_are_ascending_and_masked_out_of_reach(tiny_model):
    torch.manual_seed(5)
    args = tiny_model.args
    indexer = tiny_model.model.layers[2].attn.indexer
    plainify(indexer, "wq_b", "weights_proj", "wk")
    shared = V41SharedRuntime()
    num_tokens, ratio = 10, indexer.compress_ratio
    indexer.k_cache = None
    shared.index_k = indexer.publish_keys(
        torch.randn(num_tokens // ratio, args.head_dim), 0
    )
    idxs = indexer(
        torch.randn(num_tokens, args.dim),
        torch.randn(num_tokens, args.q_lora_rank),
        torch.arange(num_tokens),
        0,
        0,
        shared,
    )
    for token in range(num_tokens):
        row = [v for v in idxs[token].tolist() if v >= 0]
        assert row == sorted(row)
        # A latent stands for `ratio` tokens and is visible only once the query
        # has passed the last of them.
        assert all(v < (token + 1) // ratio for v in row)
    assert idxs[0].tolist() == [-1] * idxs.shape[1]


# --------------------------------------------------------------------------
# attention
# --------------------------------------------------------------------------


def test_dense_layer_attention_matches_the_reference(tiny_model):
    """Layer 0 is window-only, so this covers the ring, the sink, the inverse
    RoPE on the output and the block-diagonal wo_a einsum."""
    torch.manual_seed(6)
    args = tiny_model.args
    attn = tiny_model.model.layers[0].attn
    assert attn.compress_ratio == 0
    plainify(attn, "wq_a", "wq_b", "wkv", "wo_b")
    wo_a = attn.wo_a.weight.detach().clone().float()

    num_tokens = 11
    positions = torch.arange(num_tokens)
    x = torch.randn(num_tokens, args.dim)
    attn.window_kv_cache.zero_()
    got = attn(x, positions)

    rd = args.rope_head_dim
    freqs = attn.rotary_emb.freqs_for_positions(positions)
    qr = ref_rms(F.linear(x, attn.wq_a.weight), attn.q_norm.weight, attn.eps)
    q = F.linear(qr, attn.wq_b.weight)
    q = q.view(num_tokens, attn.n_local_heads, args.head_dim)
    q = torch.cat([q[..., :-rd], ref_rope(q[..., -rd:], freqs)], dim=-1)
    kv = ref_rms(F.linear(x, attn.wkv.weight), attn.kv_norm.weight, attn.eps)
    kv = torch.cat([kv[..., :-rd], ref_rope(kv[..., -rd:], freqs)], dim=-1)
    if v41._V41_SIM_QAT_QUANT:
        # The reference bakes this into its kv write; `_fp8_roundtrip` has its
        # own test, so borrowing it here still leaves the algebra checked.
        kv = _fp8_roundtrip(kv, v41._FP8_BLOCK)
    idxs = _window_topk_idxs(args.window_size, num_tokens, 0, torch.device("cpu"))
    out = ref_sparse_attn(q, kv, attn.attn_sink, idxs, args.head_dim**-0.5)
    out = torch.cat([out[..., :-rd], ref_rope(out[..., -rd:], freqs, True)], dim=-1)
    out = out.view(num_tokens, attn.n_local_groups, -1)
    grouped = wo_a.view(attn.n_local_groups, args.o_lora_rank, -1)
    out = torch.einsum("sgd,grd->sgr", out, grouped)
    expected = F.linear(out.flatten(1), attn.wo_b.weight)
    torch.testing.assert_close(got, expected, atol=2e-4, rtol=2e-4)


# --------------------------------------------------------------------------
# mHC
# --------------------------------------------------------------------------


def test_mhc_coefficients_match_the_reference_formulas(tiny_model):
    torch.manual_seed(7)
    block = tiny_model.model.layers[0]
    residual = torch.randn(5, block.hc_mult, tiny_model.args.dim)
    pre, post, comb = block.hc_mixes(
        residual, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base
    )
    flat = residual.flatten(-2).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + block.norm_eps)
    mixes = F.linear(flat, block.hc_attn_fn) * rsqrt
    exp_pre, exp_post, exp_comb = ref_hc_split_sinkhorn(
        mixes,
        block.hc_attn_scale,
        block.hc_attn_base,
        block.hc_mult,
        block.hc_sinkhorn_iters,
        block.hc_eps,
    )
    torch.testing.assert_close(pre, exp_pre, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(post, exp_post, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(comb, exp_comb, atol=1e-6, rtol=1e-6)
    # pre is a sigmoid plus eps, post is twice a sigmoid, comb is near doubly
    # stochastic after the Sinkhorn passes.
    assert (pre > block.hc_eps).all() and (pre < 1 + block.hc_eps).all()
    assert (post >= 0).all() and (post <= 2).all()
    torch.testing.assert_close(comb.sum(-1), torch.ones(5, 4), atol=1e-3, rtol=0)
    torch.testing.assert_close(comb.sum(-2), torch.ones(5, 4), atol=1e-3, rtol=0)


def test_mhc_uses_the_previous_sub_layers_pre(tiny_model):
    """The single-pass shift: a sub-layer mixes its input with the coefficients
    the sub-layer BEFORE it produced, and a block hands its ffn `pre` on."""
    from atom.models.deepseek_v41 import V41HCState

    torch.manual_seed(8)
    block = tiny_model.model.layers[0]
    dim, hc = tiny_model.args.dim, block.hc_mult
    scale_attn, scale_ffn = nn.Linear(dim, dim), nn.Linear(dim, dim)
    block.attn = ModuleFn(lambda x, positions: scale_attn(x))
    block.ffn = scale_ffn

    residual = torch.randn(3, hc, dim)
    incoming = torch.rand(3, hc)
    state = block(V41HCState(residual=residual, pre_mix=incoming), torch.arange(3))

    # inference/model.py:Block.forward, transcribed.
    attn_pre, attn_post, attn_comb = block.hc_mixes(
        residual, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base
    )
    x = torch.sum(incoming.unsqueeze(-1) * residual, dim=-2)
    x = scale_attn(ref_rms(x, block.attn_norm.weight, block.norm_eps))
    mid = attn_post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        attn_comb.unsqueeze(-1) * residual.unsqueeze(-2), dim=-3
    )
    ffn_pre, ffn_post, ffn_comb = block.hc_mixes(
        mid, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base
    )
    x = torch.sum(attn_pre.unsqueeze(-1) * mid, dim=-2)
    x = scale_ffn(ref_rms(x, block.ffn_norm.weight, block.norm_eps))
    expected = ffn_post.unsqueeze(-1) * x.unsqueeze(-2) + torch.sum(
        ffn_comb.unsqueeze(-1) * mid.unsqueeze(-2), dim=-3
    )

    torch.testing.assert_close(state.residual, expected, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(state.pre_mix, ffn_pre, atol=1e-6, rtol=1e-6)
    # Using this block's own attn `pre` for the attention input would be V4's
    # unshifted mHC; check the two really differ.
    unshifted = torch.sum(attn_pre.unsqueeze(-1) * residual, dim=-2)
    assert not torch.allclose(
        unshifted, torch.sum(incoming.unsqueeze(-1) * residual, dim=-2)
    )


def test_the_first_block_is_bootstrapped_one_hot_on_stream_zero(tiny_model):
    """And the final collapse uses the last block's ffn `pre`, not an hc_head."""
    model = tiny_model.model
    assert not hasattr(model, "hc_head_fn")
    residual = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    pre = torch.zeros(2, 4)
    pre[:, 0] = 1.0
    collapsed = DeepseekV41Block.hc_apply_pre(residual, pre)
    torch.testing.assert_close(collapsed, residual[:, 0])


# --------------------------------------------------------------------------
# MoE gate and SwiGLU
# --------------------------------------------------------------------------


def test_gate_reference_math_and_what_the_model_hands_the_kernel(tiny_model):
    """The routing itself is an AITER kernel, so this pins the reference formula
    on one side and the arguments the model passes on the other."""
    torch.manual_seed(9)
    args = tiny_model.args
    moe = tiny_model.model.layers[0].ffn
    experts = moe.experts
    assert experts.scoring_func == "sqrtsoftplus"
    assert experts.top_k == args.n_activated_experts == 2
    assert experts.renormalize is True
    assert moe.routed_scaling_factor == args.route_scale == 1.5
    # The bias steers selection only, and it is `gate.bias` on disk.
    assert experts.e_score_correction_bias is moe.gate.e_score_correction_bias
    assert moe.gate.e_score_correction_bias.shape == (args.n_routed_experts,)
    assert not hasattr(moe.gate, "tid2eid"), "V4.1 has no hash-routed layers"

    x = torch.randn(6, args.dim)
    weight = torch.randn(args.n_routed_experts, args.dim) * 0.1
    bias = torch.randn(args.n_routed_experts)
    weights, indices = ref_gate(x, weight, bias, 2, args.route_scale)
    scores = F.softplus(F.linear(x.float(), weight.float())).sqrt()
    # Selection follows the biased score, the weights come from the raw one.
    assert torch.equal(indices, (scores + bias).topk(2, dim=-1)[1])
    torch.testing.assert_close(
        weights.sum(-1), torch.full((6,), args.route_scale), atol=1e-5, rtol=0
    )
    assert not torch.allclose(weights, (scores + bias).gather(1, indices))


def test_swiglu_limit_reaches_both_expert_paths_and_clamps_asymmetrically(tiny_model):
    moe = tiny_model.model.layers[0].ffn
    assert moe.experts.swiglu_limit == tiny_model.args.swiglu_limit == 10.0
    if moe.shared_experts is not None:
        assert moe.shared_experts.swiglu_limit == 10.0

    limit = 10.0
    gate = torch.tensor([-50.0, 0.0, 50.0])
    up = torch.tensor([-50.0, 1.0, 50.0])
    out = ref_swiglu(gate, up, limit)
    # up is clamped on both sides, gate only from above.
    torch.testing.assert_close(
        out,
        torch.stack(
            [
                F.silu(torch.tensor(-50.0)) * -limit,
                F.silu(torch.tensor(0.0)) * 1.0,
                F.silu(torch.tensor(limit)) * limit,
            ]
        ),
        atol=1e-5,
        rtol=1e-5,
    )


# --------------------------------------------------------------------------
# engram
# --------------------------------------------------------------------------


def test_engram_matches_the_reference(tiny_model):
    torch.manual_seed(10)
    args = tiny_model.args
    engram = tiny_model.model.layers[1].engram
    plainify(engram, "wkv")
    num_tokens = 4
    cols = engram.embed.weight.shape[-1]
    n_hash_cols = engram.wkv.weight.shape[1] // cols
    rows = torch.randn(num_tokens, n_hash_cols, cols)
    engram.embed = ModuleFn(lambda ids: rows)

    hash_ids = torch.zeros(num_tokens, n_hash_cols, dtype=torch.int64)
    x = torch.randn(num_tokens, args.hc_mult, args.dim)
    got = engram(x, hash_ids)

    kv = F.linear(rows.flatten(-2), engram.wkv.weight)
    key, value = kv.split([args.hc_mult * args.dim, args.dim], dim=-1)
    key = key.float().unflatten(-1, (args.hc_mult, args.dim))
    expected = ref_engram(
        x, key, value, engram.q_weight, engram.k_weight, args.dim, engram.eps
    )
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

    # A masked token passes through untouched.
    mask = torch.tensor([True, False, True, False])
    masked = engram(x, hash_ids, mask)
    torch.testing.assert_close(masked[1], x[1], atol=1e-6, rtol=0)
    torch.testing.assert_close(masked[0], got[0], atol=1e-6, rtol=0)


def _bind_synthetic_hash(model):
    """`NgramHashState` without a tokenizer: an identity token map."""
    from atom.models.deepseek_v41_engram import NgramHashState

    args = model.args
    lookup = [i % args.engram_compressed_vocab_size for i in range(args.vocab_size)]
    model.engram_hash = NgramHashState(
        model.engram_layout,
        lookup,
        args.engram_compressed_vocab_size,
        args.engram_pad_token_id,
    )


class _FakeBatch:
    """Just the three fields `_engram_hashes_paged` reads."""

    def __init__(self, slots, cu, first_pos):
        self.bs = len(slots)
        self.first_pos = first_pos
        self.attn_md = SimpleNamespace(
            state_slot_out=torch.tensor(slots, dtype=torch.int32),
            cu_seqlens_q=torch.tensor(cu, dtype=torch.int32),
        )


def test_engram_history_is_carried_per_request_slot(tiny_model):
    """Two chunks of one request must hash the same as the whole prompt, and a
    second request in another slot must not disturb it."""
    model = tiny_model.model
    _bind_synthetic_hash(model)
    ids = torch.arange(2, 12, dtype=torch.int64)
    pos = torch.arange(10)

    whole = model._engram_hashes_paged(ids, pos, _FakeBatch([5], [0, 10], [0]))

    model.engram_slots.fill_(NgramHashState.DEAD)
    head = model._engram_hashes_paged(ids[:4], pos[:4], _FakeBatch([5], [0, 4], [0]))
    # A different request in between must read and write its own row.
    model._engram_hashes_paged(ids[:3], pos[:3], _FakeBatch([2], [0, 3], [0]))
    tail = model._engram_hashes_paged(ids[4:], pos[4:], _FakeBatch([5], [0, 6], [4]))
    torch.testing.assert_close(torch.cat([head, tail]), whole)


def test_engram_history_resets_when_a_slot_is_reused(tiny_model):
    """A fresh prompt landing on a used slot starts from no history."""
    model = tiny_model.model
    _bind_synthetic_hash(model)
    ids = torch.arange(2, 8, dtype=torch.int64)
    pos = torch.arange(6)

    first = model._engram_hashes_paged(ids, pos, _FakeBatch([3], [0, 6], [0]))
    again = model._engram_hashes_paged(ids, pos, _FakeBatch([3], [0, 6], [0]))
    torch.testing.assert_close(first, again)


def test_engram_table_is_row_sharded(tiny_model):
    engram = tiny_model.model.layers[1].engram
    embed = engram.embed
    assert embed.num_rows == 4096  # world size 1 keeps the whole table
    assert embed.row_offset == 0
    assert embed.weight.dtype == torch.float8_e4m3fn
    assert embed.weight_scale.shape == (4096, 32 // 32)
    assert engram.q_weight.shape == (tiny_model.args.hc_mult, tiny_model.args.dim)


# --------------------------------------------------------------------------
# quantization
# --------------------------------------------------------------------------


@pytest.mark.skipif(
    not os.path.isdir(CHECKPOINT), reason="DeepSeek-V4.1-Flash config not present"
)
def test_quant_config_states_the_two_things_v4_gets_wrong_for_v41():
    """`quant_method` reads fp8 at the top level, but the routed experts are
    FP4, and the compressor / indexer weights ship without a `.scale`."""
    from aiter import QuantType

    from atom.config import get_hf_config
    from atom.models.deepseek_v41 import make_v41_quant_config

    quant = make_v41_quant_config(get_hf_config(CHECKPOINT))
    for name in ("layers.0.ffn.experts.0.w1", "layers.0.ffn.experts"):
        spec = quant.get_layer_quant_config(name)
        assert spec.quant_type == QuantType.per_1x32
        assert spec.quant_dtype == torch.float4_e2m1fn_x2
    for name in (
        "layers.2.attn.compressor.wkv",
        "layers.2.attn.compressor.wgate",
        "layers.2.attn.indexer.wk",
        "layers.2.attn.indexer.weights_proj",
    ):
        spec = quant.get_layer_quant_config(name)
        assert spec.quant_type == QuantType.No
        assert spec.quant_dtype == torch.bfloat16
    # The shared expert is fp8 while the routed ones are fp4, which is why it
    # must not be folded into the fused routed buffer.
    assert quant.get_layer_quant_config(
        "layers.0.ffn.shared_experts.w1"
    ).quant_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz)


def test_wo_a_is_dequantized_with_the_block_size_the_config_declares():
    """V4.1 ships a [32, 32] block scale, so V4's 128-only hook cannot be used
    as-is: the wrong block silently rescales three quarters of the matrix."""
    from atom.models.deepseek_v41 import DeepseekV41Attention

    out_features, in_features, block = 64, 128, 32
    weight = torch.full((out_features, in_features), 2.0).to(torch.float8_e4m3fn)
    # e8m0 bytes are bare biased exponents: 129 -> 2 ** 2.
    scale = torch.full(
        (out_features // block, in_features // block), 129, dtype=torch.uint8
    )

    class FakeWoA(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(weight, requires_grad=False)
            self.weight_scale = nn.Parameter(scale, requires_grad=False)

    class FakeAttention:
        wo_a = FakeWoA()
        wo_a_block = block

    DeepseekV41Attention.process_weights_after_loading(FakeAttention)
    got = FakeAttention.wo_a.weight
    assert got.dtype == torch.bfloat16
    torch.testing.assert_close(
        got, torch.full((out_features, in_features), 8.0, dtype=torch.bfloat16)
    )
    assert not hasattr(FakeAttention.wo_a, "weight_scale")
    # A second pass must not touch the now-BF16 weight.
    DeepseekV41Attention.process_weights_after_loading(FakeAttention)
    assert FakeAttention.wo_a.weight is got


def test_fp8_roundtrip_keeps_a_power_of_two_scale_per_group():
    x = torch.randn(2, 64) * 3
    out = _fp8_roundtrip(x, 32)
    assert out.shape == x.shape
    # e4m3 carries 3 mantissa bits, so a value is within 1/16 of its own size.
    torch.testing.assert_close(out, x, atol=0, rtol=0.07)
    # A group of zeros survives the amax floor rather than dividing by it.
    assert torch.count_nonzero(_fp8_roundtrip(torch.zeros(1, 32), 32)) == 0
