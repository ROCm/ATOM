# SPDX-License-Identifier: MIT
"""Each pipeline stage builds only the embedding, final norm and LM head it
uses, and loads only the checkpoint tensors of what it builds.

The first stage embeds tokens; the last applies the final norm and the LM
head. A tied head shares the embedding's Parameter, so the last stage also
builds the embedding when the head is tied to it. Decoder layers are replaced
by a one-parameter stand-in: they are partitioned by `make_layers`, which
`test_pp.py` covers.
"""

from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter
from torch import nn

from atom.config import CompilationLevel

VOCAB, HIDDEN, LAYERS = 32, 8, 4

# (module, causal-LM class, decoder-layer class the module builds layers from)
MODELS = {
    "gpt_oss": ("atom.models.gpt_oss", "GptOssForCausalLM", "TransformerBlock"),
    "mixtral": ("atom.models.mixtral", "MixtralForCausalLM", "MixtralDecoderLayer"),
    "qwen3_next": (
        "atom.models.qwen3_next",
        "Qwen3NextForCausalLM",
        "Qwen3NextDecoderLayer",
    ),
    "qwen3_5": ("atom.models.qwen3_5", "Qwen3_5MoeForCausalLM", "Qwen3_5DecoderLayer"),
    "qwen3_5_dense": (
        "atom.models.qwen3_5",
        "Qwen3_5ForCausalLM",
        "Qwen3_5DecoderLayer",
    ),
}
# gpt-oss never ties its head.
TIED = [(m, False) for m in MODELS] + [
    (m, True) for m in ("mixtral", "qwen3_next", "qwen3_5_dense")
]


class _Layer(nn.Module):
    """A decoder layer with one checkpoint tensor, `input_layernorm.weight`."""

    def __init__(self, *_args, **_kwargs):
        super().__init__()
        self.input_layernorm = nn.Module()
        self.input_layernorm.weight = nn.Parameter(torch.zeros(HIDDEN))


def _hf_config(tied: bool) -> SimpleNamespace:
    return SimpleNamespace(
        architectures=["Test"],
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        num_hidden_layers=LAYERS,
        rms_norm_eps=1e-6,
        tie_word_embeddings=tied,
        num_local_experts=2,
        num_experts=2,
        layer_types=["full_attention"] * LAYERS,
    )


def _atom_config(tied: bool) -> SimpleNamespace:
    return SimpleNamespace(
        hf_config=_hf_config(tied),
        quant_config=SimpleNamespace(
            global_quant_config=SimpleNamespace(quant_dtype=torch.bfloat16)
        ),
        kv_cache_dtype="bf16",
        compilation_config=SimpleNamespace(level=CompilationLevel.NO_COMPILATION),
    )


def _checkpoint(tied: bool) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    ckpt = {
        "model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN, generator=gen),
        "model.norm.weight": torch.randn(HIDDEN, generator=gen),
    }
    if not tied:
        ckpt["lm_head.weight"] = torch.randn(VOCAB, HIDDEN, generator=gen)
    for i in range(LAYERS):
        ckpt[f"model.layers.{i}.input_layernorm.weight"] = torch.randn(
            HIDDEN, generator=gen
        )
    return ckpt


def _stage(monkeypatch, load, model: str, rank: int, size: int):
    """Import `model`'s module with one PP stage's view of the world."""
    group = SimpleNamespace(
        rank_in_group=rank,
        world_size=size,
        is_first_rank=rank == 0,
        is_last_rank=rank == size - 1,
    )
    tp = SimpleNamespace(rank_in_group=0, world_size=1)
    module_name, _, layer_cls = MODELS[model]
    mod = load(module_name)
    monkeypatch.setattr(mod, "get_pp_group", lambda: group)
    monkeypatch.setattr(mod, layer_cls, _Layer)
    # `make_layers` imports it at call time.
    monkeypatch.setattr(
        load("aiter.dist.parallel_state"), "get_pp_group", lambda: group, raising=False
    )
    monkeypatch.setattr(load("atom.model_ops.embed_head"), "get_tp_group", lambda: tp)
    for name in ("atom.model_ops.layernorm", module_name):
        monkeypatch.setattr(
            load(name), "get_tensor_model_parallel_world_size", lambda: 1
        )
    return mod


def _build(monkeypatch, load, model: str, rank: int, size: int, tied: bool):
    mod = _stage(monkeypatch, load, model, rank, size)
    return getattr(mod, MODELS[model][1])(_atom_config(tied))


def _load(load, model: nn.Module, tied: bool) -> tuple[set[str], set[str]]:
    """Load the checkpoint; return (names read, parameter names loaded)."""
    loading_core = load("atom.model_loader.loading_core")
    ckpt = _checkpoint(tied)
    read: list[str] = []

    def iterator(_path, _disable_mmap, wants=None):
        for name, tensor in ckpt.items():
            if wants is None or wants(name):
                read.append(name)
                yield name, tensor

    loaded = loading_core.load_weights_into_model(
        model=model,
        model_name_or_path="<synthetic>",
        hf_config=_hf_config(tied),
        default_weight_loader=lambda param, weight: param.data.copy_(weight),
        fuse_shared_expert=lambda *_a, **_k: False,
        is_rank0=lambda: True,
        weights_iterator=iterator,
    )
    for name, param in model.named_parameters():
        torch.testing.assert_close(param.data, ckpt[_ckpt_name(name, tied)])
    return set(read), loaded


def _ckpt_name(param_name: str, tied: bool) -> str:
    if tied and param_name == "lm_head.weight":
        return "model.embed_tokens.weight"
    return param_name


@pytest.mark.parametrize("model,tied", TIED)
@pytest.mark.parametrize("rank,size", [(0, 1), (0, 2), (1, 2)])
def test_stage_builds_and_loads_only_what_it_uses(monkeypatch, model, tied, rank, size):
    with atom_without_aiter() as load:
        missing = load("atom.models.utils").PPMissingLayer
        head = load("atom.model_ops.embed_head").ParallelLMHead
        lm = _build(monkeypatch, load, model, rank, size, tied)
        first, last = rank == 0, rank == size - 1

        holds_embed = first or (tied and last)
        assert isinstance(lm.model.embed_tokens, missing) is not holds_embed
        assert isinstance(lm.model.norm, missing) is not last
        assert isinstance(lm.lm_head, head if last else missing)
        if tied and last:
            assert lm.lm_head.weight is lm.model.embed_tokens.weight

        read, loaded = _load(load, lm, tied)

    start, end = lm.model.start_layer, lm.model.end_layer
    expected = {f"model.layers.{i}.input_layernorm.weight" for i in range(start, end)}
    if holds_embed:
        expected.add("model.embed_tokens.weight")
    if last:
        expected.add("model.norm.weight")
        if not tied:
            expected.add("lm_head.weight")
    # Another stage's tensors are not even read, and every parameter loads.
    assert read == expected
    assert loaded == set(dict(lm.named_parameters())) == expected


@pytest.mark.parametrize("rank", [0, 1])
def test_qwen3_5_wrapper_embeds_only_on_the_first_stage(monkeypatch, rank):
    with atom_without_aiter() as load:
        mod = _stage(monkeypatch, load, "qwen3_5", rank, 2)
        wrapper = mod.Qwen3_5ForConditionalGenerationTextOnly(_atom_config(False))
        seen = {}

        def language_model(input_ids, positions, intermediate_tensors, embeds):
            seen["inputs_embeds"] = embeds

        wrapper.language_model.forward = language_model
        wrapper(torch.arange(3), torch.arange(3))

    if rank == 0:
        assert seen["inputs_embeds"].shape == (3, HIDDEN)
    else:
        assert seen["inputs_embeds"] is None
