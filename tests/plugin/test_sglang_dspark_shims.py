"""CPU tests for the SGLang DSpark shims on the ATOM wrapper.

The head view and the two lookup methods only translate ATOM's embedding and
vocabulary head into the objects SGLang's draft reads. They do not run mHC.
"""

import torch
from torch import nn

from atom.plugin.sglang.models.base_model_wrapper import (
    _AtomCausalLMBaseForSglang,
    _AtomDSparkLmHeadView,
    _VocabShardIndices,
)


class _Head(nn.Module):
    def __init__(self, per_partition: int = 4) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4))
        self.num_embeddings = 8
        self.num_embeddings_per_partition = per_partition
        self.tp_size = 2
        self.vocab_start_idx = 4
        self.vocab_end_idx = 8


class _Parts:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)


def _wrapper(**attrs):
    """Build a wrapper without running the model-loading constructor."""
    wrapper = _AtomCausalLMBaseForSglang.__new__(_AtomCausalLMBaseForSglang)
    wrapper.__dict__.update(attrs)
    return wrapper


def _v4_wrapper(embed, head):
    return _wrapper(
        model_arch="DeepseekV4ForCausalLM",
        model=_Parts(model=_Parts(embed=embed, head=head)),
    )


def test_vocab_shard_indices_use_sglang_field_names():
    shard = _VocabShardIndices(4, 8)

    assert shard.org_vocab_start_index == 4
    assert shard.org_vocab_end_index == 8


def test_lm_head_view_exposes_partition_metadata_and_same_weight():
    head = _Head()
    view = _AtomDSparkLmHeadView(head)

    assert view.org_vocab_size == 8
    assert view.tp_size == 2
    assert view.num_embeddings_per_partition == 4
    assert view.num_embeddings_padded == 8
    assert view.use_attn_tp_group is False
    assert view.quant_method is None
    assert view.shard_indices.org_vocab_start_index == 4
    assert view.shard_indices.org_vocab_end_index == 8
    assert view.weight is head.weight
    assert "forward" not in _AtomDSparkLmHeadView.__dict__


def test_lm_head_view_rejects_a_partition_that_does_not_cover_vocab():
    head = _Head(per_partition=3)

    try:
        _AtomDSparkLmHeadView(head)
    except RuntimeError as exc:
        assert "does not cover the vocabulary" in str(exc)
    else:
        raise AssertionError("incomplete partition was accepted")


def test_get_dspark_shared_modules_returns_embed_and_head_view():
    embed = nn.Embedding(8, 4)
    head = _Head()
    wrapper = _v4_wrapper(embed, head)

    got_embed, got_head = _AtomCausalLMBaseForSglang.get_dspark_shared_modules(wrapper)

    assert got_embed is embed
    assert isinstance(got_head, _AtomDSparkLmHeadView)
    assert got_head.weight is head.weight


def test_get_dspark_shared_modules_rejects_a_non_v4_model():
    wrapper = _wrapper(model_arch="Qwen3NextForCausalLM", model=None)

    try:
        _AtomCausalLMBaseForSglang.get_dspark_shared_modules(wrapper)
    except RuntimeError as exc:
        assert "DeepseekV4ForCausalLM" in str(exc)
    else:
        raise AssertionError("non-V4 model was accepted")


def test_get_input_embeddings_returns_v4_module_or_embedded_rows():
    embed = nn.Embedding(8, 4)
    wrapper = _v4_wrapper(embed, _Head())
    ids = torch.tensor([1, 3])

    assert _AtomCausalLMBaseForSglang.get_input_embeddings(wrapper) is embed
    rows = _AtomCausalLMBaseForSglang.get_input_embeddings(wrapper, ids)
    assert torch.equal(rows, embed(ids))


def test_get_input_embeddings_forwards_ids_for_other_architectures():
    seen = {}

    class _Model(nn.Module):
        def get_input_embeddings(self, input_ids):
            seen["ids"] = input_ids
            return input_ids + 1

    wrapper = _wrapper(model_arch="Qwen3NextForCausalLM", model=_Model())
    ids = torch.tensor([2, 4])

    out = _AtomCausalLMBaseForSglang.get_input_embeddings(wrapper, ids)

    assert torch.equal(seen["ids"], ids)
    assert torch.equal(out, ids + 1)


def test_get_input_embeddings_without_ids_returns_the_other_models_module():
    tokens = nn.Embedding(4, 2)
    wrapper = _wrapper(
        model_arch="Qwen3NextForCausalLM",
        model=_Parts(model=_Parts(embed_tokens=tokens)),
    )

    assert _AtomCausalLMBaseForSglang.get_input_embeddings(wrapper) is tokens
