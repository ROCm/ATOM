# SPDX-License-Identifier: MIT
"""Which attention a config asks for.

Four predicates on ModelRunner used to answer this, keyed on `model_type`, and
a chain of ifs in the selector re-composed their four booleans into one answer
-- so the priority between them lived in the order of that chain and nowhere
else. It is `attn_family` now, and what is pinned here is the two things that
made the old spelling wrong rather than merely long:

  - An EAGLE draft is not an attention. `eagle` is a decoding algorithm, and
    two drafts of one target need not share a stack: the deployed
    MiniMax-M3 draft declares `model_type=llama`, and someone else's would
    declare an MLA base. The old predicate branched on the algorithm and then
    hard-coded two model types for what was underneath it.
  - MLA is a latent rank, not a list of names. The list it replaces named six
    model types and missed `k3_dspark`, whose config carries
    `kv_lora_rank=512`.

Runs where there is no GPU build: the selector imports its return annotation
under TYPE_CHECKING for exactly that reason.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from atom.utils.selector import Family, attn_family, has_mla_indexer

GDN, KIMI_MLA, MHA, MLA, V4 = (
    Family.GDN,
    Family.KIMI_MLA,
    Family.MHA,
    Family.MLA,
    Family.V4,
)


def cfg(**fields) -> SimpleNamespace:
    """A config as the selector reads one: attributes, absent when unset."""
    return SimpleNamespace(**fields)


class TestTheFamilyIsReadOffTheShape:

    def test_a_latent_rank_is_what_makes_a_model_mla(self):
        """Not the model's name. `k3_dspark` is the config that proves it: the
        list this replaces did not name it, and it carries a rank of 512."""
        assert attn_family(cfg(model_type="k3_dspark", kv_lora_rank=512)) == MLA

    def test_no_latent_rank_is_mha(self):
        assert attn_family(cfg(model_type="llama")) == MHA

    def test_a_config_that_says_nothing_is_mha(self):
        """A `model_type` is not guaranteed -- MiniMax-M3's text config has
        none -- and the answer for one that declares nothing is the default."""
        assert attn_family(cfg()) == MHA


class TestTheAlgorithmIsNotTheAttention:
    """An EAGLE draft declares the stack it was trained on, and the family
    follows that. The predicate this replaces branched on `model_type ==
    "eagle"` and then read two hard-coded model types off the inner config."""

    def test_the_deployed_eagle3_draft_is_mha(self):
        """MiniMax-M3's draft, verbatim: an EAGLE3 head over a llama stack."""
        draft = cfg(model_type="llama", architectures=["LlamaForCausalLMEagle3"])

        assert attn_family(draft) == MHA

    def test_an_eagle_draft_over_an_mla_stack_is_mla(self):
        """Same algorithm, other stack. Nothing about the family may depend on
        the two being trained together."""
        draft = cfg(model_type="deepseek_v3", kv_lora_rank=512)

        assert attn_family(draft) == MLA

    def test_a_config_that_says_eagle_is_read_for_its_rank(self):
        """`get_hf_text_config` unwraps `text_config` and nothing else, so a
        checkpoint whose `model_type` IS the algorithm arrives as it is. The
        predicate this replaces branched on that name and then accepted only
        `deepseek_v2`/`deepseek_v3` underneath -- an EAGLE trained on any other
        MLA base read as MHA, and would have been given a split K/V pool."""
        draft = cfg(
            model_type="eagle",
            kv_lora_rank=512,
            model=cfg(model_type="kimi_k2"),
        )

        assert attn_family(draft) == MLA

    def test_an_eagle_config_without_a_rank_is_mha(self):
        assert (
            attn_family(cfg(model_type="eagle", model=cfg(model_type="llama"))) == MHA
        )


class TestThePriorityBetweenFamilies:
    """Three of the five overlap on the evidence, so the order is the answer
    and it is one function's to give."""

    def test_v4_wins_over_its_latent_rank(self):
        """DeepSeek-V4 reads as `deepseek_v3` -- the schema it reuses -- and
        carries a rank, so only the architecture tells them apart."""
        v4 = cfg(
            model_type="deepseek_v3",
            kv_lora_rank=512,
            architectures=["DeepseekV4ForCausalLM"],
        )

        assert attn_family(v4) == V4

    def test_a_v4_draft_is_v4_without_an_architecture_of_its_own(self):
        assert attn_family(cfg(model_type="deepseek_v4_mtp")) == V4

    def test_a_hybrid_wins_over_its_latent_rank(self):
        """Kimi-Linear has both a rank and linear layers; what makes it its own
        family is the second, which no single field states."""
        assert attn_family(cfg(model_type="kimi_linear", kv_lora_rank=512)) == KIMI_MLA

    @pytest.mark.parametrize(
        "model_type",
        ["qwen3_next", "qwen3_next_mtp", "qwen3_5_text", "qwen3_5_moe_text"],
    )
    def test_the_gdn_hybrids(self, model_type):
        assert attn_family(cfg(model_type=model_type)) == GDN


class TestWhatRidesAnMlaPool:

    @pytest.mark.parametrize(
        "family, expected",
        [(MLA, True), (KIMI_MLA, True), (V4, False), (GDN, False), (MHA, False)],
    )
    def test_both_mla_families_answer_together(self, family, expected):
        """Two questions spelled this as `use_mla or use_kimi_mla`."""
        assert family.is_mla is expected

    def test_an_indexer_needs_mla_rows_to_ride(self):
        """`index_topk` alone is not enough: the cache is a field of the MLA
        pool, so a model without one has nowhere to put it."""
        assert has_mla_indexer(
            cfg(model_type="deepseek_v3", kv_lora_rank=512, index_topk=64)
        )
        assert not has_mla_indexer(cfg(model_type="llama", index_topk=64))

    def test_a_sparse_hybrid_counts(self):
        """GLM-5.3-Flash is a hybrid AND sparse -- asking only about the pure
        MLA family would leave its index cache unbound."""
        assert has_mla_indexer(cfg(model_type="glm5_next_text", index_topk=64))

    def test_an_mla_model_without_an_indexer_does_not(self):
        assert not has_mla_indexer(cfg(model_type="deepseek_v3", kv_lora_rank=512))
