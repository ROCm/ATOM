# SPDX-License-Identifier: MIT
"""Kimi-K3 vocab-embedding replication, and the DSpark draft that inherits it.

All GPU-free. Nothing here constructs ``KimiLinearModel`` or
``VocabParallelEmbedding``: both need an initialized TP group (and the former
93 layers of weights). What CAN be checked without a device is the whole chain
that decides the draft's lookup —

  1. ``use_replicated_vocab_embed`` — the gate,
  2. ``KimiLinearModel.__init__`` consults it to pick the embedding class,
  3. ``ReplicatedEmbedding.forward`` really has no collective,
  4. ``share_with_target`` / the vLLM loader hand the draft the target's *object*,

— and (2)+(4) are what make (1) reach the drafting step at all.
"""

import inspect
from types import SimpleNamespace

import pytest

# The embedding kernels pull aiter/triton in at import time, which the plain CPU
# runner in .github/scripts/run_unit_tests.sh does not have — skip visibly there
# instead of erroring at collection. Nothing below touches a device.
pytest.importorskip("atom.models.kimi_k3")

from atom.model_ops.embed_head import ReplicatedEmbedding, VocabParallelEmbedding
from atom.models.deepseek_v2 import use_replicated_vocab_embed
from atom.models.kimi_k3 import KimiLinearModel
from atom.models.kimi_k3_dspark import KimiK3DSpark

_MASTER = "ATOM_REPLICATE_VOCAB_EMBED"
_K3 = "ATOM_KIMI_K3_REPLICATE_VOCAB_EMBED"


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(_MASTER, raising=False)
    monkeypatch.delenv(_K3, raising=False)


def _k3_text_config(**overrides):
    """The config ``KimiLinearModel`` sees: Kimi-K3's ``text_config``."""
    cfg = SimpleNamespace(
        model_type="kimi_linear",
        vocab_size=163840,
        hidden_size=7168,
        tie_word_embeddings=False,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ── the gate ──────────────────────────────────────────────────────────────


def test_kimi_k3_shards_until_asked_otherwise():
    # Unlike GLM-5.2, K3 does not ride the master switch: no end-to-end number
    # yet, and the table it would replicate is the largest ATOM replicates.
    assert use_replicated_vocab_embed(_k3_text_config()) is False
    # Root config too — served through the multimodal wrapper.
    assert use_replicated_vocab_embed(_k3_text_config(model_type="kimi_k3")) is False


@pytest.mark.parametrize(
    ("master", "opt_in", "expected"),
    [
        (None, None, False),  # default: sharded
        (None, "1", True),  # opted in
        (None, "0", False),  # explicitly off reads the same as unset
        ("0", "1", False),  # the master switch is still a global off
        ("1", "1", True),
    ],
)
def test_kimi_k3_opt_in(monkeypatch, master, opt_in, expected):
    if master is not None:
        monkeypatch.setenv(_MASTER, master)
    if opt_in is not None:
        monkeypatch.setenv(_K3, opt_in)
    assert use_replicated_vocab_embed(_k3_text_config()) is expected


def test_kimi_k3_opt_in_leaves_other_families_alone(monkeypatch):
    # The K3 knob exists precisely so K3 can be moved without moving GLM-5.2
    # (validated) off the replicated path, or a plain DeepSeek onto it.
    monkeypatch.setenv(_K3, "1")
    glm = SimpleNamespace(model_type="glm_moe_dsa", tie_word_embeddings=False)
    ds = SimpleNamespace(model_type="deepseek_v3", tie_word_embeddings=False)
    assert use_replicated_vocab_embed(glm) is True
    assert use_replicated_vocab_embed(ds) is False


def test_tied_embedding_is_never_replicated(monkeypatch):
    # A tied embedding IS the sharded lm_head's weight; replicating it would
    # un-shard the head. The safety guard must outrank both switches.
    monkeypatch.setenv(_K3, "1")
    cfg = _k3_text_config(tie_word_embeddings=True)
    assert use_replicated_vocab_embed(cfg) is False


# ── the wiring ────────────────────────────────────────────────────────────


def _class_body_init(cls):
    """The ``__init__`` the class actually declares.

    ``support_torch_compile`` swaps in a wrapper that keeps the original only in
    its closure -- no ``functools.wraps``, so ``inspect.unwrap`` cannot see past
    it.
    """
    init = cls.__init__
    for cell in init.__closure__ or ():
        inner = cell.cell_contents
        if inspect.isfunction(inner) and inner.__name__ == "__init__":
            return inner
    return init


def test_kimi_linear_model_picks_its_embedding_from_the_gate():
    names = _class_body_init(KimiLinearModel).__code__.co_names
    assert "use_replicated_vocab_embed" in names
    assert "ReplicatedEmbedding" in names
    assert "VocabParallelEmbedding" in names


def test_replicated_lookup_has_no_collective_and_sharded_one_does():
    replicated = ReplicatedEmbedding.forward.__code__.co_names
    assert "replicated_embedding" in replicated
    assert "all_reduce" not in replicated
    # mark_trace wraps with functools.wraps, so unwrap to reach the real body.
    sharded = inspect.unwrap(VocabParallelEmbedding.forward).__code__.co_names
    assert "masked_embedding" in sharded
    assert "all_reduce" in sharded


# ── the draft inherits it ─────────────────────────────────────────────────


def _fake_target(embed):
    return SimpleNamespace(model=SimpleNamespace(embed_tokens=embed), lm_head=object())


def _fake_draft(vocab_size=163840):
    # share_with_target only reads self.hf_config and assigns two attributes,
    # so a stub self exercises the real method without building the draft.
    return SimpleNamespace(hf_config=SimpleNamespace(vocab_size=vocab_size))


def test_draft_inherits_a_replicated_target_embedding():
    # A real ReplicatedEmbedding (small dims): it needs no TP group, and this
    # also pins that it exposes the `num_embeddings` the vocab check reads.
    embed = ReplicatedEmbedding(163840, 8)
    target = _fake_target(embed)
    draft = _fake_draft()
    KimiK3DSpark.share_with_target(draft, target, set())
    assert draft.embed_tokens is embed
    assert draft.lm_head is target.lm_head


def test_draft_inherits_a_sharded_target_embedding():
    # Same binding with the fallback (ATOM_REPLICATE_VOCAB_EMBED=0) shape: the
    # draft gets the sharded module, collective and all. Stubbed rather than a
    # real VocabParallelEmbedding, which needs an initialized TP group.
    embed = SimpleNamespace(num_embeddings=163840)
    draft = _fake_draft()
    KimiK3DSpark.share_with_target(draft, _fake_target(embed), set())
    assert draft.embed_tokens is embed


def test_draft_never_loads_its_own_embedding():
    # The checkpoint ships an embed_tokens that is a fine-tuned derivative of
    # the target's, not a copy — loading it would change what the draft embeds.
    assert "embed_tokens." in KimiK3DSpark.skip_weight_prefixes


def test_draft_rejects_a_target_with_a_different_vocab():
    embed = ReplicatedEmbedding(1024, 8)
    with pytest.raises(ValueError, match="vocab"):
        KimiK3DSpark.share_with_target(_fake_draft(163840), _fake_target(embed), set())


def test_vllm_loader_rebinds_the_draft_to_the_targets_embedding():
    pytest.importorskip("vllm")
    from vllm.v1.worker.gpu.spec_decode.eagle.utils import _should_share

    from atom.plugin.vllm.models.kimi_k3_dspark import KimiK3DSparkVllm

    # load_dspark_model does `_should_share(draft, "has_own_embed_tokens", ...)`
    # and, when true, `draft.model.embed_tokens = target_inner.embed_tokens`.
    # has_own_embed_tokens=False short-circuits it before any weight compare,
    # so the draft always ends up on the target's module — replicated or not.
    assert KimiK3DSparkVllm.has_own_embed_tokens is False
    wrapper = SimpleNamespace(
        has_own_embed_tokens=KimiK3DSparkVllm.has_own_embed_tokens
    )
    assert _should_share(wrapper, "has_own_embed_tokens", None, object()) is True
