"""The lossless ratio rule and relaxed MTP must be refused together, at startup.

Relaxed top-N acceptance (ATOM_ENABLE_RELAXED_MTP, #411) deliberately accepts
tokens the target would not have drawn; the ratio rule's whole claim is that
the emitted stream carries the target's distribution exactly. There is no
coherent composition, so the pair is a contradictory request rather than a
capability gap -- hence a raise, where PP and PCP warn and fall back.

Regression test for WHERE the refusal lives. It used to sit in
`rejection_sample`, which only sees `draft_probs` on a sampled unfiltered
batch: an all-greedy batch silently ran relaxed, a filtered one silently ran
exact match, and only the third case raised -- mid-stream, inside every TP
worker, long after the server had booted and armed.
"""

import pytest

from atom.spec_decode.dspark_proposer import DSparkProposer


class _Model:
    supports_lossless_draft = True
    vocab_size = 32


class _Config:
    pipeline_parallel_size = 1
    max_num_seqs = 8


def _proposer():
    """A proposer assembled without __init__, the shape the metadata tests use."""
    p = DSparkProposer.__new__(DSparkProposer)
    p.model = _Model()
    p.config = _Config()
    return p


def _init(monkeypatch, *, lossless, relaxed):
    from atom.utils import envs

    monkeypatch.setattr(envs, "ATOM_SPEC_LOSSLESS_SAMPLING", lossless, raising=False)
    monkeypatch.setattr(envs, "ATOM_ENABLE_RELAXED_MTP", relaxed, raising=False)
    return _proposer()


def test_both_flags_on_raises_at_construction(monkeypatch):
    p = _init(monkeypatch, lossless=True, relaxed=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        p._init_lossless_sampling()


def test_the_error_names_both_flags(monkeypatch):
    # An operator reading only the message must be able to act on it.
    p = _init(monkeypatch, lossless=True, relaxed=True)
    with pytest.raises(ValueError) as e:
        p._init_lossless_sampling()
    msg = str(e.value)
    assert "ATOM_SPEC_LOSSLESS_SAMPLING" in msg
    assert "ATOM_ENABLE_RELAXED_MTP" in msg


def test_relaxed_alone_is_untouched(monkeypatch):
    # Lossless off: the gate returns before the exclusion is ever consulted, so
    # relaxed MTP keeps working exactly as #411 shipped it.
    p = _init(monkeypatch, lossless=False, relaxed=True)
    p._init_lossless_sampling()
    assert p.draft_sampling is None


def test_refusal_precedes_the_sampler(monkeypatch):
    # The point of the fix: the pair is rejected before any verify step runs,
    # not on the hot path. Reaching rejection_sample at all would mean the
    # construction gate was bypassed, which is what the assert there now says.
    import inspect

    from atom.model_ops import rejection_sampler

    src = inspect.getsource(rejection_sampler.rejection_sample)
    assert "RELAXED_TOP_N <= 1" in src, "hot-path check should be an assert"
    assert "raise NotImplementedError" not in src
