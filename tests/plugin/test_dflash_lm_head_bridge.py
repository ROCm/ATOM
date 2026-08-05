"""Unit tests for the DFLASH draft-sampling lm_head monkey patch.

The patch replaces SGLang's ``DFlashWorkerV2._greedy_sample_from_vocab_parallel_head``
so an external (ATOM) vocab-parallel head performs its own TP reduction. These
tests use a stand-in worker class rather than importing SGLang, so they run on
CPU and do not depend on a SGLang install.
"""

from __future__ import annotations

import sys
import types

import pytest
import torch

from atom.plugin.sglang.dflash_lm_head_bridge import (
    _PATCH_FLAG,
    install_dflash_lm_head_patch,
)

VOCAB = 32
HIDDEN = 8
TP_SIZE = 4
LOCAL_VOCAB = VOCAB // TP_SIZE


class _StockWorker:
    """Stands in for SGLang's DFlashWorkerV2.

    ``_greedy_sample_from_vocab_parallel_head`` reproduces the upstream
    behaviour that breaks on ATOM's head: with no ``shard_indices`` it treats
    the per-rank local argmax index as a global token id.
    """

    def _greedy_sample_from_vocab_parallel_head(
        self, *, hidden_states, lm_head, chunk_size: int = 256
    ):
        logits = torch.matmul(hidden_states, lm_head.weight.T)
        return torch.argmax(logits, dim=-1).to(torch.long)


class _AtomHead:
    """Stands in for ATOM's ParallelLMHead on TP rank ``tp_rank``."""

    def __init__(self, full_weight: torch.Tensor, tp_rank: int):
        self.vocab_start_idx = tp_rank * LOCAL_VOCAB
        self.weight = full_weight[
            self.vocab_start_idx : self.vocab_start_idx + LOCAL_VOCAB
        ]
        self._full_weight = full_weight
        self.calls = 0

    def compute_argmax_token(self, x: torch.Tensor) -> torch.Tensor:
        # Emulate the all-gathered global reduction ATOM performs across ranks.
        self.calls += 1
        return torch.argmax(torch.matmul(x, self._full_weight.T), dim=-1).to(torch.long)


class _PlainHead:
    """A head without ``compute_argmax_token``; must keep the stock path."""

    def __init__(self, weight: torch.Tensor):
        self.weight = weight


@pytest.fixture
def dflash_module(monkeypatch):
    """Install a fake ``sglang.srt.speculative.dflash_worker_v2`` module."""
    # Put the method in the class's own __dict__ (not inherited) so a test can
    # delete it to simulate SGLang renaming the patch target.
    worker_cls = type(
        "DFlashWorkerV2",
        (),
        {
            "_greedy_sample_from_vocab_parallel_head": (
                _StockWorker._greedy_sample_from_vocab_parallel_head
            )
        },
    )
    module = types.ModuleType("sglang.srt.speculative.dflash_worker_v2")
    module.DFlashWorkerV2 = worker_cls
    for name in (
        "sglang",
        "sglang.srt",
        "sglang.srt.speculative",
        "sglang.srt.speculative.dflash_worker_v2",
    ):
        monkeypatch.setitem(
            sys.modules,
            name,
            module if name.endswith("dflash_worker_v2") else types.ModuleType(name),
        )
    return module


def _inputs(seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    weight = torch.randn(VOCAB, HIDDEN, generator=generator)
    hidden = torch.randn(6, HIDDEN, generator=generator)
    return weight, hidden


def test_patch_routes_external_head_and_fixes_tp_token_ids(dflash_module):
    """Without the patch a sharded external head yields local indices; with it,
    every rank returns the same correct global token ids."""
    weight, hidden = _inputs()
    expected = torch.argmax(torch.matmul(hidden, weight.T), dim=-1).to(torch.long)
    worker = dflash_module.DFlashWorkerV2()

    stock = worker._greedy_sample_from_vocab_parallel_head(
        hidden_states=hidden, lm_head=_AtomHead(weight, tp_rank=1)
    )
    assert not torch.equal(stock, expected), (
        "the stand-in must reproduce upstream's broken behaviour, "
        "otherwise this test proves nothing"
    )

    install_dflash_lm_head_patch()

    for tp_rank in range(TP_SIZE):
        head = _AtomHead(weight, tp_rank=tp_rank)
        got = worker._greedy_sample_from_vocab_parallel_head(
            hidden_states=hidden, lm_head=head
        )
        torch.testing.assert_close(got, expected, rtol=0, atol=0)
        assert head.calls > 0, "the external head reduction must be used"


def test_patch_falls_back_for_heads_without_the_hook(dflash_module):
    """A head with no compute_argmax_token must keep SGLang's own path."""
    weight, hidden = _inputs(seed=7)
    worker = dflash_module.DFlashWorkerV2()
    before = worker._greedy_sample_from_vocab_parallel_head(
        hidden_states=hidden, lm_head=_PlainHead(weight)
    )

    install_dflash_lm_head_patch()

    after = worker._greedy_sample_from_vocab_parallel_head(
        hidden_states=hidden, lm_head=_PlainHead(weight)
    )
    torch.testing.assert_close(after, before, rtol=0, atol=0)


def test_patch_is_idempotent(dflash_module):
    install_dflash_lm_head_patch()
    first = dflash_module.DFlashWorkerV2._greedy_sample_from_vocab_parallel_head
    install_dflash_lm_head_patch()
    second = dflash_module.DFlashWorkerV2._greedy_sample_from_vocab_parallel_head
    assert first is second
    assert getattr(dflash_module.DFlashWorkerV2, _PATCH_FLAG) is True


def test_patch_respects_chunking(dflash_module):
    """Chunked calls must stitch back into the same result as one call."""
    weight, hidden = _inputs(seed=3)
    expected = torch.argmax(torch.matmul(hidden, weight.T), dim=-1).to(torch.long)
    install_dflash_lm_head_patch()
    worker = dflash_module.DFlashWorkerV2()
    head = _AtomHead(weight, tp_rank=2)

    got = worker._greedy_sample_from_vocab_parallel_head(
        hidden_states=hidden, lm_head=head, chunk_size=2
    )
    torch.testing.assert_close(got, expected, rtol=0, atol=0)
    assert head.calls == 3, f"expected 3 chunks of 2, got {head.calls} calls"


def test_patch_handles_empty_input(dflash_module):
    weight, _ = _inputs(seed=5)
    install_dflash_lm_head_patch()
    worker = dflash_module.DFlashWorkerV2()
    out = worker._greedy_sample_from_vocab_parallel_head(
        hidden_states=torch.empty(0, HIDDEN), lm_head=_AtomHead(weight, tp_rank=0)
    )
    assert out.shape == (0,)
    assert out.dtype == torch.long


def test_patch_rejects_bad_shape_from_head(dflash_module):
    """A head returning the wrong shape must fail loudly, not silently corrupt
    the draft block."""
    weight, hidden = _inputs(seed=11)

    class _BadHead(_AtomHead):
        def compute_argmax_token(self, x):
            return torch.zeros(x.shape[0] + 1, dtype=torch.long)

    install_dflash_lm_head_patch()
    worker = dflash_module.DFlashWorkerV2()
    with pytest.raises(ValueError, match="invalid shape"):
        worker._greedy_sample_from_vocab_parallel_head(
            hidden_states=hidden, lm_head=_BadHead(weight, tp_rank=0)
        )


def test_install_is_noop_without_sglang_dflash(monkeypatch):
    """Older SGLang releases have no DFLASH worker; installing must not raise."""
    for name in ("sglang", "sglang.srt", "sglang.srt.speculative"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    monkeypatch.setitem(sys.modules, "sglang.srt.speculative.dflash_worker_v2", None)
    install_dflash_lm_head_patch()


def test_install_warns_when_target_method_is_gone(dflash_module, caplog):
    """If SGLang renames the method the patch must warn, not fail silently."""
    del dflash_module.DFlashWorkerV2._greedy_sample_from_vocab_parallel_head
    with caplog.at_level("WARNING"):
        install_dflash_lm_head_patch()
    assert "DFLASH" in caplog.text
    assert not getattr(dflash_module.DFlashWorkerV2, _PATCH_FLAG, False)
