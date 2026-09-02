# SPDX-License-Identifier: MIT
"""The Eagle3 flash draft must not read last batch's block_tables.

``propose`` takes ``forward_vars["block_tables"].gpu`` unconditionally, but the
target's builders upload only under their own conditions. When none of them
fired this step the buffer is not None: its GPU copy still holds the previous
batch's pages. Those indices are in bounds, so the draft attends over the wrong
KV, nothing faults, and the only symptom is a lower acceptance rate.

Freshness is tracked on the GPU copy, not on the CPU pack: `prepare_block_tables`
only fills `.np`, and not every caller of it goes on to upload (`gdn_attn.py`
packs and never uploads), while `deepseek_v4_attn` uploads without calling it.
All nine upload sites do go through ``CpuGpuBuffer.copy_to_gpu``, so that is
where the flag is set and ``CommonAttentionBuilder.build`` is where it is
cleared.

The guard covers flash drafts only. A SHUFFLE draft on a prefill with no cached
prefix takes the raw-varlen path and never reads block_tables, so guarding it
would fail a config that is fine -- `test_guard_ignores_a_shuffle_draft` pins
that.

No GPU: every method is called unbound against a namespace.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

backends = pytest.importorskip(
    "atom.model_ops.attentions.backends",
    reason="requires the full atom import environment (aiter)",
)
eagle_proposer = pytest.importorskip(
    "atom.spec_decode.eagle_proposer",
    reason="requires the full atom import environment (aiter)",
)
atom_utils = pytest.importorskip("atom.utils")

_guard = eagle_proposer.EagleProposer._assert_block_tables_are_this_steps
_build = backends.CommonAttentionBuilder.build
_pack = backends.CommonAttentionBuilder.prepare_block_tables
_copy_to_gpu = atom_utils.CpuGpuBuffer.copy_to_gpu


def _buffer():
    """A block_tables buffer: the two tensors copy_to_gpu touches, no device."""
    return SimpleNamespace(
        np=np.zeros((4, 8), dtype=np.int32),
        cpu=torch.zeros(4, 8, dtype=torch.int32),
        gpu=torch.zeros(4, 8, dtype=torch.int32),
    )


def _runner(buf=None):
    return SimpleNamespace(forward_vars={"block_tables": buf or _buffer()})


def _batch(prefill=True):
    return SimpleNamespace(
        block_tables=[np.array([1, 2, 3], dtype=np.int32)],
        state_maintenance_ops=SimpleNamespace(
            relocations=(), checkpoint_stores=(), checkpoint_restores=()
        ),
        total_tokens_num_prefill=1 if prefill else 0,
    )


def _builder(runner):
    return SimpleNamespace(
        model_runner=runner,
        prepare_prefill=lambda b, r: ("prefill", None),
        prepare_decode=lambda b, r, t, m: ("decode", None),
    )


def _proposer(runner, flash=True):
    return SimpleNamespace(
        runner=SimpleNamespace(
            forward_vars=runner.forward_vars,
            eagle3_draft_builder=SimpleNamespace(uses_flash_layout=flash),
        )
    )


def _ctx(dummy=False):
    return SimpleNamespace(is_dummy_run=dummy)


# --- the two ends of the flag ------------------------------------------------


@pytest.mark.parametrize("prefill", [True, False], ids=["prefill", "decode"])
def test_build_clears_the_flag(prefill):
    r = _runner()
    r.forward_vars["block_tables"].gpu_is_current = True
    _build(_builder(r), _batch(prefill), 1, 1, 1)
    assert r.forward_vars["block_tables"].gpu_is_current is False


def test_copy_to_gpu_sets_the_flag():
    """The upload, not the CPU pack, is what a `.gpu` reader depends on."""
    buf = _buffer()
    buf.gpu_is_current = False
    _copy_to_gpu(buf, 2)
    assert buf.gpu_is_current is True


def test_packing_alone_does_not_set_the_flag():
    """`gdn_attn` packs and never uploads; that must not read as fresh."""
    r = _runner()
    b = _builder(r)
    _build(b, _batch(), 1, 1, 1)
    _pack(b, _batch())
    assert r.forward_vars["block_tables"].gpu_is_current is False


# --- the guard ---------------------------------------------------------------


def test_guard_raises_on_last_batchs_rows():
    """The failure this exists for: non-None, in bounds, wrong batch."""
    r = _runner()
    b = _builder(r)
    _build(b, _batch(), 1, 1, 1)  # new step, nobody uploads
    with pytest.raises(AssertionError, match="nobody refreshed"):
        _guard(_proposer(r), _ctx())


def test_guard_passes_after_an_upload():
    r = _runner()
    b = _builder(r)
    _build(b, _batch(), 1, 1, 1)
    _pack(b, _batch())
    _copy_to_gpu(r.forward_vars["block_tables"], 1)
    _guard(_proposer(r), _ctx())


def test_guard_ignores_a_shuffle_draft():
    """A SHUFFLE draft's pure prefill never reads block_tables at all.

    Guarding it would fail a config that works today -- the regression this
    check must not introduce.
    """
    r = _runner()
    _build(_builder(r), _batch(), 1, 1, 1)
    _guard(_proposer(r, flash=False), _ctx())


def test_guard_ignores_a_target_without_an_eagle3_draft():
    r = _runner()
    _build(_builder(r), _batch(), 1, 1, 1)
    p = SimpleNamespace(runner=SimpleNamespace(forward_vars=r.forward_vars))
    _guard(p, _ctx())


def test_guard_is_quiet_on_a_dummy_run():
    r = _runner()
    _build(_builder(r), _batch(), 1, 1, 1)
    _guard(_proposer(r), _ctx(dummy=True))


def test_guard_is_quiet_before_any_build():
    """No `build` has run, so the flag was never cleared."""
    _guard(_proposer(_runner()), _ctx())


# --- wiring ------------------------------------------------------------------


def test_propose_actually_calls_the_guard():
    """Without this, deleting the one call site leaves every test above green.

    A source check: `propose` needs a loaded draft model and a live forward
    context, so the call cannot be observed from here.
    """
    import inspect

    src = inspect.getsource(eagle_proposer.EagleProposer.propose)
    guard = src.find("_assert_block_tables_are_this_steps")
    read = src.find('attn_metadata.block_tables = var["block_tables"].gpu')
    assert guard != -1, "propose no longer calls the freshness guard"
    assert read != -1, "the guarded read moved; re-anchor this test"
    assert guard < read, "the guard must run before the buffer is read"


def test_draft_builder_publishes_its_layout_choice(monkeypatch):
    """The guard keys off this; if the builder stops setting it, it goes quiet.

    Two layers, flash first and SHUFFLE second, because the predicate is
    per-layer (`head_dim`, `sliding_window`): a later SHUFFLE layer must not
    clear a flag an earlier flash layer needed.
    """
    kv = pytest.importorskip("atom.spec_decode.eagle3_kv_builder")
    monkeypatch.setenv("ATOM_FORCE_ATTN_TRITON", "0")
    L, BLOCKS, BS, NKV, HD = 2, 4, 16, 2, 128

    b = object.__new__(kv.Eagle3DraftBuilder)
    b.block_size, b.num_layers, b.num_blocks = BS, L, BLOCKS
    b.num_kv_heads, b.head_dim, b._next_layer_id = NKV, HD, 0
    b.model_runner = SimpleNamespace(
        eagle3_kv_cache=torch.zeros(2, L, BLOCKS, BS, NKV, HD, dtype=torch.bfloat16),
        eagle3_kv_scale=torch.zeros(2, L, BLOCKS, NKV, BS, dtype=torch.float32),
        config=SimpleNamespace(max_model_len=4096, kv_cache_dtype="bf16"),
    )

    def _mod(head_dim):
        return SimpleNamespace(
            base_attention=True,
            use_mla=False,
            impl=SimpleNamespace(
                rotary_emb=object(),
                q_norm=None,
                k_norm=None,
                sliding_window=-1,
                head_dim=head_dim,
                use_flash_layout=False,
            ),
        )

    b.build_kv_cache_tensor(0, _mod(64))  # head_dim != 128 -> flash
    assert b.uses_flash_layout is True
    b.build_kv_cache_tensor(1, _mod(128))  # -> SHUFFLE, must not clear it
    assert b.uses_flash_layout is True
