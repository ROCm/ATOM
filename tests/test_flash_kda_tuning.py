# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""`balanced_chunks_per_seg`: a multi-sequence step's FlashKDA segment length,
sized on its total chunks by FlashKDA's own chooser (which also applies
FlashKDA's ``CHUNK_DELTA_ATTN_FLASH_KDA_SEG`` override)."""

import sys
from types import SimpleNamespace

import pytest
import torch
from aiter_stub import atom_without_aiter

# `atom.model_ops.kimi_k3` binds AITER kernels at import; the CPU job has no AITER.
with atom_without_aiter() as load:
    tuning = load("atom.model_ops.kimi_k3.flash_kda_tuning")
    balanced_chunks_per_seg = tuning.balanced_chunks_per_seg

H, V = 12, 128
PACKAGE = "aiter.ops.triton._triton_kernels.kimi_delta_attn"


@pytest.fixture
def fk(monkeypatch):
    """The FlashKDA names `balanced_chunks_per_seg` reads; its chooser
    records each call and returns ``fk.choice``."""
    calls = []

    def choose(n_chunks_max, n_seqs, heads, head_v_dim):
        calls.append((n_chunks_max, n_seqs, heads, head_v_dim))
        return fake.choice

    fake = SimpleNamespace(
        FLASH_KDA_CHUNK=64,
        _SEG_MIN_CHUNKS=64,
        get_num_sms=lambda: 256,
        _seq_bounds=lambda cu: tuple(zip(cu.tolist()[:-1], cu.tolist()[1:])),
        _choose_chunks_per_seg=choose,
        choice=32,
        calls=calls,
    )
    monkeypatch.setitem(sys.modules, PACKAGE, SimpleNamespace(flash_kda=fake))
    return fake


def cu(*lens):
    return torch.tensor([0, *torch.tensor(lens).cumsum(0).tolist()], dtype=torch.int32)


def test_flash_kdas_chooser_sizes_the_segments_on_the_total(fk):
    # 8K + 4K + 2K + 2K tokens: 128 + 64 + 32 + 32 chunks of 64.
    assert balanced_chunks_per_seg(cu(8192, 4096, 2048, 2048), H, V) == 32
    assert fk.calls == [(256, 4, H, V)]


@pytest.mark.parametrize("choice, want", [(1024, 128), (8, 8)])
def test_the_length_stays_within_one_and_the_longest_sequence(fk, choice, want):
    """A choice past the longest sequence (an override, say) leaves every
    sequence unsegmented, as FlashKDA's own call would."""
    fk.choice = choice
    assert balanced_chunks_per_seg(cu(8192, 4096, 2048, 2048), H, V) == want


@pytest.mark.parametrize(
    "lens", [(16384,), (2048, 2048), (8192,) * 7], ids=["one", "short", "full"]
)
def test_left_to_flash_kda(fk, lens):
    """One sequence, every sequence shorter than a segment's minimum, or no
    idle CU to fill (7 x 48 blocks > 256 CUs)."""
    assert balanced_chunks_per_seg(cu(*lens), H, V) is None
    assert fk.calls == []


@pytest.fixture
def flash_kda_modules(monkeypatch):
    """Stand-in FlashKDA modules holding every name in `FLASH_KDA_NAMES`."""
    tuning.check_flash_kda_names.cache_clear()
    modules = {}
    for module_name, names in tuning.FLASH_KDA_NAMES.items():
        module = SimpleNamespace(**dict.fromkeys(names, object()))
        modules[module_name] = module
        monkeypatch.setitem(sys.modules, f"{PACKAGE}.{module_name}", module)
    yield modules
    tuning.check_flash_kda_names.cache_clear()


def test_the_flash_kda_names_are_all_there(flash_kda_modules):
    tuning.check_flash_kda_names()


def test_a_missing_flash_kda_name_fails_the_model_build(flash_kda_modules, monkeypatch):
    del flash_kda_modules["flash_kda"]._k2_gluon_schedule
    monkeypatch.setattr(tuning, "_aiter_version", lambda: "9.9.9")
    with pytest.raises(RuntimeError) as error:
        tuning.check_flash_kda_names()
    message = str(error.value)
    assert f"{PACKAGE}.flash_kda._k2_gluon_schedule" in message
    assert "aiter 9.9.9" in message
