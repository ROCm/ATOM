# SPDX-License-Identifier: MIT
"""atom.model_ops.expert_counts on CPU: counts match a bincount, a forward's counts carry its own label."""

import glob
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from atom.model_ops import expert_counts as ec

E, TOPK = 896, 16


class _Event:
    def record(self):
        pass

    def query(self):
        return True

    def synchronize(self):
        pass


@pytest.fixture
def counts(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "DIR", str(tmp_path))
    monkeypatch.setattr(ec, "_buf", None)
    monkeypatch.setattr(ec, "_pending", [])
    monkeypatch.setattr(ec, "_rows", [])
    monkeypatch.setattr(ec, "_label", None)
    monkeypatch.setattr(ec, "_seq", 0)
    monkeypatch.setattr(ec.torch.cuda, "Event", _Event)
    empty = torch.empty
    monkeypatch.setattr(
        ec.torch, "empty", lambda *a, pin_memory=False, **k: empty(*a, **k)
    )
    return tmp_path


def _layer(i):
    return SimpleNamespace(
        prefix=f"model.layers.{i}.mlp.experts",
        top_k=TOPK,
        expert_layout=SimpleNamespace(num_logical=E),
    )


def _ids(rows, pad, gen):
    ids = torch.stack(
        [torch.randperm(E, generator=gen)[:TOPK] for _ in range(rows)]
    ).to(torch.int32)
    ids[rows - pad :] = -1
    # an appended shared-expert column past top_k must not be counted
    return torch.cat([ids, torch.full((rows, 1), E, dtype=torch.int32)], dim=1)


def test_counts_and_labels(counts, monkeypatch):
    gen = torch.Generator().manual_seed(0)
    expect = []
    for step in range(5):
        ec.start_forward(f"decode[step={step}]", dp_rank=3)
        want = np.zeros((93, E), np.int64)
        for layer in (1, 2, 92):
            ids = _ids(40, 7, gen)
            ec.record(_layer(layer), ids)
            real = ids[:, :TOPK].numpy().ravel()
            want[layer] = np.bincount(real[real >= 0], minlength=E)
        expect.append(want)
    ec._at_exit()  # closes step 4
    (f,) = glob.glob(f"{counts}/dp3_*_00000.npz")
    z = np.load(f)
    assert list(z["label"]) == [f"decode[step={s}]" for s in range(5)]
    assert z["counts"].shape == (5, 93, E) and z["counts"].dtype == np.int16
    for s in range(5):
        np.testing.assert_array_equal(z["counts"][s], expect[s])
        assert (
            z["counts"][s].sum() == 3 * 33 * TOPK
        )  # 40 rows, 7 pad, 16 picks each, 3 layers


def test_unlabelled_counts_dropped(counts):
    # graph capture runs the layers before the first labelled forward: not recorded
    ec.record(_layer(5), _ids(8, 0, torch.Generator().manual_seed(1)))
    ec.start_forward("decode[a]", dp_rank=0)
    ec.start_forward("decode[b]", dp_rank=0)
    ec._at_exit()
    (f,) = glob.glob(f"{counts}/dp0_*.npz")
    z = np.load(f)
    assert list(z["label"]) == ["decode[a]", "decode[b]"] and z["counts"].sum() == 0


def test_flush_every_256(counts):
    layer = _layer(1)
    ids = _ids(4, 0, torch.Generator().manual_seed(2))
    for step in range(ec._FLUSH + 2):
        ec.start_forward(f"decode[{step}]", dp_rank=1)
        ec.record(layer, ids)
    assert len(glob.glob(f"{counts}/dp1_*.npz")) == 1
    ec._at_exit()
    files = sorted(glob.glob(f"{counts}/dp1_*.npz"))
    n = [len(np.load(f)["label"]) for f in files]
    assert n == [ec._FLUSH, 2]
