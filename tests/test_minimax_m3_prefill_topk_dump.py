"""CPU checks of the default-off MiniMax-M3 prefill top-k dump
(``ATOM_M3_DUMP_PREFILL_TOPK``, ``minimax_m3/prefill_topk_dump.py``)."""

import importlib

import torch


def _module(monkeypatch, tmp_path, **env):
    monkeypatch.setenv("ATOM_M3_DUMP_PREFILL_TOPK", str(tmp_path))
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    mod = importlib.import_module("atom.model_ops.minimax_m3.prefill_topk_dump")
    mod._state.update(chunk=-1, last_layer=None, skip=False)
    return mod


def _forward(mod, layers, prefix):
    rows = 8
    for layer in layers:
        mod.maybe_dump_prefill_topk(
            layer,
            torch.arange(rows * 16, dtype=torch.int32).view(1, rows, 16),
            None,
            torch.full((rows,), 2048, dtype=torch.int32),
            torch.tensor([0, rows], dtype=torch.int32),
            torch.tensor([prefix], dtype=torch.int32),
            torch.tensor([prefix + rows], dtype=torch.int32),
        )


def test_dumps_long_prefix_chunks_only(monkeypatch, tmp_path):
    mod = _module(
        monkeypatch,
        tmp_path,
        ATOM_M3_DUMP_PREFILL_TOPK_CHUNKS=2,
        ATOM_M3_DUMP_PREFILL_TOPK_MIN_PREFIX=1000,
    )
    _forward(mod, [0, 1, 2], prefix=10)  # warm-up / short prompt: skipped
    _forward(mod, [0, 1, 2], prefix=5000)  # chunk 0
    _forward(mod, [0, 2], prefix=6000)  # chunk 1 (a skip-index layer missing)
    _forward(mod, [0, 1, 2], prefix=7000)  # past the chunk budget
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [
        "chunk000_layer00.pt",
        "chunk000_layer01.pt",
        "chunk000_layer02.pt",
        "chunk001_layer00.pt",
        "chunk001_layer02.pt",
    ]
    saved = torch.load(tmp_path / "chunk001_layer02.pt")
    assert saved["layer"] == 2 and saved["sparse_bt"] is None
    assert int(saved["prefix_lens"][0]) == 6000
    assert saved["topk_idx"].shape == (1, 8, 16)


def test_off_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("ATOM_M3_DUMP_PREFILL_TOPK", raising=False)
    mod = importlib.import_module("atom.model_ops.minimax_m3.prefill_topk_dump")
    mod._state.update(chunk=-1, last_layer=None, skip=False)
    _forward(mod, [0, 1], prefix=50000)
    assert not any(tmp_path.iterdir())
