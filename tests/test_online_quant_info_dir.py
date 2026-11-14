# SPDX-License-Identifier: MIT
# Tests for the directory rank 0's online-quantization dump lands in.

import importlib
from types import SimpleNamespace

import pytest


def _import_loader():
    """The loader module, against AITER's stand-ins when it cannot import.

    Its import chain binds AITER kernels, which a CPU-only runner does not
    have. ``atom_under_stubs`` drops the modules it imported on the way out;
    the module object returned here keeps its own globals, which is all these
    tests reach for.
    """
    try:
        return importlib.import_module("atom.model_loader.loader")
    except ImportError as exc:
        # A third-party module missing is this CPU image. An import error
        # naming one of ours is a bug in the loader and has to surface
        # (tests/import_guard.py's rule), not be retried behind stand-ins.
        if (exc.name or "").startswith("atom"):
            raise
        from aiter_stub import atom_under_stubs

        with atom_under_stubs() as load:
            return load("atom.model_loader.loader")


@pytest.fixture(scope="module")
def loader():
    return _import_loader()


@pytest.fixture(autouse=True)
def _clear_dir_envs(monkeypatch):
    for name in ("ATOM_ONLINE_QUANT_INFO_DIR", "ATOM_TORCH_PROFILER_DIR"):
        monkeypatch.delenv(name, raising=False)


def _save(loader, monkeypatch, tmp_path, rank):
    monkeypatch.setattr(
        loader, "get_tp_group", lambda: SimpleNamespace(rank_in_group=rank)
    )
    monkeypatch.chdir(tmp_path)
    loader._save_online_quant_info(
        oq_layers=[],
        model_name_or_path="dummy",
        elapsed_seconds=1.0,
        online_quant_config={},
        timing_scope="load",
        peak_gpu_memory_gb=None,
    )


def _saved(directory):
    return list(directory.glob("online_quant_info_*.json"))


def test_own_knob_names_the_directory(loader, monkeypatch, tmp_path):
    own = tmp_path / "oq"
    monkeypatch.setenv("ATOM_ONLINE_QUANT_INFO_DIR", str(own))
    monkeypatch.setenv("ATOM_TORCH_PROFILER_DIR", str(tmp_path / "prof"))

    _save(loader, monkeypatch, tmp_path, rank=0)

    assert len(_saved(own)) == 1
    assert not (tmp_path / "prof").exists()


@pytest.mark.parametrize("own_value", [None, ""])
def test_unset_knob_falls_back_to_the_profiler_dir(
    loader, monkeypatch, tmp_path, own_value
):
    # The dump lived under ATOM_TORCH_PROFILER_DIR before it had a knob of its
    # own; unset or empty keeps landing there.
    if own_value is not None:
        monkeypatch.setenv("ATOM_ONLINE_QUANT_INFO_DIR", own_value)
    prof = tmp_path / "prof"
    monkeypatch.setenv("ATOM_TORCH_PROFILER_DIR", str(prof))

    _save(loader, monkeypatch, tmp_path, rank=0)

    assert len(_saved(prof)) == 1
    assert not _saved(tmp_path)


def test_no_directory_set_falls_back_to_the_working_directory(
    loader, monkeypatch, tmp_path
):
    _save(loader, monkeypatch, tmp_path, rank=0)

    assert len(_saved(tmp_path)) == 1


def test_other_ranks_write_nothing(loader, monkeypatch, tmp_path):
    own = tmp_path / "oq"
    monkeypatch.setenv("ATOM_ONLINE_QUANT_INFO_DIR", str(own))

    _save(loader, monkeypatch, tmp_path, rank=1)

    assert not own.exists()
