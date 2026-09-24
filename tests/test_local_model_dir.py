# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""A hub id must never reach a path join.

`Config.model` is whatever the caller typed -- a checkpoint directory or a hub
id. Code that opens a file by joining onto it gets `org/name/file` for the
second kind: a relative path that does not exist, whose absence reads as a
missing checkpoint. That is how CI's
`deepseek-ai/DeepSeek-V4.1-Flash/model.safetensors.index.json` came about, in
the V4.1 Engram loader, on a run whose weights had loaded fine moments before.
"""

import pathlib

import pytest

from atom.model_loader.weight_utils import local_model_dir


def test_a_directory_is_its_own_answer(tmp_path):
    """A local checkpoint must not be handed to the hub at all.

    An offline box with no hub cache still has to serve a path, so this is the
    branch that cannot be allowed to reach the network for its answer.
    """
    called = []
    from atom.model_loader import weight_utils

    original = weight_utils.snapshot_download
    weight_utils.snapshot_download = lambda *a, **k: called.append(a) or "wrong"
    try:
        assert local_model_dir(str(tmp_path)) == str(tmp_path)
    finally:
        weight_utils.snapshot_download = original
    assert not called, "a real directory went to the hub"


def test_an_id_resolves_to_the_cached_snapshot(monkeypatch):
    """The id goes to the hub cache, and what comes back is a real directory.

    `allow_patterns=[]` and `local_files_only=True` together are the contract:
    resolve where the files already are, download nothing. A caller runs after
    the weights have loaded, so asking the network again would be both slow and
    a new failure mode on an offline box.
    """
    seen = {}

    def fake_snapshot_download(model, **kwargs):
        seen.update(model=model, **kwargs)
        return "/somewhere/models--org--name/snapshots/abc"

    monkeypatch.setattr(
        "atom.model_loader.weight_utils.snapshot_download", fake_snapshot_download
    )
    resolved = local_model_dir("org/name")
    assert resolved == "/somewhere/models--org--name/snapshots/abc"
    assert seen["model"] == "org/name"
    assert seen["local_files_only"] is True
    assert seen["allow_patterns"] == []


def test_joining_onto_the_resolved_path_is_what_the_caller_wanted(monkeypatch):
    """The regression itself, stated as the thing that used to be built.

    Armed: without the resolution the join produces a relative `org/name/...`,
    which is exactly the string CI reported -- so this fails on the old code
    rather than passing either way.
    """
    monkeypatch.setattr(
        "atom.model_loader.weight_utils.snapshot_download",
        lambda model, **kwargs: "/hub/snapshots/abc",
    )
    joined = pathlib.Path(local_model_dir("deepseek-ai/DeepSeek-V4.1-Flash"))
    joined = joined / "model.safetensors.index.json"
    assert joined.is_absolute()
    assert "deepseek-ai/DeepSeek-V4.1-Flash" not in str(joined)


def test_a_name_that_resolves_to_nothing_comes_back_unchanged(monkeypatch):
    """Best-effort, because every caller already answers "no such directory".

    Three of the four callers are probes whose whole contract is to degrade on
    a checkpoint they cannot read, and raising here would make each of them
    wrap this in the same `try`. The one caller that must be loud is loud on
    its own: `CheckpointReader` refuses the string and names this function,
    which the test below pins.
    """

    def boom(model, **kwargs):
        raise OSError("not in the cache")

    monkeypatch.setattr("atom.model_loader.weight_utils.snapshot_download", boom)
    assert local_model_dir("org/never-downloaded") == "org/never-downloaded"


def test_an_unresolved_id_reaching_the_reader_names_the_real_problem():
    """The guard for the caller this helper cannot reach.

    `local_model_dir` fixes the one call site that had the bug; nothing stops
    the next one from passing `Config.model` straight through. What the reader
    can do is refuse the string it cannot use and say why, instead of letting
    it become a `FileNotFoundError` on a path that looks like a checkpoint that
    went missing.
    """
    from atom.models.deepseek_v41.weights import CheckpointReader

    with pytest.raises(NotADirectoryError, match="local_model_dir"):
        CheckpointReader("deepseek-ai/DeepSeek-V4.1-Flash", {})
