"""Unit tests for the checkpoint shard preflight check.

Synthetic checkpoints only -- no real model config is read, so these stay valid
as model configs churn.
"""

from __future__ import annotations

import json

import pytest

from atom.model_loader.loading_core import verify_shard_files_present

INDEX_NAME = "model.safetensors.index.json"


def _write_checkpoint(root, shards: list[str], present: list[str] | None = None):
    """Declare `shards` in the index; materialize only `present` (default: all)."""
    weight_map = {f"layer.{i}.weight": s for i, s in enumerate(shards)}
    (root / INDEX_NAME).write_text(json.dumps({"weight_map": weight_map}))
    for shard in shards if present is None else present:
        (root / shard).write_bytes(b"")
    return root


def test_complete_checkpoint_passes(tmp_path):
    _write_checkpoint(tmp_path, ["a.safetensors", "b.safetensors"])
    verify_shard_files_present(str(tmp_path))  # must not raise


def test_missing_shard_raises_and_names_it(tmp_path):
    _write_checkpoint(
        tmp_path,
        ["a.safetensors", "b.safetensors", "c.safetensors"],
        present=["a.safetensors", "c.safetensors"],
    )
    with pytest.raises(FileNotFoundError) as exc:
        verify_shard_files_present(str(tmp_path))
    msg = str(exc.value)
    assert "b.safetensors" in msg
    assert "references 3 shard file(s)" in msg
    assert "1 of them are absent" in msg
    # The shards that ARE present must not be blamed.
    assert "a.safetensors" not in msg
    assert "c.safetensors" not in msg


def test_repeated_shard_counted_once(tmp_path):
    """weight_map maps many tensors onto few shards; dedupe before counting."""
    root = tmp_path
    weight_map = {f"layer.{i}.weight": "only.safetensors" for i in range(50)}
    (root / INDEX_NAME).write_text(json.dumps({"weight_map": weight_map}))
    with pytest.raises(FileNotFoundError) as exc:
        verify_shard_files_present(str(root))
    assert "references 1 shard file(s)" in str(exc.value)


def test_long_missing_list_is_elided(tmp_path):
    shards = [f"s{i:03d}.safetensors" for i in range(25)]
    _write_checkpoint(tmp_path, shards, present=[])
    with pytest.raises(FileNotFoundError) as exc:
        verify_shard_files_present(str(tmp_path))
    msg = str(exc.value)
    assert "25 of them are absent" in msg
    assert "... and 5 more" in msg


def test_no_index_is_noop(tmp_path):
    """Single-file checkpoint: nothing to cross-check."""
    (tmp_path / "model.safetensors").write_bytes(b"")
    verify_shard_files_present(str(tmp_path))


def test_nonexistent_path_is_noop(tmp_path):
    """A bare HF repo id reaches the loader as a non-directory string."""
    verify_shard_files_present(str(tmp_path / "some-org" / "some-model"))


def test_unreadable_index_is_noop(tmp_path):
    """Malformed index is the existing load path's error to report, not ours."""
    (tmp_path / INDEX_NAME).write_text("{not json")
    verify_shard_files_present(str(tmp_path))


def test_index_without_weight_map_is_noop(tmp_path):
    (tmp_path / INDEX_NAME).write_text(json.dumps({"metadata": {}}))
    verify_shard_files_present(str(tmp_path))
