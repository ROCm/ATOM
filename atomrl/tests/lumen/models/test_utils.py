# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Tests for atomrl.lumen.models.utils — CLI helpers, file hashing, download helpers.

Covers:
  - safe_add_argument: idempotent registration, no duplicate errors
  - peek_backend: extract --backend from sys.argv
  - sha256_file: deterministic hash computation
  - download_hf_model / download_hf_dataset: argument forwarding (mocked I/O)
"""

import argparse
import hashlib
import os
import sys
import tempfile
from unittest import mock

import pytest
from atomrl.lumen.models.utils import (
    download_hf_dataset,
    download_hf_model,
    peek_backend,
    safe_add_argument,
    sha256_file,
)

# ===================================================================
# safe_add_argument
# ===================================================================


class TestSafeAddArgument:
    def test_adds_new_argument(self):
        parser = argparse.ArgumentParser()
        safe_add_argument(parser, "--foo", type=int, default=42)
        args = parser.parse_args([])
        assert args.foo == 42

    def test_skips_duplicate(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--foo", type=int, default=1)
        safe_add_argument(parser, "--foo", type=int, default=2)
        args = parser.parse_args([])
        assert args.foo == 1

    def test_works_with_argument_group(self):
        parser = argparse.ArgumentParser()
        group = parser.add_argument_group("test")
        safe_add_argument(group, "--bar", type=str, default="hello")
        args = parser.parse_args([])
        assert args.bar == "hello"

    def test_skips_duplicate_in_group(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--baz", type=float, default=1.0)
        group = parser.add_argument_group("test")
        safe_add_argument(group, "--baz", type=float, default=2.0)
        args = parser.parse_args([])
        assert args.baz == 1.0

    def test_short_and_long_option(self):
        parser = argparse.ArgumentParser()
        safe_add_argument(parser, "-v", "--verbose", action="store_true")
        args = parser.parse_args(["-v"])
        assert args.verbose is True

    def test_skips_if_short_already_registered(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-v", "--verbose", action="store_true")
        safe_add_argument(parser, "-v", "--verbose", action="store_false")
        args = parser.parse_args(["-v"])
        assert args.verbose is True


# ===================================================================
# peek_backend
# ===================================================================


class TestPeekBackend:
    def test_default_when_absent(self):
        with mock.patch.object(sys, "argv", ["script.py"]):
            assert peek_backend() == "megatron"

    def test_custom_default(self):
        with mock.patch.object(sys, "argv", ["script.py"]):
            assert peek_backend(default="fsdp") == "fsdp"

    def test_space_separated(self):
        with mock.patch.object(sys, "argv", ["script.py", "--backend", "fsdp"]):
            assert peek_backend() == "fsdp"

    def test_equals_separated(self):
        with mock.patch.object(sys, "argv", ["script.py", "--backend=fsdp"]):
            assert peek_backend() == "fsdp"

    def test_other_args_ignored(self):
        with mock.patch.object(
            sys,
            "argv",
            ["script.py", "--lr", "0.001", "--backend", "megatron", "--epochs", "10"],
        ):
            assert peek_backend() == "megatron"

    def test_backend_as_last_arg_without_value(self):
        with mock.patch.object(sys, "argv", ["script.py", "--backend"]):
            assert peek_backend() == "megatron"


# ===================================================================
# sha256_file
# ===================================================================


class TestSha256File:
    def test_deterministic(self):
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".bin") as f:
            f.write(b"hello world")
            path = f.name
        try:
            h1 = sha256_file(path)
            h2 = sha256_file(path)
            assert h1 == h2
        finally:
            os.unlink(path)

    def test_matches_hashlib(self):
        content = b"test content for hashing"
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".bin") as f:
            f.write(content)
            path = f.name
        try:
            expected = hashlib.sha256(content).hexdigest()
            assert sha256_file(path) == expected
        finally:
            os.unlink(path)

    def test_empty_file(self):
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".bin") as f:
            path = f.name
        try:
            expected = hashlib.sha256(b"").hexdigest()
            assert sha256_file(path) == expected
        finally:
            os.unlink(path)

    def test_returns_lowercase_hex(self):
        with tempfile.NamedTemporaryFile(mode="wb", delete=False, suffix=".bin") as f:
            f.write(b"abc")
            path = f.name
        try:
            result = sha256_file(path)
            assert result == result.lower()
            assert len(result) == 64
        finally:
            os.unlink(path)


# ===================================================================
# download_hf_model (mocked)
# ===================================================================


class TestDownloadHfModel:
    def test_calls_snapshot_download(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch("huggingface_hub.snapshot_download") as mock_snap:
                result = download_hf_model("meta-llama/test", tmpdir, revision="abc123")
            mock_snap.assert_called_once_with(
                repo_id="meta-llama/test",
                revision="abc123",
                local_dir=tmpdir,
                local_dir_use_symlinks=False,
            )
            assert result == tmpdir

    def test_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as base:
            outdir = os.path.join(base, "sub", "dir")
            with mock.patch("huggingface_hub.snapshot_download"):
                download_hf_model("test/model", outdir, revision="abc123")
            assert os.path.isdir(outdir)

    def test_verify_prints_hashes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            test_file = os.path.join(tmpdir, "weights.bin")
            with open(test_file, "wb") as f:
                f.write(b"test data")

            with mock.patch("huggingface_hub.snapshot_download"):
                result = download_hf_model(
                    "test/model", tmpdir, verify=True, revision="abc123"
                )
            assert result == tmpdir

    def test_requires_immutable_revision(self):
        with (
            tempfile.TemporaryDirectory() as tmpdir,
            pytest.raises(ValueError, match="revision"),
        ):
            download_hf_model("test/model", tmpdir)


# ===================================================================
# download_hf_dataset (mocked)
# ===================================================================


class TestDownloadHfDataset:
    def _make_mock_dataset(self, splits=None):
        if splits is None:
            splits = {"train": 100}
        dataset = {}
        for name, count in splits.items():
            split = mock.MagicMock()
            split.__len__ = mock.MagicMock(return_value=count)
            dataset[name] = split
        return dataset

    def test_creates_output_dir(self):
        with tempfile.TemporaryDirectory() as base:
            outdir = os.path.join(base, "ds_out")
            mock_dataset = self._make_mock_dataset()
            with mock.patch("datasets.load_dataset", return_value=mock_dataset):
                download_hf_dataset("test/dataset", outdir)
            assert os.path.isdir(outdir)

    def test_writes_splits_as_jsonl(self):
        with tempfile.TemporaryDirectory() as base:
            outdir = os.path.join(base, "ds")
            mock_dataset = self._make_mock_dataset({"train": 50, "validation": 10})
            with mock.patch("datasets.load_dataset", return_value=mock_dataset):
                download_hf_dataset("test/dataset", outdir)
            mock_dataset["train"].to_json.assert_called_once()
            mock_dataset["validation"].to_json.assert_called_once()

    def test_subset_forwarded(self):
        with tempfile.TemporaryDirectory() as base:
            outdir = os.path.join(base, "ds")
            mock_dataset = self._make_mock_dataset()
            with mock.patch(
                "datasets.load_dataset", return_value=mock_dataset
            ) as mock_load:
                download_hf_dataset("test/dataset", outdir, subset="mini")
            mock_load.assert_called_once_with("test/dataset", "mini")

    def test_returns_output_dir(self):
        with tempfile.TemporaryDirectory() as base:
            outdir = os.path.join(base, "ds")
            mock_dataset = self._make_mock_dataset()
            with mock.patch("datasets.load_dataset", return_value=mock_dataset):
                result = download_hf_dataset("test/dataset", outdir)
            assert result == outdir
