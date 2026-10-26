# SPDX-License-Identifier: MIT
"""Exercise draft subfolder resolution without downloading model weights."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / ".github/scripts/atomesh/resolve_draft_model.py"
)
SPEC = importlib.util.spec_from_file_location("resolve_draft_model", SCRIPT)
resolver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resolver)


class ResolveDraftModelTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def test_downloads_only_nested_draft_and_returns_local_directory(self):
        tmp_path = self.root
        draft = tmp_path / "dspark"
        draft.mkdir()
        (draft / "config.json").write_text("{}")
        download = Mock(return_value=str(tmp_path))
        with patch.dict(
            sys.modules, huggingface_hub=SimpleNamespace(snapshot_download=download)
        ):
            assert resolver.resolve_draft_model("amd/model", "amd/model/dspark") == str(
                draft
            )
        download.assert_called_once_with(
            repo_id="amd/model", allow_patterns=["dspark/*"]
        )

    def test_missing_draft_config_is_an_error(self):
        tmp_path = self.root
        download = Mock(return_value=str(tmp_path))
        with (
            patch.dict(
                sys.modules, huggingface_hub=SimpleNamespace(snapshot_download=download)
            ),
            self.assertRaisesRegex(FileNotFoundError, "Draft config missing"),
        ):
            resolver.resolve_draft_model("amd/model", "amd/model/dspark")

    def test_existing_paths_and_independent_repositories_are_preserved(self):
        tmp_path = self.root
        download = Mock(side_effect=AssertionError("Unexpected model download"))
        with patch.dict(
            sys.modules, huggingface_hub=SimpleNamespace(snapshot_download=download)
        ):
            for model, draft in [
                ("amd/model", ""),
                ("amd/model", "org/draft"),
                ("amd/model", str(tmp_path)),
                (str(tmp_path), str(tmp_path / "dspark")),
            ]:
                assert resolver.resolve_draft_model(model, draft) == draft

    def test_rejects_parent_directory_in_subfolder(self):
        with self.assertRaisesRegex(ValueError, "Invalid draft subfolder"):
            resolver.resolve_draft_model("amd/model", "amd/model/../draft")
