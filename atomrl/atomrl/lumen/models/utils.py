# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

###############################################################################
#
# See LICENSE for license information.
###############################################################################

"""Model-agnostic training utilities.

Common helpers shared across all model implementations (LLaMA2, etc.).
No dependency on any specific model architecture, Megatron, or HuggingFace
Transformers — only Python stdlib and optional ``huggingface_hub`` /
``datasets`` for download helpers.
"""

import hashlib
import logging
import os
import sys

__all__ = [
    "download_hf_dataset",
    "download_hf_model",
    "peek_backend",
    "safe_add_argument",
    "sha256_file",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------


def safe_add_argument(parser_or_group, *args, **kwargs):
    """Add an argument only when none of its option strings are already registered.

    Megatron-LM-AMD pre-registers several ``--fp8-*`` arguments (and potentially
    others) in its own argument parser.  Calling ``add_argument`` a second time
    for the same flag raises ``argparse.ArgumentError``.  This helper silently
    skips registration when the argument is already present so that the Megatron
    definition takes precedence.

    Works with both a plain :class:`argparse.ArgumentParser` and an
    :class:`argparse._ArgumentGroup` (created via ``parser.add_argument_group``).
    """
    root_parser = getattr(parser_or_group, "_parser", parser_or_group)
    option_strings = [a for a in args if a.startswith("-")]
    if any(opt in root_parser._option_string_actions for opt in option_strings):
        return
    parser_or_group.add_argument(*args, **kwargs)


def peek_backend(default: str = "megatron") -> str:
    """Extract ``--backend`` from *sys.argv* without consuming it.

    This allows a unified entry-point script to dispatch to the correct
    backend before any framework-specific imports happen, while still
    leaving ``--backend`` in argv for the backend's own argument parser.

    Args:
        default: Value to return when ``--backend`` is not present.

    Returns:
        The backend string (e.g. ``"megatron"``, ``"fsdp"``).
    """
    for i, arg in enumerate(sys.argv):
        if arg == "--backend" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
        if arg.startswith("--backend="):
            return arg.split("=", 1)[1]
    return default


# ---------------------------------------------------------------------------
# File integrity
# ---------------------------------------------------------------------------


def sha256_file(path: str) -> str:
    """Compute the SHA-256 hex digest of a file.

    Reads in 1 MiB chunks to keep memory usage constant regardless of
    file size.

    Args:
        path: Absolute or relative path to the file.

    Returns:
        Lowercase hex string of the SHA-256 hash.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# HuggingFace download helpers
# ---------------------------------------------------------------------------


def download_hf_model(
    model_name: str,
    output_dir: str,
    verify: bool = False,
    revision: str | None = None,
) -> str:
    """Download a model snapshot from HuggingFace Hub.

    Args:
        model_name: Repository id (e.g. ``"NousResearch/Llama-2-7b-hf"``).
        output_dir: Local directory to store the downloaded files.
        verify: If ``True``, print SHA-256 hashes of every downloaded file.
        revision: Immutable HuggingFace commit to download.

    Returns:
        The *output_dir* path (for chaining).
    """
    if not revision:
        raise ValueError("revision must be an immutable HuggingFace commit")
    from huggingface_hub import snapshot_download

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Downloading %s@%s to %s ...", model_name, revision, output_dir)

    snapshot_download(
        repo_id=model_name,
        revision=revision,
        local_dir=output_dir,
        local_dir_use_symlinks=False,
    )
    logger.info("Download complete: %s", output_dir)

    if verify:
        logger.info("Verifying file hashes ...")
        for root, _dirs, files in os.walk(output_dir):
            for fname in sorted(files):
                fpath = os.path.join(root, fname)
                relpath = os.path.relpath(fpath, output_dir)
                digest = sha256_file(fpath)
                print(f"  {relpath}: {digest}")
        logger.info("Verification complete.")

    return output_dir


def download_hf_dataset(
    dataset_name: str,
    output_dir: str,
    subset: str | None = None,
    verify: bool = False,
) -> str:
    """Download a dataset from HuggingFace and save splits as jsonl.

    Args:
        dataset_name: HuggingFace dataset id (e.g. ``"tau/scrolls"``).
        output_dir: Local directory to store the ``.jsonl`` files.
        subset: Optional dataset subset / configuration name.
        verify: If ``True``, print SHA-256 hashes of each output file.

    Returns:
        The *output_dir* path (for chaining).
    """
    from datasets import load_dataset

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Downloading %s%s ...", dataset_name, f"/{subset}" if subset else "")

    dataset = load_dataset(dataset_name, subset)

    for split_name, split_data in dataset.items():
        output_path = os.path.join(output_dir, f"{split_name}.jsonl")
        split_data.to_json(output_path)
        logger.info(
            "  Saved %s: %d samples -> %s", split_name, len(split_data), output_path
        )
        if verify:
            digest = sha256_file(output_path)
            print(f"    SHA-256: {digest}")

    logger.info("Download complete.")
    return output_dir
