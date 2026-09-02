# SPDX-License-Identifier: MIT
"""The draft KV layout decision, and the invariant the freshness guard rests on.

**This file is the part of the change that runs on CI.** CI has no aiter
(`tests/conftest.py`), and `eagle3_kv_builder` does `from aiter import dtypes`
at import time, so a test reaching the predicate through the builder is skipped
-- on the very CI whose OOM this fixes. `atom.spec_decode.draft_kv_layout` does
not need aiter, so it imports here.

Scope: the shape terms of the decision. The env read that feeds it now lives in
`PagedAttentionImpl.__init__` as `use_triton_attn`, behind the aiter barrier, so
here it is an input rather than something these tests cover.

Getting the decision wrong is silent both ways: every layout holds the same
element count, so a mismatched pool reads transposed data without faulting. A
draft that today gets a working SHUFFLE V (not on the Triton path -- the
Kimi-K2.5 shape) must keep it.

The last test guards something else: `block_tables` freshness is tracked in
`CpuGpuBuffer.copy_to_gpu`, which is only sound while every upload goes through
it. Nothing else pins that, and breaking it fails in the loud direction -- the
guard would raise and take the worker down on a config that works today.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from atom.spec_decode.draft_kv_layout import use_flash_layout

_ROOT = Path(__file__).resolve().parents[1]


def _impl(*, rotary=True, qk_norm=False, triton=False):
    """`use_triton_attn` is set in PagedAttentionImpl.__init__ from the env,
    the sliding window and head_dim; the predicate only reads the result."""
    norm = object() if qk_norm else None
    return SimpleNamespace(
        rotary_emb=object() if rotary else None,
        q_norm=norm,
        k_norm=norm,
        use_triton_attn=triton,
    )


def test_the_predicate_imports_without_aiter(tmp_path):
    """The property that makes everything here CI coverage rather than a skip.

    In a subprocess with `aiter` shadowed by a module that raises, because
    `sys.modules` in this process is not evidence: the sibling test files
    `importorskip` modules that pull aiter in, so a check here would pass or
    fail depending on which tests ran first.
    """
    stub = tmp_path / "aiter"
    stub.mkdir()
    (stub / "__init__.py").write_text('raise ImportError("no aiter here")')
    env = dict(os.environ, PYTHONPATH=f"{tmp_path}{os.pathsep}{_ROOT}")
    got = subprocess.run(
        [sys.executable, "-c", "import atom.spec_decode.draft_kv_layout"],
        capture_output=True,
        text=True,
        env=env,
        cwd=_ROOT,
        check=False,  # the returncode is the assertion
    )
    assert got.returncode == 0, (
        "the predicate no longer imports without aiter, so on CI this file is "
        "skipped and the layout decision goes untested:\n" + got.stderr[-2000:]
    )


# --- the decision ------------------------------------------------------------


def test_minimax_m3_draft_gets_flash():
    """use_triton_attn is what sends rope_cache down the 4D-V writer."""
    assert use_flash_layout(_impl(triton=True)) is True


def test_kimi_k25_draft_keeps_its_layout():
    """Not on the Triton path: already a working SHUFFLE V from
    reshape_and_cache(asm_layout=True), and a flash pool would break it."""
    assert use_flash_layout(_impl(triton=False)) is False


def test_a_sparse_impl_is_classified_by_its_own_flag():
    """SparseMHAPagedAttentionImpl sets `use_triton_attn = False` in `__init__`
    and hardcodes SHUFFLE in its `rope_cache` override.

    `qk_norm=False` on purpose: with both norms the early return above fires
    first and the flag is never read, so the case would pass whatever the flag
    said -- which is the thing being checked.
    """
    assert use_flash_layout(_impl(triton=False, qk_norm=False)) is False


def test_qk_norm_draft_is_left_alone():
    """rope_cache's first branch re-views V to SHUFFLE on its own, so this
    module never sees the 4D V and must not be handed a flash pool."""
    assert use_flash_layout(_impl(triton=True, qk_norm=True)) is False


def test_only_one_norm_still_counts_as_the_4d_writer():
    """rope_cache's first branch needs BOTH norms; one alone falls through."""
    impl = _impl(triton=True)
    impl.q_norm = object()  # k_norm stays None
    assert use_flash_layout(impl) is True


def test_non_attention_module():
    """build_kv_cache_tensor passes getattr(module, "impl", None)."""
    assert use_flash_layout(None) is False


def test_rope_less_draft_is_a_known_gap():
    """A KNOWN GAP, pinned so it is not mistaken for a deliberate choice.

    Without rotary_emb this returns False, which is right on its own -- but not
    when `use_triton_attn` also holds: rope_cache then falls through to
    reshape_and_cache with asm_layout=False, which writes the very 4D V this
    predicate exists to avoid, so that draft keeps paying the whole-pool
    permute().contiguous(). No such draft exists in-tree today
    (Eagle3LlamaAttention always builds a rotary_emb), which is why it is
    recorded rather than fixed. If you are here because a rope-less draft OOMs
    on a whole-pool convert: this assertion is the bug, not the contract.
    """
    assert use_flash_layout(_impl(rotary=False, triton=True)) is False


# --- the invariant the freshness guard rests on ------------------------------


def test_every_block_tables_upload_goes_through_copy_to_gpu():
    """`copy_to_gpu` is where freshness is recorded, so it must be the only way
    the GPU copy is written.

    A producer that refreshes `.gpu` by another route leaves the flag false and
    the guard raises on a config that works -- the loud failure direction, and
    the one nothing else here covers. Source scan rather than a runtime check:
    the offending write would be in a backend that needs a GPU to reach.
    """
    # Any direct write to a `.gpu` tensor, not just one naming block_tables on
    # the same line: `b = var["block_tables"]` followed by `b.gpu[:n].copy_(...)`
    # is the shape a refactor actually takes, and matching on the name misses it.
    bad = re.compile(r"\.gpu\s*(\[[^\]]*\])?\s*(=[^=]|\.copy_\()")
    # `CpuGpuBuffer` is the implementation; `kv_last_page_lens` is a pre-existing
    # direct write on a different buffer, harmless because nothing tracks its
    # freshness. Both allowlisted deliberately -- a new entry here is a decision,
    # which is the point of the test.
    allow_file = {"atom/utils/__init__.py"}
    allow_line = ("kv_last_page_lens",)
    offenders = []
    for path in (_ROOT / "atom").rglob("*.py"):
        rel = str(path.relative_to(_ROOT))
        if "plugin" in path.parts or rel in allow_file:
            continue
        for i, line in enumerate(path.read_text(errors="ignore").splitlines(), 1):
            if bad.search(line) and not any(a in line for a in allow_line):
                offenders.append(f"{rel}:{i}: {line.strip()}")
    assert not offenders, (
        "block_tables' GPU copy is written outside CpuGpuBuffer.copy_to_gpu, "
        "so CommonAttentionBuilder.build's freshness flag would read false "
        "after a real upload and the Eagle3 guard would fire on a working "
        "config:\n" + "\n".join(offenders)
    )


def test_the_prefill_producer_still_uploads_block_tables():
    """The base prefill upload the guard depends on for a cached-prefix step.

    Deleting it is invisible to every other test here and to CI, and shows up
    only as a worker killed by the guard on the first prefix-carrying prefill.
    """
    src = (_ROOT / "atom/model_ops/attentions/backends.py").read_text()
    assert 'vars_used.append(("block_tables"' in src, (
        "CommonAttentionBuilder.prepare_prefill no longer uploads block_tables "
        "on a cached-prefix step; the Eagle3 freshness guard will fire"
    )
