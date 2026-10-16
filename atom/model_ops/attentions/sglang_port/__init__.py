# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
#
# This package contains code copied from the SGLang project, which is licensed
# under the Apache License, Version 2.0.
"""Kernels vendored from SGLang, for the DFlash GQA draft read.

Upstream project: https://github.com/sgl-project/sglang (Apache-2.0).
Pinned sources, under ``python/sglang/kernels/ops/``:

  verify_mla.py   attention/verify_mla.py   @ 34180a0d3544
  kv_indices.py   kvcache/kv_indices.py     @ d7190b0122c5

Copied rather than reimplemented because this is the exact code path that was
validated at TP8 / DCP8 / DSpark k=5 against a DFlash draft checkpoint,
including ``--speculative-draft-kv-cache-dtype fp8_e4m3``. Reproducing the math
from the paper would forfeit that provenance.

Neither module imports anything but torch/triton, so vendoring them does not
drag in an ``sglang`` dependency.

Local modifications
-------------------
Every deviation from the pinned upstream files is marked inline with a
``# LOCAL FIX:`` comment. They are, all in ``verify_mla.py``:

* The single cross-file import (``_AMD_LAUNCH_KWARGS`` from
  ``verify_splitkv.py``) is inlined as the two-key literal that file defines
  for HIP, so the module stands alone.
* ``kv_indices`` loads are widened to ``tl.int64`` before being scaled by the
  pool stride. Upstream pages at ``ENTRY_PAGE_SIZE`` 1, so its slot ids stay
  small; ATOM's entry ids are ``block_size * dcp`` times larger and overflow
  int32 on a large pool.
* ``TARGET_PROGRAMS`` is retuned 512 -> 2048, so that the stage-1 split count
  still reaches ``MAX_N_SPLITS`` at the batch sizes ATOM serves.

``kv_indices.py`` is byte-identical to its pinned source. ATOM drives its
``TOKEN_BLOCK_PARALLEL`` arm via the ``kv_indices_num_token_blocks`` helper the
same file exports; that is a call-site choice, not a modification.

``verify_mla`` is, despite the name, not MLA-only: ``NOPE_DIM``/``PE_DIM``
generalize it, and at ``PE_DIM == 0`` it is a plain GQA attention over separate
K and V buffers. Its own tuning table names this case --
``_BLOCK_CONFIG[64] = (4, 256, 4)  # K3 GQA (dspark draft attention)``.
"""
