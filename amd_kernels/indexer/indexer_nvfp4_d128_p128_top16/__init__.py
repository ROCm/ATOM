# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Lightning-indexer kernels over NVFP4 index pages on gfx950 (FlyDSL): index head
dim 128, 128-token blocks (one index page each), top-16 blocks.

- ``score_decode``: decode block scores (the one-head scorers and the chunked
  shared-page kernel);
- ``score_prefill``: prefill block scores (512-row tiles, and the shared-dequant
  kernel for small work);
- ``topk``: the top-16 blocks of each score row, with or without the attend's
  page table;
- ``prefill_top16``: the prefill top-16 entry (the one-GPU scorer + top-16 of
  every prefill row in one call), its support check and warm-up;
- ``decode_top16``: the decode top-16 route (four-GPU scorer by request count, else
  the one-GPU scorer + top-16), its warm-up and the four-GPU route's buffers;
- ``split4``: the decode indexer split over four GPUs on the rank-owned index
  cache (rank r holds the blocks b % 4 == r).

The contract is ``reference.py``.
"""
