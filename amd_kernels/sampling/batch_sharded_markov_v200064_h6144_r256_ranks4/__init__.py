# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""Batch-sharded sampling on four ranks (vocab 200064, hidden 6144, Markov rank 256): the target's verify logits
exchange, the sampler-output gather and the Gumbel Markov draft chain, sequenced on the per-kernel entries."""

from .host import (MAX_OWNED, MAX_REQS, MAX_ROWS, MAX_STEPS, BatchShardedSampling, OwnerTables, markov_blocker,
                   owner_tables, sampling_blocker)

__all__ = ["BatchShardedSampling", "OwnerTables", "markov_blocker", "owner_tables", "sampling_blocker", "MAX_OWNED",
           "MAX_REQS", "MAX_ROWS", "MAX_STEPS"]
