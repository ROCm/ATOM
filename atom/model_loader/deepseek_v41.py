# SPDX-License-Identifier: MIT
"""Lifetime of V4.1's host-owned Engram tables.

Weights go through `atom.model_loader.loader.load_model` like every other
model's. These tables do not: they are mmap'd host embedding tables that the
Engram prefetcher reads row-wise, not parameters, so nothing in the parameter
path ever sees them and the mapping has to stay alive as long as the host does.
"""

from contextlib import contextmanager

from atom.models.deepseek_v41.weights import CheckpointReader, checkpoint_schema


@contextmanager
def engram_tables(directory, config):
    with CheckpointReader(directory, checkpoint_schema(config)) as reader:
        yield reader.engram_tables(config)
