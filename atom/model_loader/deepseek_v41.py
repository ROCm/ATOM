# SPDX-License-Identifier: MIT
"""Native V4.1 checkpoint loading and mmap table resource lifetime."""

from contextlib import contextmanager

from aiter.dist.parallel_state import get_tp_group

from atom.models.deepseek_v41.weights import (
    CheckpointReader,
    build_weight_manifest,
    checkpoint_schema,
)


def load_checkpoint(model, directory, config, load_dummy=None):
    """Own the native source slices and their one post-load finalization pass."""
    if load_dummy:
        from atom.model_loader.loader import initialize_dummy_weights

        if load_dummy != "empty":
            initialize_dummy_weights(model, load_dummy)
        loaded = set(dict(model.named_parameters()))
    else:
        group = get_tp_group()
        schema = checkpoint_schema(config)
        manifest = build_weight_manifest(
            schema,
            tp_rank=group.rank_in_group,
            tp_size=group.world_size,
            ep_rank=group.rank_in_group,
            ep_size=group.world_size,
        )
        with CheckpointReader(directory, schema) as reader:
            loaded = reader.load_parameters(model, manifest)
    # The accepted native modules finalize their own quantized storage. They
    # must not also pass through the generic quant-method traversal.
    model.process_weights_after_loading()
    return loaded


@contextmanager
def engram_tables(directory, config):
    with CheckpointReader(directory, checkpoint_schema(config)) as reader:
        yield reader.engram_tables(config)
