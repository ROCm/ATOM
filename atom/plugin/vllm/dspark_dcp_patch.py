"""Let Kimi-K3 DSpark draft against a DCP-sharded KV cache.

vLLM's DFlash/DSpark speculator predates decode context parallelism: it rejects
the combination in config validation, and its input-preparation kernel derives
every KV slot from the *global* token position, which is only the right answer
when one rank owns the whole cache. Two patches here lift both restrictions for
the ATOM plugin, which owns the Kimi-K3 model implementation the guard was
written against.

The third piece of the story lives elsewhere: the draft's per-rank sequence
lengths are derived in ``atom.plugin.vllm.attention.metadata`` rather than
plumbed through vLLM, because they are a pure function of the global ones.
"""

import inspect
import logging

import torch

logger = logging.getLogger("atom")

# (cp_size, cp_rank, cp_interleave), captured from the BlockTables the
# speculator is handed so the draft shards exactly the way the target does.
# None until the speculator has been wired up; a draft step cannot precede it.
_DCP_LAYOUT: tuple[int, int, int] | None = None


def apply_vllm_dspark_dcp_config_patch() -> None:
    """Drop vLLM's blanket "K3 DSpark cannot run under DCP" rejection.

    The check sits mid-way through ``SpeculativeConfig.__post_init__``, which
    pydantic binds when it builds the dataclass -- replacing the attribute
    afterwards has no effect. Its sole input is the target ``ParallelConfig``
    handed to the constructor, so mask DCP on that object across the one call
    that builds the config instead. Nothing else in ``__post_init__`` reads
    ``decode_context_parallel_size`` (the draft's own parallel config is built
    without it), and the value is restored before the config is returned.
    """
    from vllm.engine.arg_utils import EngineArgs

    if getattr(EngineArgs, "_atom_dspark_dcp_patch", False):
        return

    original_create = EngineArgs.create_speculative_config

    def create_speculative_config(
        self, target_model_config, target_parallel_config, *args, **kwargs
    ):
        dcp_size = getattr(target_parallel_config, "decode_context_parallel_size", 1)
        if dcp_size <= 1 or self.speculative_config is None:
            return original_create(
                self, target_model_config, target_parallel_config, *args, **kwargs
            )

        logger.info("ATOM patch: allowing speculative decoding under DCP%d.", dcp_size)
        target_parallel_config.decode_context_parallel_size = 1
        try:
            return original_create(
                self, target_model_config, target_parallel_config, *args, **kwargs
            )
        finally:
            target_parallel_config.decode_context_parallel_size = dcp_size

    EngineArgs.create_speculative_config = create_speculative_config
    EngineArgs._atom_dspark_dcp_patch = True


def _dcp_local_slots(
    positions: torch.Tensor,
    block_table: torch.Tensor,
    token_req: torch.Tensor,
    block_size: int,
    cp_size: int,
    cp_rank: int,
    cp_interleave: int,
    pad_slot_id: int,
) -> torch.Tensor:
    """Paged slots for ``positions``, as this DCP rank stores them.

    Mirrors ``BlockTables._compute_slot_mappings_kernel``: one block-table entry
    spans ``block_size * cp_size`` global tokens, of which this rank holds every
    ``cp_interleave``-sized run whose index modulo ``cp_size`` is its own, packed
    densely into its physical block. Positions owned by another rank come back as
    ``pad_slot_id``, which the cache-write kernels drop.

    ``token_req`` gives each position its row of ``block_table``; the two index
    tensors are applied together so no per-token copy of a row is materialized.
    """
    virtual_block = block_size * cp_size
    stride = block_table.shape[1]

    block_idx = torch.div(positions, virtual_block, rounding_mode="floor")
    block_offset = positions - block_idx * virtual_block
    block_number = block_table[token_req, block_idx.clamp_(max=stride - 1)]

    run = torch.div(block_offset, cp_interleave, rounding_mode="floor")
    is_local = torch.remainder(run, cp_size) == cp_rank
    local_offset = torch.div(run, cp_size, rounding_mode="floor") * cp_interleave + (
        block_offset - run * cp_interleave
    )
    slots = block_number.to(torch.int64) * block_size + local_offset
    return torch.where(is_local, slots, pad_slot_id)


def apply_vllm_dspark_dcp_input_patch() -> None:
    """Make the DFlash/DSpark draft address the KV cache in per-rank terms.

    The draft shares the target's block tables and writes into the same paged
    pool, so it inherits that pool's DCP sharding rather than choosing its own --
    but ``_prepare_dflash_inputs_kernel`` computes ``block_id * block_size +
    position % block_size``, the layout of an unsharded cache. Both the draft
    block's own K/V and the target-derived context rows would land on whichever
    rank-local slot that arithmetic happens to name.

    Rewriting the two slot mappings after the fact, instead of forking the
    kernel, keeps every other output it produces (input ids, sample indices,
    padding) on vLLM's implementation.
    """
    from vllm.v1.worker.gpu.spec_decode.dflash import speculator as dflash_speculator

    if getattr(dflash_speculator, "_atom_dspark_dcp_patch", False):
        return

    from vllm.v1.attention.backends.utils import PAD_SLOT_ID

    original_set_attn = dflash_speculator.DFlashSpeculator.set_attn
    original_prepare = dflash_speculator.prepare_dflash_inputs
    signature = inspect.signature(original_prepare)

    def set_attn(self, model_state, kv_cache_config, block_tables, *args, **kwargs):
        global _DCP_LAYOUT
        _DCP_LAYOUT = (
            block_tables.cp_size,
            block_tables.cp_rank,
            block_tables.cp_interleave,
        )
        if block_tables.cp_size > 1:
            logger.info(
                "ATOM patch: DSpark draft KV addressing localized for DCP%d "
                "(rank %d, interleave %d).",
                *_DCP_LAYOUT,
            )
        return original_set_attn(
            self, model_state, kv_cache_config, block_tables, *args, **kwargs
        )

    def prepare_dflash_inputs(*args, **kwargs):
        original_prepare(*args, **kwargs)

        assert _DCP_LAYOUT is not None, "prepare_dflash_inputs before set_attn"
        cp_size, cp_rank, cp_interleave = _DCP_LAYOUT
        if cp_size == 1:
            return

        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        arg = bound.arguments
        block_table = arg["block_table"]
        block_size = arg["block_size"]
        num_query_per_req = arg["num_query_per_req"]
        input_batch = arg["input_batch"]
        num_reqs = input_batch.num_reqs

        # Draft block: num_query_per_req consecutive query rows per request, so
        # the block-table row is the token index divided by that width. The
        # kernel clamps the positions it stores to max_model_len - 1 and derived
        # its own slots from the unclamped value; the two only differ past the
        # model length, which nothing schedules.
        num_query_tokens = num_reqs * num_query_per_req
        query_positions = arg["input_buffers"].positions[:num_query_tokens]
        query_req = torch.div(
            torch.arange(num_query_tokens, device=query_positions.device),
            num_query_per_req,
            rounding_mode="floor",
        )
        arg["query_slot_mapping"][:num_query_tokens].copy_(
            _dcp_local_slots(
                query_positions,
                block_table,
                query_req,
                block_size,
                cp_size,
                cp_rank,
                cp_interleave,
                PAD_SLOT_ID,
            )
        )

        # Context rows: one per target token, laid out back to back in target
        # batch order, so query_start_loc says which request each belongs to.
        num_context_tokens = input_batch.num_tokens
        context_positions = arg["context_positions"][:num_context_tokens]
        context_req = torch.searchsorted(
            input_batch.query_start_loc[1 : num_reqs + 1].to(torch.int64),
            torch.arange(num_context_tokens, device=context_positions.device),
            right=True,
        )
        arg["context_slot_mapping"][:num_context_tokens].copy_(
            _dcp_local_slots(
                context_positions,
                block_table,
                context_req,
                block_size,
                cp_size,
                cp_rank,
                cp_interleave,
                PAD_SLOT_ID,
            )
        )

    dflash_speculator.DFlashSpeculator.set_attn = set_attn
    dflash_speculator.prepare_dflash_inputs = prepare_dflash_inputs
    dflash_speculator._atom_dspark_dcp_patch = True
