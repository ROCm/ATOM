# SPDX-License-Identifier: MIT
"""Startup-only CPU offload round trips, before the scheduler owns KV blocks."""

import logging
import time
import uuid

import torch

from atom.kv_transfer.offload._offload_common import tokens_to_tensor

logger = logging.getLogger("atom")


def warmup_cpu_offload(worker) -> None:
    """Warm the real LMCache and thread-local staging paths, or fail startup.

    A request warmup can hit HBM exclusively. Here we explicitly store unused
    MLA/KV blocks to LocalCPUBackend, then retrieve into different unused blocks
    on the serving load executor. No scheduler completions or KDA state images
    are created. Call only after graph capture and before request admission.

    The initial implementation deliberately matches the single-save/single-load
    CPU-only recipe. Other executor widths need one warmup per thread (staging
    is thread-local); silently warming just one would leave a cold serving path.
    """
    if getattr(worker, "_offload_warmed", False):
        return
    engine, codec = worker._engine, worker._codec
    if engine is None or codec is None:
        raise RuntimeError("OFFLOAD_WARMUP requires registered KV caches")
    if not (worker._do_save and worker._do_load):
        raise RuntimeError("OFFLOAD_WARMUP requires both save and load enabled")
    if worker.save_workers != 1 or worker.load_workers != 1:
        raise RuntimeError("OFFLOAD_WARMUP requires one save and one load worker")
    backends = set(engine.storage_manager.storage_backends)
    if backends != {"LocalCPUBackend"}:
        raise RuntimeError("OFFLOAD_WARMUP requires only LocalCPUBackend")
    if engine.async_loading or engine.save_only_first_rank:
        raise RuntimeError("OFFLOAD_WARMUP requires synchronous per-rank retrieval")

    gpu = engine.gpu_connector
    if gpu.release_gpu_staging_after_transfer:
        raise RuntimeError("OFFLOAD_WARMUP requires persistent GPU staging buffers")
    chunk_size = int(worker.chunk_size)
    virtual_block_size = int(worker.virtual_block_size)
    if chunk_size <= 0 or chunk_size % virtual_block_size:
        raise RuntimeError("OFFLOAD_WARMUP requires block-aligned LMCache chunks")
    blocks_per_chunk = chunk_size // virtual_block_size
    # Warm a full group plus its tail, in addition to the first single chunk.
    # C48 TP8/DCP8: 1 then 33 chunks, with 32 staging chunks (~56 MiB/rank).
    sizes = (1, int(gpu.gpu_staging_buffer_chunks) + 1)
    max_blocks = sizes[-1] * blocks_per_chunk
    if 1 + 2 * max_blocks > codec.num_blocks:
        raise RuntimeError("OFFLOAD_WARMUP needs two disjoint scratch KV block ranges")

    rank = worker._rank
    started = time.perf_counter()
    logger.info(
        "[OFFLOAD-WARMUP] rank=%s start chunks=%s backend=LocalCPUBackend",
        rank,
        sizes,
    )
    # Graph-capture work must finish before a different stream reads its blocks.
    if codec.device.type == "cuda":
        torch.cuda.synchronize(codec.device)

    def save(tokens, source_blocks, req_id):
        engine.store(tokens, block_ids=source_blocks, req_id=req_id)
        # LocalCPUBackend publishes synchronously. A partial/disabled store
        # must fail here, rather than turn into a successful empty warmup.
        hit = engine.lookup(tokens, search_range=["LocalCPUBackend"], pin=False)
        if hit != len(tokens):
            raise RuntimeError(f"OFFLOAD_WARMUP stored only {hit}/{len(tokens)} tokens")

    def load(tokens, target_blocks, req_id, expected_bytes):
        # Read stats on this thread: both staging and accounting are TLS.
        gpu.reset_transfer_stats()
        mask = engine.retrieve(tokens, block_ids=target_blocks, req_id=req_id)
        stats = gpu.last_transfer_stats()
        if len(mask) != len(tokens) or not bool(mask.all().item()):
            raise RuntimeError("OFFLOAD_WARMUP CPU restore missed tokens")
        if int(stats.get("completed_bytes", 0)) != expected_bytes or not stats.get(
            "transfer_succeeded"
        ):
            raise RuntimeError("OFFLOAD_WARMUP did not execute the expected H2D copy")
        return stats

    for chunks in sizes:
        nblocks = chunks * blocks_per_chunk
        ntokens = chunks * chunk_size
        # Negative synthetic tokens cannot collide with real tokenizer IDs.
        # A random prefix also separates restarts. Tokens are only hashed, never
        # fed through the model. clear(tokens=...) removes just this round.
        prefix = -(uuid.uuid4().int % (2**62)) - 1
        tokens = tokens_to_tensor([prefix] + [-1] * (ntokens - 1))
        req_id = f"offload-warmup-{rank}-{chunks}-{uuid.uuid4().hex}"
        source_blocks = list(range(1, 1 + nblocks))  # Leave null block 0 alone.
        target_blocks = list(range(1 + max_blocks, 1 + max_blocks + nblocks))
        expected_bytes = nblocks * codec.bytes_per_block

        try:
            store_start = time.perf_counter()
            worker._save_executor.submit(save, tokens, source_blocks, req_id).result()
            load_start = time.perf_counter()
            stats = worker._load_executor.submit(
                load, tokens, target_blocks, req_id, expected_bytes
            ).result()
            load_end = time.perf_counter()
            logger.info(
                "[OFFLOAD-WARMUP] rank=%s chunks=%d tokens=%d bytes=%d "
                "groups=%d store_ms=%.2f load_ms=%.2f status=ok",
                rank,
                chunks,
                ntokens,
                expected_bytes,
                int(stats.get("groups", 0)),
                (load_start - store_start) * 1000,
                (load_end - load_start) * 1000,
            )
        finally:
            engine.clear(tokens=tokens, locations=["LocalCPUBackend"])
        if engine.lookup(tokens, search_range=["LocalCPUBackend"], pin=False) != 0:
            raise RuntimeError("OFFLOAD_WARMUP could not remove scratch CPU entries")

    worker._offload_warmed = True
    logger.info(
        "[OFFLOAD-WARMUP] rank=%s complete rounds=%d total_ms=%.2f",
        rank,
        len(sizes),
        (time.perf_counter() - started) * 1000,
    )
