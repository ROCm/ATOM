# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

"""Reading checkpoint tensors off disk.

Split out of `loader.py`, which imports AITER at module level: the unit test
gate has no AITER build, and the shard-skipping logic here is worth covering
against real files.
"""

import atexit
import concurrent.futures
import itertools
import json
import logging
import os
import threading
import time
from collections.abc import Callable, Generator
from glob import glob

import safetensors
import safetensors.torch
import torch
from tqdm import tqdm
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from atom.model_loader.weight_utils import (
    download_weights_from_hf,
    filter_duplicate_safetensors_files,
)
from atom.utils import envs

logger = logging.getLogger("atom")

# safetensors<=0.7.0 ships a Python `_TYPES` dict missing the `F8_E8M0`
# (MX scale) entry, even though both torch and the safetensors-rust binary
# support it. The mmap'd `safe_open` path goes through Rust and works, but
# the `safetensors.torch.load(bytes)` path used when `ATOM_DISABLE_MMAP=true`
# raises `KeyError: 'F8_E8M0'` on DeepSeek-V4-Pro shards. Register the
# missing dtype string so both paths behave identically.
if "F8_E8M0" not in safetensors.torch._TYPES and hasattr(torch, "float8_e8m0fnu"):
    safetensors.torch._TYPES["F8_E8M0"] = torch.float8_e8m0fnu


_MAX_SAFETENSORS_HEADER_BYTES = 100 * 1024 * 1024


def _shard_tensor_names(st_file: str) -> list[str] | None:
    """Tensor names in a safetensors file, from its header alone.

    The header is a little-endian u64 byte count followed by that much JSON, so
    this costs one small read and never touches the tensor data.

    Returns None if the header cannot be read, so the caller loads the shard
    anyway and the real reader produces the real diagnostic -- a truncated or
    corrupt file should not be reported as a JSON error from a fast path whose
    only job is to decide whether the file is worth opening.
    """
    try:
        with open(st_file, "rb") as f:
            raw_len = f.read(8)
            if len(raw_len) != 8:
                return None
            header_len = int.from_bytes(raw_len, "little")
            if not 0 < header_len <= _MAX_SAFETENSORS_HEADER_BYTES:
                return None
            raw_header = f.read(header_len)
            if len(raw_header) != header_len:
                return None
            header = json.loads(raw_header)
    except (OSError, ValueError):
        return None
    if not isinstance(header, dict):
        return None
    return [name for name in header if name != "__metadata__"]


def _node_local_rank() -> tuple[int, int]:
    """This process's rank and world size *within its node*.

    Page cache is per node, so a prefetch split has to be node-local: splitting
    by global rank would hand each node only 1/world of the shards and leave
    the rest of that node's checkpoint to be faulted in on demand.

    Falls back to ``(0, 1)`` -- every rank prefetches everything -- whenever the
    node layout cannot be established. That errs the safe way: duplicate reads
    are absorbed by the shared page cache, whereas over-estimating the node
    count would silently leave part of the checkpoint unprefetched.
    """
    env_rank, env_size = os.environ.get("LOCAL_RANK"), os.environ.get(
        "LOCAL_WORLD_SIZE"
    )
    if env_rank is not None and env_size is not None:
        try:
            return int(env_rank), max(1, int(env_size))
        except ValueError:
            pass
    if not torch.distributed.is_initialized():
        return 0, 1
    world = torch.distributed.get_world_size()
    try:
        visible = torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        visible = 0
    if visible and world <= visible:
        # Every rank fits on this node's GPUs, so global rank is node-local.
        return torch.distributed.get_rank(), world
    return 0, 1


# Set when prefetching is no longer useful -- the load finished, or the process
# is exiting. `ThreadPoolExecutor`'s workers are not daemon threads and are
# joined by `concurrent.futures`' own atexit hook, so the `daemon=True` on the
# thread below cannot by itself keep a failed load from blocking process exit
# behind an in-flight multi-GB read. The read loop has to bail out on its own.
_prefetch_stop = threading.Event()

# Registered after `concurrent.futures.thread` imported (and registered
# `_python_exit`), so LIFO ordering runs this first and the reads are already
# unwinding by the time the executor is joined.
atexit.register(_prefetch_stop.set)


def _read_whole_file(path: str, block_size: int) -> None:
    """Read `path` sequentially so the kernel caches it, discarding the data.

    Abandons the file as soon as `_prefetch_stop` is set: the pages read so far
    stay cached, and warming the rest is worthless once nobody is loading.
    """
    with open(path, "rb") as f:
        while not _prefetch_stop.is_set() and f.read(block_size):
            pass


def _start_prefetch(files: list[str], num_threads: int, block_size: int) -> None:
    """Warm the page cache for this rank's share of `files`, in the background.

    A plain sequential ``read()`` rather than ``posix_fadvise(WILLNEED)``:
    WILLNEED is a hint the kernel may drop, and for a 350 GiB checkpoint it
    drops most of it, leaving the real work to demand faults through the mmap.
    Measured on a local NVMe here, that fault pattern sustains 3.2 GB/s while
    the device does 6.9 GB/s and even a *single* sequential reader reaches
    6.06 GB/s -- so the gap is the access pattern, not queue depth, and a
    sequential reader is exactly what closes it.

    Detached on purpose: loading starts immediately and rides whatever is
    already cached instead of waiting for the whole checkpoint.
    """
    rank, local_world = _node_local_rank()
    mine = files[rank::local_world]
    if not mine:
        return
    # A thread count of zero is the obvious way to try to switch this off, and
    # `ThreadPoolExecutor` rejects it -- from inside the detached thread below,
    # where it would surface only as a bare traceback. `ATOM_LOADER_PREFETCH`
    # is the documented off switch; treat 0 as "as few as possible" instead.
    num_threads = max(1, num_threads)
    _prefetch_stop.clear()

    def _run() -> None:
        started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as pool:
            remaining = iter(mine)
            pending = {
                pool.submit(_read_whole_file, p, block_size)
                for p in itertools.islice(remaining, num_threads)
            }
            # Bounded window: the point is to stay ahead of the loader, not to
            # queue every shard at once and compete with it for bandwidth.
            while pending:
                done, pending = concurrent.futures.wait(
                    pending, return_when=concurrent.futures.FIRST_COMPLETED
                )
                for fut in done:
                    try:
                        fut.result()
                    except OSError as e:
                        # A prefetch failure costs speed, never correctness:
                        # the loader still reads the shard itself.
                        logger.debug("Prefetch failed: %s", e)
                    nxt = next(remaining, None)
                    if nxt is not None and not _prefetch_stop.is_set():
                        pending.add(pool.submit(_read_whole_file, nxt, block_size))
        logger.info(
            "Checkpoint prefetch finished: %d/%d shards in %.1fs",
            len(mine),
            len(files),
            time.perf_counter() - started,
        )

    logger.info(
        "Prefetching %d/%d shards into page cache in the background "
        "(node-local rank %d of %d, %d threads, %d MiB blocks)",
        len(mine),
        len(files),
        rank,
        local_world,
        num_threads,
        block_size // (1024 * 1024),
    )
    threading.Thread(target=_run, name="ckpt-prefetch", daemon=True).start()


def _shards_worth_reading(
    files: list[str], wants: Callable[[str], bool] | None
) -> list[str]:
    """Drop shards holding nothing the caller wants, by header alone.

    A drafter load reads the target's checkpoint to pick out the MTP block --
    typically one shard out of dozens -- so this is what keeps both the loader
    and the prefetcher off the other 98%.

    A shard whose header cannot be read is kept: the real reader should produce
    the real diagnostic, not a fast path whose only job is to skip files.
    """
    if wants is None:
        return files
    kept = []
    for st_file in files:
        names = _shard_tensor_names(st_file)
        if names is None or any(map(wants, names)):
            kept.append(st_file)
    return kept


def safetensors_weights_iterator(
    model_name_or_path: str,
    disable_mmap: bool = False,
    wants: Callable[[str], bool] | None = None,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Iterate over the weights in the model safetensor files.

    `wants` lets the caller reject a tensor by name before it is materialized.
    Without it every tensor of every shard is built and then thrown away by the
    caller -- which is what a drafter load does, since it reads the target's
    checkpoint to pick out the MTP block and discards the other ~98%.
    """
    logger.info(f"disable_mmap: {disable_mmap}")
    path = (
        model_name_or_path
        if os.path.isdir(model_name_or_path)
        else download_weights_from_hf(
            model_name_or_path, None, ["*.safetensors"], ignore_patterns=["original/*"]
        )
    )
    hf_weights_files = filter_duplicate_safetensors_files(
        glob(os.path.join(path, "*.safetensors")), path, SAFE_WEIGHTS_INDEX_NAME
    )
    hf_weights_files = _shards_worth_reading(hf_weights_files, wants)
    enable_tqdm = (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )

    # The prefetcher and the WILLNEED hint do the same job, and doing both is
    # worse than either: the hint asks the kernel to read-ahead 350 GiB of
    # random-ish ranges while the prefetcher is streaming the same files
    # sequentially, so they compete for the same device.
    prefetching = envs.ATOM_LOADER_PREFETCH and not disable_mmap
    if prefetching:
        # Sort only now, and read in the same order below. `glob` returns
        # directory order, which is stable enough within one run but is not a
        # promise, and the ranks must agree exactly or the stride partition
        # leaves shards unclaimed. Reading in that same order also keeps the
        # loader trailing the prefetchers instead of racing ahead of them.
        hf_weights_files = sorted(hf_weights_files)
        _start_prefetch(
            hf_weights_files,
            envs.ATOM_LOADER_PREFETCH_THREADS,
            envs.ATOM_LOADER_PREFETCH_BLOCK_MB * 1024 * 1024,
        )

    iters = tqdm(
        hf_weights_files,
        desc=f"Loading safetensors shards[{model_name_or_path}]",
        disable=not enable_tqdm,
    )
    try:
        yield from _iter_shards(iters, disable_mmap, prefetching, wants)
    finally:
        # Whether the caller drained this or abandoned it, nothing is going to
        # read these files again, so stop warming the cache for them. Without
        # this a load that raises leaves the prefetcher streaming the rest of
        # the checkpoint and the process cannot exit until it finishes.
        _prefetch_stop.set()


def _iter_shards(
    iters, disable_mmap: bool, prefetching: bool, wants: Callable[[str], bool] | None
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Yield every wanted tensor of each shard, in the order given."""
    for st_file in iters:
        # Advise kernel for sequential read-ahead (mmap optimization)
        if (
            not prefetching
            and envs.ATOM_LOADER_FADVISE
            and not disable_mmap
            and hasattr(os, "posix_fadvise")
        ):
            try:
                fd = os.open(st_file, os.O_RDONLY)
                file_size = os.fstat(fd).st_size
                os.posix_fadvise(
                    fd,
                    0,
                    file_size,
                    os.POSIX_FADV_SEQUENTIAL | os.POSIX_FADV_WILLNEED,
                )
                os.close(fd)
            except OSError:
                pass

        if disable_mmap:
            # `safetensors.torch.load` has no partial API, so a shard that
            # holds anything wanted is still deserialized whole.
            with open(st_file, "rb") as f:
                result = safetensors.torch.load(f.read())
                for name, param in result.items():
                    if wants is None or wants(name):
                        yield name, param
        else:
            with safetensors.safe_open(st_file, framework="pt", device="cpu") as f:
                # `.keys()` is not redundant here: `safe_open` is a Rust object
                # with no `__iter__`, so iterating it directly raises TypeError.
                for name in f.keys():  # noqa: SIM118
                    if wants is None or wants(name):
                        yield name, f.get_tensor(name)
