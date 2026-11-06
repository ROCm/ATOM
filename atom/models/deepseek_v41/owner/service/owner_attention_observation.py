"""Bound private observations by actual CSR reader capacity, never arena size.

This module only observes. It must stay outside clean cost graphs.
"""

MAX_OBSERVATION_BYTES = 64 * 1024 * 1024
ATTENTION_OBSERVATION_LAYERS = (2, 8, 14, 20)


def observe_bounded_tensor(slot, key, tensor):
    import torch

    if tensor.numel() * tensor.element_size() > MAX_OBSERVATION_BYTES:
        raise MemoryError("Bound each private attention observation to 64 MiB")
    if key not in slot:
        slot[key] = torch.empty_like(tensor)
    elif slot[key].shape != tensor.shape or slot[key].dtype != tensor.dtype:
        raise ValueError("Observation shape/dtype must remain fixed across replay")
    slot[key].copy_(tensor)


def observe_window_readers(ring, ids, indptr, observe):
    """Gather only CSR-addressed rows with a fixed, graph-safe GPU shape.

    ring_view is the trailing BF16 arena. IDs index that view, including native
    layer strides. CSR capacity beyond indptr[-1], sentinels and out-of-range
    addresses are masked BEFORE gather. No host read or dynamic compaction.
    """
    import torch

    if ring.ndim != 2 or ring.shape[0] == 0:
        raise ValueError("Bind the native nonempty two-dimensional ring view")
    if ids.ndim != 1 or indptr.ndim != 1 or indptr.numel() == 0:
        raise ValueError("Bind native one-dimensional CSR IDs and indptr")
    if ids.dtype not in (torch.int32, torch.int64) or indptr.dtype != ids.dtype:
        raise ValueError("CSR IDs and indptr must share a native integer dtype")
    if ids.device != ring.device or indptr.device != ring.device:
        raise ValueError("CSR metadata and ring must share a device")
    payload = ids.numel() * ring.shape[1] * ring.element_size()
    if payload > MAX_OBSERVATION_BYTES:
        raise MemoryError("Window reader capacity exceeds 64 MiB before gather")
    ordinal = torch.arange(ids.numel(), device=ids.device)
    valid = (ordinal < indptr[-1]) & (ids >= 0) & (ids < ring.shape[0])
    safe_ids = torch.where(valid, ids, 0).long()
    values = ring.index_select(0, safe_ids)
    values.masked_fill_(~valid[:, None], 0)
    observe("window_reader_ids", torch.where(valid, ids, -1))
    observe("window_reader_valid", valid)
    observe("window_reader_values", values)


def observe_prefix_readers(pool, ids, indptr, observe):
    """Keep every active prefix row; split observations into <=64MiB chunks."""
    import torch

    if (
        pool.ndim != 2
        or pool.shape[0] == 0
        or ids.ndim != 1
        or indptr.ndim != 1
        or not indptr.numel()
    ):
        raise ValueError("Bind native prefix pool and one-dimensional CSR")
    if ids.dtype not in (torch.int32, torch.int64) or indptr.dtype != ids.dtype:
        raise ValueError("Preserve native prefix integer addresses")
    if pool.device != ids.device or ids.device != indptr.device:
        raise ValueError("Prefix CSR and pool must share a device")
    row_bytes = pool.shape[1] * pool.element_size()
    rows_per_chunk = MAX_OBSERVATION_BYTES // row_bytes
    if not rows_per_chunk or ids.numel() > 192 * 512:
        raise MemoryError("Bound owner192 prefix reserve to the actual top512 contract")
    ordinal = torch.arange(ids.numel(), device=ids.device)
    valid = (ordinal < indptr[-1]) & (ids >= 0) & (ids < pool.shape[0])
    observe("prefix_reader_ids", torch.where(valid, ids, -1))
    observe("prefix_reader_valid", valid)
    for chunk, start in enumerate(range(0, ids.numel(), rows_per_chunk)):
        selected, selected_valid = (
            ids[start : start + rows_per_chunk],
            valid[start : start + rows_per_chunk],
        )
        safe = torch.where(selected_valid, selected, 0).long()
        values = pool.index_select(0, safe)
        values.masked_fill_(~selected_valid[:, None], 0)
        observe(f"prefix_reader_values_chunk{chunk}", values)
