# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""M3 PAGE identities and head runs, independent of the local PP pool order."""

from dataclasses import replace

from atom.model_ops.minimax_m3.cache_layout import ASM_PAGE_SIZE


def configure_m3_kv_transfer_tensors(
    transfer_tensors,
    builder,
    *,
    num_layers: int,
    sparse_layers: set[int],
    index_fp4: bool,
    sparse_nvfp4: bool,
    sparse_mxfp4: bool = False,
    computing_index_layers: set[int] | None = None,
) -> None:
    """Publish field-major/global-layer pages, retaining the backend's byte views.

    NVFP4 splits sparse KV into a second pool; the first *local* pool holds all
    index rows, even if those layers' KV lives in the other pool. Neither pool
    order nor its local row number identifies a consumer region under PP.
    Canonical field/layer order makes each stage a subset of the full decode
    layout, including stages with no sparse layers and the last stage's draft.
    """
    pools = builder.kv_pools
    if not pools:
        raise ValueError("MiniMax-M3 transfer requires KV pools")
    dtypes = {pool.kv_dtype for pool in pools.values()}
    if len(dtypes) != 1:
        raise ValueError("MiniMax-M3 KV pools must share the configured KV dtype")
    dtype = next(iter(dtypes))
    layers = list(range(num_layers))
    if not sparse_layers <= set(layers):
        raise ValueError("MiniMax-M3 sparse layers exceed the model's layer range")
    if computing_index_layers is None:
        computing_index_layers = sparse_layers
    if not computing_index_layers <= sparse_layers:
        raise ValueError("MiniMax-M3 index rows must belong to sparse layers")
    indexed = sorted(computing_index_layers)
    if sparse_nvfp4 and sparse_mxfp4:
        raise ValueError("MiniMax-M3 sparse KV formats are mutually exclusive")
    scale_layers = [
        layer
        for layer in layers
        if (layer not in sparse_layers and dtype.itemsize == 1)
        or (
            layer in sparse_layers
            and not sparse_mxfp4
            and (dtype.itemsize == 1 or sparse_nvfp4)
        )
    ]
    fields = (
        ("k", layers),
        ("v", layers),
        ("k_scale", scale_layers),
        ("v_scale", scale_layers),
        ("index", indexed),
        ("index_scale", indexed if index_fp4 else []),
    )
    consumer_indices = {
        (layer, field): i
        for i, (field, layer) in enumerate(
            (field, layer) for field, held in fields for layer in held
        )
    }

    def row_layers(rows):
        ordered = sorted(rows.items(), key=lambda item: item[1])
        if [row for _, row in ordered] != list(range(len(ordered))):
            raise ValueError("MiniMax-M3 pool rows must be contiguous")
        return [module.layer_num for module, _ in ordered]

    # This is the backend's independent index row space, shared by all pools.
    index_layers = row_layers(builder.pool_rows.get("index", {}))
    native_layout = []
    local_layers = []
    block_size = builder.model_runner.block_size
    for geometry, pool in pools.items():
        held = row_layers(builder.pool_rows[geometry])
        local_layers.extend(held)
        for layer in held:
            expected_format = (
                ("mxfp4" if sparse_mxfp4 else "nvfp4" if sparse_nvfp4 else "")
                if layer in sparse_layers
                else ""
            )
            if pool.kv_format != expected_format:
                raise ValueError(f"MiniMax-M3 layer {layer} has unexpected KV format")
        for group in pool.field_groups:
            for field in group:
                per_head = field in pool.cache_fields or field in pool.scale_fields
                field_layers = held if per_head else index_layers
                if field.layers != len(field_layers):
                    raise ValueError(f"MiniMax-M3 {field.name} row count mismatch")
                for row, layer in enumerate(field_layers):
                    runs = None
                    if per_head:
                        # NVFP4: [page, head, token, packed_dim]. Sparse FP8/BF16
                        # and the draft use smaller page-16 SHUFFLE head runs.
                        page_tokens = (
                            ASM_PAGE_SIZE
                            if layer in sparse_layers
                            and pool.kv_format not in ("nvfp4", "mxfp4")
                            else pool.block_size
                        )
                        if page_tokens <= 0 or block_size % page_tokens:
                            raise ValueError("MiniMax-M3 head runs must divide a block")
                        runs = (block_size // page_tokens, pool.num_kv_heads)
                    native_layout.append(
                        (
                            f"mha.{geometry}.{field.name}.layer_{row}",
                            layer,
                            field.name,
                            runs,
                        )
                    )
    if len(set(local_layers)) != len(local_layers):
        raise ValueError("MiniMax-M3 layers cannot occupy multiple KV pools")
    if set(index_layers) != set(local_layers) & computing_index_layers:
        raise ValueError(
            "MiniMax-M3 index rows do not match the local computing layers"
        )

    pages = {}
    for page, (native_role, layer, field, runs) in zip(
        transfer_tensors.pages, native_layout, strict=True
    ):
        if page.region.semantic_role != native_role:
            raise RuntimeError(
                f"MiniMax-M3 transfer expected {native_role!r}, got "
                f"{page.region.semantic_role!r}"
            )
        key = (layer, field)
        if key not in consumer_indices or key in pages:
            raise ValueError(f"Unexpected or duplicate MiniMax-M3 PAGE {key}")
        pages[key] = replace(
            page,
            region=replace(page.region, semantic_role=f"m3.layer_{layer}.{field}"),
            head_major_runs=runs,
        )
    local_layers = set(local_layers)
    expected = {key for key in consumer_indices if key[0] in local_layers}
    if pages.keys() != expected:
        raise ValueError("MiniMax-M3 PAGE fields do not cover the local layer layout")
    ordered = sorted(pages, key=consumer_indices.__getitem__)
    transfer_tensors.pages = [pages[key] for key in ordered]
    transfer_tensors.block_region_consumer_indices = [
        consumer_indices[key] for key in ordered
    ]
    transfer_tensors.block_region_consumer_count = len(consumer_indices)
