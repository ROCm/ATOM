# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Token copy ranges from an unsharded prefill to a DCP decode cache."""


def plan_dcp_copy_spans(
    source_blocks: list[int],
    destination_blocks: list[int],
    *,
    num_tokens: int,
    block_size: int,
    destination_block_size: int | None = None,
    dcp_size: int,
    dcp_rank: int,
    interleave_size: int,
) -> list[tuple[int, int, int, int, int]]:
    """Return (source block, offset, destination block, offset, token count).

    Source blocks cover the entire prefilled prefix. Destination blocks are
    its uncached suffix, after removing lookahead allocation. Offsets and
    lengths are in tokens, independent of a registered layer's byte stride.
    """
    if destination_block_size is None:
        destination_block_size = block_size
    if (
        block_size <= 0
        or dcp_size <= 0
        or not 0 <= dcp_rank < dcp_size
        or interleave_size <= 0
        or block_size % interleave_size
        or destination_block_size <= 0
        or destination_block_size % interleave_size
        or num_tokens < 0
    ):
        raise ValueError("Invalid DCP token layout")
    if any(block < 0 for block in source_blocks + destination_blocks):
        raise ValueError("DCP block IDs must be nonnegative")
    if not destination_blocks:
        return []
    if num_tokens > len(source_blocks) * block_size:
        raise ValueError("Source blocks do not cover the prefilled prefix")
    total_destination_blocks = (num_tokens + destination_block_size * dcp_size - 1) // (
        destination_block_size * dcp_size
    )
    first_destination_block = total_destination_blocks - len(destination_blocks)
    if first_destination_block < 0:
        raise ValueError("Destination includes blocks beyond the prefilled prefix")

    spans: list[tuple[int, int, int, int, int]] = []
    for index, destination_block in enumerate(destination_blocks):
        offset = 0
        while offset < destination_block_size:
            local_position = (
                first_destination_block + index
            ) * destination_block_size + offset
            global_position = (
                local_position // interleave_size * dcp_size + dcp_rank
            ) * interleave_size + local_position % interleave_size
            if global_position >= num_tokens:
                break
            source_index, source_offset = divmod(global_position, block_size)
            length = min(
                interleave_size - local_position % interleave_size,
                block_size - source_offset,
                destination_block_size - offset,
                num_tokens - global_position,
            )
            source_block = source_blocks[source_index]
            if spans and spans[-1][:1] == (source_block,):
                previous_source, previous_offset, previous_dest, dest_offset, count = (
                    spans[-1]
                )
                if (
                    previous_offset + count == source_offset
                    and previous_dest == destination_block
                    and dest_offset + count == offset
                ):
                    spans[-1] = (
                        previous_source,
                        previous_offset,
                        previous_dest,
                        dest_offset,
                        count + length,
                    )
                    offset += length
                    continue
            spans.append(
                (source_block, source_offset, destination_block, offset, length)
            )
            offset += length
    # shortcut: interleave=1 emits token-sized spans; stage GPU gathers if
    # descriptor construction or submission limits throughput.
    return spans
