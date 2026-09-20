# SPDX-License-Identifier: MIT
"""The torch body `unit_table`'s kernel replaced, kept as its oracle.

It lives beside the test rather than beside the kernel because nothing in
`atom` calls it: a reference in the production op reads like a second
implementation a caller might pick, and there is no such caller.
"""

import torch


def unit_table_reference(block_tables, batch_ids, units_per_page):
    """Pure-torch equivalent of :func:`atom.model_ops.deepseek_v41.unit_table`."""
    pages = block_tables[batch_ids.clamp_min(0).long()]
    tiles = torch.arange(
        units_per_page, dtype=block_tables.dtype, device=block_tables.device
    )
    table = (pages[..., None] * units_per_page + tiles).flatten(-2)
    return table.masked_fill((batch_ids < 0)[:, None], 0).int()
