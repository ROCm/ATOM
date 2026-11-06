"""Per-rank chunk summation-order codes for the TP4 ordered BF16 reduce-scatter.

Each 128-element output chunk carries the order in which the four rank
contributions are summed; six fixed patterns per rank repeat over contiguous
segments, so the reduction is bitwise reproducible across graph replays.
"""

import torch

PATTERNS = (
    (45, 57, 45, 57, 30, 30),
    (78, 108, 120, 78, 108, 120),
    (180, 141, 156, 156, 180, 141),
    (216, 216, 201, 228, 201, 228),
)


def make_codes(rows, rank, device="cpu"):
    assert (rows == 1536 or 4096 <= rows <= 16384) and rows % 4 == 0 and 0 <= rank < 4
    chunks = rows * 10  # outputnumel /128: rows/4 *5120 /128
    segment = ((chunks + 48 * 16 - 1) // (48 * 16)) * 16
    index = torch.arange(chunks, device=device)
    patterns = torch.tensor(PATTERNS[rank], device=device, dtype=torch.uint8)
    return patterns[(index // segment) % 6], segment
