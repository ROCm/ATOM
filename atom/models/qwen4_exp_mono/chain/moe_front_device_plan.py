"""Integer ownership for a single-wave compact M16 task publisher."""

ROWS, TOPK, EXPERTS, BM = 4, 11, 513, 16
LANES = 64
SORT_STEPS = tuple(
    (width, distance)
    for width in (2, 4, 8, 16, 32, 64)
    for distance in (32, 16, 8, 4, 2, 1)
    if distance < width
)


def key(eid, token, route):
    return eid * 64 + token * 16 + route


def native_input_offset(pick):
    return (pick // 16) * TOPK + pick % 16


def sorted_offset(tile, row):
    return tile * BM + row
