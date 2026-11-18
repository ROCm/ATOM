"""Pure-Python launch policy for the MiniMax-M3 decode scorer."""


def decode_score_num_warps(
    arch: str,
    batch: int,
    max_block: int,
    max_query_len: int,
    num_idx_heads: int = 1,
) -> int:
    if (
        arch == "gfx1250"
        and batch >= 4
        and max_block > 512
        and (
            max_query_len in (1, 4)
            or (max_query_len == 8 and num_idx_heads == 1 and 8 <= batch <= 16)
        )
    ):
        return 2
    return 4


def use_transposed_decode_score(
    arch: str, batch: int, max_block: int, max_query_len: int
) -> bool:
    if arch != "gfx1250" or max_query_len not in (1, 4) or batch <= 0 or max_block <= 0:
        return False
    work = batch * max_block
    if max_query_len == 1:
        return work >= 12_000
    return (max_block >= 1_000 and batch >= 4) or (max_block >= 500 and work >= 8_000)
