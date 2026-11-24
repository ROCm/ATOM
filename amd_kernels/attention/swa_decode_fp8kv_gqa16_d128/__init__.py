# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the amd-kernels project
"""MiniMax-M3.1 DSpark draft decode attention over fp8 pages (FlyDSL, gfx950)."""

NW = 4  # waves per workgroup
CH = 32  # keys per wave step
MAX_QL = 8


def max_window(ql: int) -> int:
    """Largest window for ``ql`` query rows: each wave has 64 page-lookup lanes,
    including query rows and chunk alignment. Rows <= 6 keep the 6-row bound."""
    return 64 * CH * NW - max(ql, 6) - CH + 2


# The bound for 1..6 rows (backend selection; 7..8 rows check max_window(ql)).
MAX_WINDOW = max_window(6)


def layer_blocker(head_dim: int, num_heads: int, num_kv_heads: int, window: int | None) -> str | None:
    """Why a sliding-window layer (this rank's heads) does not fit the kernel at up to ``MAX_QL`` rows (None: it
    fits)."""
    if head_dim != 128:
        return f"head_dim {head_dim}"
    if num_heads != 16 * num_kv_heads:
        return f"{num_heads} query / {num_kv_heads} KV heads"
    if window is None or not 0 < window <= max_window(MAX_QL):
        return f"window {window}"
    return None

__all__ = ["layer_blocker", "swa_decode_fp8kv", "supports_swa_decode_fp8kv", "swa_decode_fp8kv_oproj_input"]


def __getattr__(name):
    if name == "swa_decode_fp8kv_oproj_input":
        from . import oproj_input

        return oproj_input.swa_decode_fp8kv_oproj_input
    if name in __all__:
        from . import host

        return getattr(host, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
