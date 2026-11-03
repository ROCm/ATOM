# SPDX-License-Identifier: MIT
"""The request buckets DeepSeek-V4.1's decoder is recorded at (`DecoderGraph`).

A prefill step lays its decoder tail out for the step's `running_bs`
(`DeepseekV41MetadataBuilder._publish_decoder`), and `ForwardMode.decide`
settles `running_bs` by rounding the DP group's batch UP the cudagraph capture
ladder. So the tail only ever runs at a capture size: a recording at any other
size is never replayed, and a capture size without a recording always runs
eagerly. With the old fixed default (1, 2, 3, 4, 8, 16) on the usual ladder
(1..8, 12, 16, ...), steps of 5, 6, 7 and 9..12 requests ran their decoder
eagerly.

Kept free of torch and aiter so the CPU test gate can import it.
"""

from collections.abc import Iterable

# The default reaches as far as the old fixed default did: a tail of more
# requests is GPU-bound, and each larger recording grows the graph pool.
DECODER_GRAPH_DEFAULT_MAX_BS = 16


def decoder_graph_sizes(
    configured: Iterable[int] | None,
    capture_sizes: Iterable[int],
    *,
    max_bs: int,
    max_tokens: int,
    window: int,
) -> tuple[int, ...]:
    """The `running_bs` to record the decoder at, ascending.

    ``configured``: `ATOM_DECODER_GRAPH_SIZES`, or None when it is unset, which
    takes every capture size up to `DECODER_GRAPH_DEFAULT_MAX_BS` -- exactly the
    `running_bs` a prefill step can settle on. An explicit list is honored as
    given (an empty one records nothing).

    Either way a size must fit a step: no more requests than ``max_bs``, one
    ``window`` each within the ``max_tokens`` token buffer
    (`build_for_req_graph_capture`).
    """
    if configured is None:
        configured = (n for n in capture_sizes if 0 < n <= DECODER_GRAPH_DEFAULT_MAX_BS)
    return tuple(
        sorted({n for n in configured if 0 < n <= max_bs and n * window <= max_tokens})
    )
