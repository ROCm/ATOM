# SPDX-License-Identifier: MIT
"""`model_options.seg_fuse`: opt-in launch fusion of V4.1's non-MoE decode segment.

A list of item numbers; empty (the default) leaves every path as it was (no
op is registered, no AITER module is loaded). Each item replaces two
back-to-back launches by one whose output is bit for bit the pair's:

  1  the FFN seam's Mega-MHC also writes the MXFP8 pair of its normed output
     (`mega_mhc_kernel<..., MXFP8=true, POST=true>`, the instantiation the
     attention seam already runs) and the shared expert's `gate_up_proj` reads
     that pair: its own `quantize_fp8` launch goes away.
  7  the shared expert's `silu_and_mul` and the `quantize_fp8` in front of
     `w2` become AITER's `silu_mul_mxfp8` (module_dsv41_seg_fuse, gfx950).

Both need native MXFP8 `gate_up_proj` / `w2`; anywhere else the item is
ignored. How the pair of item 1 reaches the shared expert depends on who runs
the FFN sublayer (`v41.block.ffn`):

  - ATOM itself, inline in the compiled graph: `MoE.forward_quant` hands it
    down as an argument, where the shared expert runs on the layer's own
    stream (no dual stream, no comm-fused dispatch).
  - a plugin plan (one opaque op, e.g. a MegaMoE provider that calls the
    module's own `single_stream_moe_forward`, or its shared expert and routed
    experts one after the other, on the BF16 rows): the op takes the pair as
    two more inputs and holds it on the shared expert for the length of the
    call (`PluginExpert.seg_fuse_input`); the expert uses it only for the very
    tensor it was written with, whichever dispatch reaches it. The op's body
    is the only writer, so a compiled graph orders it with the call.

Item 1's Mega-MHC launch is the one `model_options.ffn_mhc_fuse_quant` asks
for; that option parks the pair on the MoE module instead, where only
`single_stream_moe_forward` reads it. Item 7 is another kernel than
`model_options.shared_act_quant`'s. A recipe sets one of each pair.
"""

import logging

import torch

from atom import plugin_host

logger = logging.getLogger("atom")

KNOWN_ITEMS = frozenset({1, 7})


def enabled(item: int) -> bool:
    return item in plugin_host.model_options().seg_fuse


def act_quant_supported(width: int) -> bool:
    """Whether `silu_mul_mxfp8` runs a shared expert of this per-rank width."""
    from aiter.jit.utils.chip_info import get_gfx_runtime

    return get_gfx_runtime() == "gfx950" and width > 0 and width % 32 == 0


# Calls of the opaque FFN op that offered the pair, and those whose shared
# expert read it (decisions, not graph replays): what a boot check reads.
COUNTS = {"ffn_pair_offered": 0, "ffn_pair_taken": 0}
_logged: set[str] = set()


def _log_once(key: str, message: str, *args) -> None:
    if key not in _logged:
        _logged.add(key)
        logger.info(message, *args)


def _shared(moe):
    return getattr(moe, "shared_experts", None)


def _native_mx(linear) -> bool:
    return getattr(linear, "native_a8_group_rows", None) is not None


def bind_ffn_quant(block):
    """At block build, item 1: `ffn(hidden, hidden_fp8, hidden_scale)` as
    `Block.ffn_forward` calls it with the FFN seam's fused norm output, or
    None (item off, or no native MXFP8 shared expert to read the pair)."""
    moe = block.ffn
    shared = _shared(moe)
    if not enabled(1) or shared is None or not _native_mx(shared.gate_up_proj):
        return None
    if block._ffn == moe.__call__:
        # No plugin owns the seam: the module's forward, inline.
        return moe.forward_quant

    ffn = block._ffn

    def run(hidden, hidden_fp8, hidden_scale):
        COUNTS["ffn_pair_offered"] += 1
        shared.seg_fuse_input = (hidden, hidden_fp8, hidden_scale)
        try:
            return ffn(hidden)
        finally:
            # The expert clears it when it takes the pair.
            taken = shared.seg_fuse_input is None
            shared.seg_fuse_input = None
            COUNTS["ffn_pair_taken"] += taken
            _log_once(
                f"ffn-{taken}",
                "[SEG-FUSE] item 1: %s (first at %s, input %s): the shared "
                "expert's gate_up_proj %s the Mega-MHC MXFP8 pair",
                "served" if taken else "not served",
                block.layer_name,
                tuple(hidden.shape),
                "read" if taken else "did not read",
            )

    def fake(hidden, hidden_fp8, hidden_scale):
        return torch.empty_like(hidden)

    return plugin_host.span_n(f"v41.block.ffn+mxfp8@{block.layer_name}", run, fake)


def ffn_reads_pair(block) -> bool:
    """After loading, item 1: whether the FFN this block runs can read the
    pair, so the seam's Mega-MHC should write it."""
    moe = block.ffn
    if block._ffn_quant is None:
        return False
    inline = block._ffn_quant == moe.forward_quant
    reads = not inline or not (moe._use_dual_stream or moe._use_comm_fused_dispatch)
    _log_once(
        f"mhc-{inline}-{reads}",
        "[SEG-FUSE] item 1: FFN seam Mega-MHC MXFP8 output %s (first at %s, FFN %s)",
        "on" if reads else "off: the shared expert runs off the layer's stream",
        block.layer_name,
        "inline" if inline else "behind a plugin op",
    )
    return reads


def install_act(moe) -> None:
    """At block build, item 7: the shared expert's activation becomes the
    one-launch SiLU-mul + MXFP8 quant where `w2` reads that pair natively."""
    shared = _shared(moe)
    if not enabled(7) or shared is None:
        return
    if (
        shared.use_fused_clamp_act_mul
        or not _native_mx(shared.w2)
        or not act_quant_supported(shared.w2.input_size)
    ):
        logger.warning(
            "model_options.seg_fuse: item 7 ignored for %s (needs gfx950, a native "
            "MXFP8 w2 of a width that is a multiple of 32, no Triton fusion)",
            getattr(moe, "prefix", "?"),
        )
        return
    try:
        from atom.model_ops.deepseek_v41.seg_fuse_ops import v41_silu_mul_mxfp8
    except ImportError as error:  # an AITER without module_dsv41_seg_fuse
        logger.warning("model_options.seg_fuse: item 7 ignored (%s)", error)
        return
    shared.seg_fuse_act = v41_silu_mul_mxfp8
    _log_once(
        "act",
        "[SEG-FUSE] item 7: shared expert SiLU-mul + MXFP8 quant in one launch "
        "(first at %s, width %d)",
        getattr(moe, "prefix", "?"),
        shared.w2.input_size,
    )
