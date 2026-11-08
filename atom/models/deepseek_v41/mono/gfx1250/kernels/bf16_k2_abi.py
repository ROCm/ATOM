"""bf16-KV K2 ABI and attn-post argument mapping for gfx1250.

The shared attn_post / layer_post ABIs (CDNA reference, mono/kernels) carry
four names the gfx1250 bf16 K2 signature does not take:

  pool_scale, window_scale  fp8 E8M0 scale planes (read only under kv_fp8)
  window                    SWA base; on the bf16 path the SWA ring is a row
                            range inside the same pool buffer (ring_off), so
                            window == pool
  qoff                      shared MoE queue tag offset; the dynamic K2 uses
                            run_moe_front and does not take qoff

Dropping them gives exactly the 75-name gfx1250 index() signature. A K2 that
takes work from the shared MoE launch queue (take_launch_queue) keeps qoff:
ABI_QOFF. The shared ABIs are not mutated.
"""

from atom.mono.runtime.abi import KernelAbi
from atom.models.deepseek_v41.mono.kernels.attn_post import ABI as ATTN_ABI
from atom.models.deepseek_v41.mono.kernels.layer_post import ABI as LAYER_ABI

DROPPED = ("pool_scale", "window", "window_scale", "qoff")

ABI = KernelAbi(tuple(n for n in LAYER_ABI.names if n not in DROPPED))
ABI_QOFF = KernelAbi(tuple(n for n in LAYER_ABI.names if n not in DROPPED or n == "qoff"))

# attn_post_args consumes the attention-side names only (through "normed");
# window is re-derived from pool below, so it is not a positional value.
_ATTN_VALUE_NAMES = tuple(
    n
    for n in ATTN_ABI.names[: ATTN_ABI.names.index("scratch")]
    if n not in ("pool_scale", "window", "window_scale")
)


def attn_post_args(*values) -> dict:
    assert len(values) == len(_ATTN_VALUE_NAMES), (
        len(values),
        len(_ATTN_VALUE_NAMES),
    )
    out = dict(zip(_ATTN_VALUE_NAMES, values))
    out["attn_w"] = out.pop("ffn_w")
    # bf16: the SWA ring lives in the same pool buffer, addressed by ring_off.
    out["window"] = out["pool"]
    return out
