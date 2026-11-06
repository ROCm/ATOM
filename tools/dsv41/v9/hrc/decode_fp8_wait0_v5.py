"""DEC-MOE2-TUNE v5: scoped decode fp8 wait0 G1 for the owner384 path (h64 moe2_layout CSV row makes G1 emit
fp8 + v2_output_layout, which retained D's guard minimal_wait (bf16-only) cannot take).
Installed OUTERMOST (after service_resources_v6 / guarded_service_v2), outside every guard-hashed file.
Scope: ACTIVE.owner384 and a=(1536,5120), out fp8 v2 layout, tiles 64/128/256 -> compile_fp8_wait0
(prefill_fp8_wait0_v1, byte-equal to native fp8 G1 in g1wait_2x2_r1). Everything else -> previous dispatcher.
"""

import aiter.ops.flydsl.moe_kernels as mk
import guarded_service_v2 as dispatcher
from prefill_fp8_wait0_v1 import compile_fp8_wait0

INSTALLED = False


def install():
    global INSTALLED
    assert not INSTALLED and dispatcher.INSTALLED
    prev = mk.flydsl_moe_stage1

    def stage(**kw):
        if (
            getattr(dispatcher.ACTIVE, "owner384", False)
            and tuple(kw["a"].shape) == (1536, 5120)
            and kw.get("out_dtype") == "fp8"
            and kw.get("v2_output_layout") is True
            and (kw.get("tile_m"), kw.get("tile_n"), kw.get("tile_k")) == (64, 128, 256)
        ):
            dispatcher.STATE["counts"]["owner_wait0_fp8v2"] += 1
            return mk._flydsl_moe_stage1_impl(**kw, _compile_kernel=compile_fp8_wait0)
        return prev(**kw)

    mk.flydsl_moe_stage1 = stage
    INSTALLED = True
    return {
        "installed": True,
        "prev": getattr(prev, "__module__", "?")
        + "."
        + getattr(prev, "__qualname__", "?"),
    }
