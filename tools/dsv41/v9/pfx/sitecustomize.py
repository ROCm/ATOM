"""Q opt-in bootstrap: the HRC layer (D + G10 + H + R + C), then on C's
comm-fused prefill the rebased-main ``num_local_tokens`` shim, PF-CLR v3 and
PF-SHS v3 (``DSV41_PFX=1``).

The HRC modules are not modified; this layer rebinds two functions after they
load. Gated by `DSV41_PRE_COMMFUSED=1`; PF-CLR/PF-SHS additionally by `DSV41_PFX=1`.
A failed install exits the server process with 89.
"""

import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
runpy.run_path(str(HERE.parent / "hrc" / "sitecustomize.py"))
sys.path.insert(0, str(HERE))
from owner_bootstrap_scope import service_process

HITS = {"nl_tensor_dropped": 0, "clear_routes": 0, "local_shared": 0}


def _install():
    import torch

    cf = sys.modules["commfused_v13"]

    # ATOM main passes ``num_local_tokens`` as a device int32 tensor on
    # prefill; commfused_v13 accepts only None. The fused path masks the padded
    # rows itself (topk_ids=-1, weight 0 past ``rows``), so the tensor is
    # redundant there and is dropped.
    run = cf._run_commfused

    def run_commfused(ctx, args, kwargs):
        if isinstance(kwargs.get("num_local_tokens"), torch.Tensor):
            kwargs = {k: v for k, v in kwargs.items() if k != "num_local_tokens"}
            HITS["nl_tensor_dropped"] += 1
        return run(ctx, args, kwargs)

    cf._run_commfused = run_commfused
    if os.environ.get("DSV41_PFX") != "1":
        return "pfx=0"

    from aiter.ops.flydsl.kernels.comm_fused_moe.gfx950.a8w4 import producer
    from clear_routes import compile_clear_missing_routes
    from local_shared import local_shared

    def clear_routes(rows, row_bytes, payload_bytes):
        HITS["clear_routes"] += 1
        return compile_clear_missing_routes(rows, row_bytes, payload_bytes)

    def shared(reps, value, scale, valid, out_rows):
        HITS["local_shared"] += 1
        return local_shared(reps, value, scale, valid, out_rows, cf._HIDDEN)

    producer.compile_clear_missing_routes = clear_routes
    cf._local_shared = shared
    return "pfx=1"


if service_process(sys.orig_argv) and os.environ.get("DSV41_PRE_COMMFUSED") == "1":
    try:
        info = _install()
        print(f"PFX_INSTALL_OK pid={os.getpid()} {info}", flush=True)
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        os._exit(89)
