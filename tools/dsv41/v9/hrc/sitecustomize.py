"""HRC opt-in bootstrap: the V9 owner bootstrap (D), then H, R and C.

Each layer is gated by its own variable and exits the server process with its
code on a failed install: decode FP8 wait0 G1 (80), RoPE num_warps (83) and the
comm-fused prefill MoE (86).
"""

import importlib.util
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
runpy.run_path(str(HERE.parent / "bootstrap" / "sitecustomize.py"))
from owner_bootstrap_scope import service_process

LAYERS = (
    ("decode_fp8_wait0_v5", "DSV41_DECODE_FP8_WAIT0", "DECODE_FP8_WAIT0", 80),
    ("rope_nw_v7", "DSV41_ROPE_OBS", "ROPE_NW", 83),
    ("commfused_v13", "DSV41_PRE_COMMFUSED", "COMMFUSED", 86),
)

if service_process(sys.orig_argv):
    for name, gate, tag, code in LAYERS:
        if os.environ.get(gate) != "1":
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                name, str(HERE / f"{name}.py")
            )
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            info = mod.install()
            print(f"{tag}_INSTALL_OK pid={os.getpid()} {info}", flush=True)
        except BaseException:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            os._exit(code)
