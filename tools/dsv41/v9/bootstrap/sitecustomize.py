# ruff: noqa: I001
# Keep measured import ordering/namespace; AST parity is verified by prepare.py.
"""Actual-service-only bootstrap; no owner shadow kernels in timed A/B."""

import os
import runpy
import sys
from pathlib import Path

root = Path(os.environ["WORKSPACE_ROOT"])
sys.path.insert(0, str(root / "ATOM/tools/dsv41/owner_bootstrap"))
from owner_bootstrap_scope import service_process

runpy.run_path(str(root / "ATOM/tools/dsv41/owner_bootstrap/sitecustomize.py"))
if (
    service_process(sys.orig_argv)
    and os.environ.get("DSV41_OWNER_SERVING_INSTALLED") == "1"
):
    try:
        sys.path.insert(0, os.environ["DSV41_COMBINED_FROZEN_SOURCE"])
        sys.path.insert(0, os.environ["DSV41_SERVICE_RUN"])
        import formal_counters, workload_counter, trace_control

        formal_counters.install()
        if os.environ.get("P2_WORKLOAD_RING") == "1":
            workload_counter.install()
        trace_control.install()
        import service_resources_v6

        service_resources_v6.install()
        print(
            f"SERVICE_RESOURCES_INSTALL_OK pid={os.getpid()} candidate={os.environ.get('DSV41_COMBINED_GUARDED')} workload_ring={os.environ.get('P2_WORKLOAD_RING')}",
            flush=True,
        )
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        os._exit(78)
