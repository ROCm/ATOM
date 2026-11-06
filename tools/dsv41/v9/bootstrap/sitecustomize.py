"""V9 service bootstrap: the owner bootstrap, then the V9 service layers.

Runs the owner bootstrap (`tools/dsv41/owner_bootstrap`). In a service process
where the owner serving path installed, it then installs, in order:
`formal_counters`, `workload_counter` (`P2_WORKLOAD_RING=1`), `trace_control`
and `service_resources_v6`, which installs the guarded dispatcher when
`DSV41_COMBINED_GUARDED=1`. A failed install exits the process with 78.
"""

import os
import runpy
import sys
from pathlib import Path

V9 = Path(__file__).resolve().parents[1]
OWNER_BOOTSTRAP = V9.parent / "owner_bootstrap"
sys.path.insert(0, str(OWNER_BOOTSTRAP))
sys.path.insert(0, str(V9))
from owner_bootstrap_scope import service_process

runpy.run_path(str(OWNER_BOOTSTRAP / "sitecustomize.py"))
if (
    service_process(sys.orig_argv)
    and os.environ.get("DSV41_OWNER_SERVING_INSTALLED") == "1"
):
    try:
        from dsv41_v9 import (
            formal_counters,
            service_resources_v6,
            trace_control,
            workload_counter,
        )

        formal_counters.install()
        if os.environ.get("P2_WORKLOAD_RING") == "1":
            workload_counter.install()
        trace_control.install()
        service_resources_v6.install()
        print(
            f"SERVICE_RESOURCES_INSTALL_OK pid={os.getpid()}"
            f" candidate={os.environ.get('DSV41_COMBINED_GUARDED')}"
            f" workload_ring={os.environ.get('P2_WORKLOAD_RING')}",
            flush=True,
        )
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        os._exit(78)
