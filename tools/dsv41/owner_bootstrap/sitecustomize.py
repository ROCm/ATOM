"""Optional FP8 owner installation for the server and its spawned workers."""

import os
import sys

from owner_bootstrap_scope import service_process

if os.environ.get("DSV41_FP8_OWNER_ENTRY") == "1" and service_process(sys.orig_argv):
    try:
        from atom.models.deepseek_v41.owner_service import install
        from atom.models.deepseek_v41.owner_service.lifecycle import (
            install as install_lifecycle,
        )

        install()
        install_lifecycle()
    except BaseException:  # noqa: BLE001 - refuse partially installed serving
        import traceback

        traceback.print_exc()
        os._exit(78)
