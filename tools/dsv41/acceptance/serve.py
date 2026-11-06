"""Execute the same server command used by the measured acceptance driver."""

import os

from run_rebase_graph_coverage_r6 import SERVER_COMMAND

if __name__ == "__main__":
    argv = SERVER_COMMAND[:]
    if os.environ["DSV41_OWNER_SERVING"] != "1":
        argv = argv[: argv.index("--method")]
    # Preserve the native module entry so spawn imports it as __mp_main__.
    # runpy without replacing __main__ makes children reenter this wrapper.
    os.execv(argv[0], argv)
