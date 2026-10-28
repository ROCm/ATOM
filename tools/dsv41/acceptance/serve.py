"""Execute the same server command used by the measured acceptance driver."""

import os
import runpy
import sys

from run_rebase_graph_coverage_r6 import SERVER_COMMAND

argv = SERVER_COMMAND[:]
if os.environ["DSV41_OWNER_SERVING"] != "1":
    argv = argv[: argv.index("--method")]
sys.argv = [argv[2], *argv[3:]]
runpy.run_module(argv[2], run_name="__main__")
