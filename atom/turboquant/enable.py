"""
TurboQuant entry point for ATOM.

Usage:
  python3 -m atom.turboquant.enable [ATOM server args...]

Example:
  python3 -m atom.turboquant.enable \
    --model /data/models/amd/Kimi-K2.5-MXFP4 \
    --server-port 8888 -tp 2 \
    --kv_cache_dtype fp8 --block-size 16 \
    --trust-remote-code
"""

import os
import sys

os.environ["ATOM_TURBOQUANT"] = "1"

from atom.turboquant.kv_hook import enable_turboquant  # noqa: E402

enable_turboquant()

sys.argv[0] = "atom/entrypoints/openai_server.py"
from atom.entrypoints.openai_server import main  # noqa: E402
main()
