"""Run the diagnostic LMCache server with an internal stack watchdog."""

import faulthandler
import os

faulthandler.dump_traceback_later(120, repeat=True)
print(f"K3_STACKS LMCache pid={os.getpid()} self-dump every 120 seconds", flush=True)

from lmcache.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
