"""Run the diagnostic LMCache server with an internal stack watchdog."""

from k3_stack_probe import start

start("LMCache")

from lmcache.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
