"""Run the diagnostic LMCache server with an internal stack watchdog."""


def main():
    from k3_stack_probe import start

    start("LMCache")

    from timeline_event import install

    install()

    from lmcache.cli.main import main as server_main

    return server_main()


if __name__ == "__main__":
    raise SystemExit(main())
