"""Allow ``python -m atom.autotuner`` as a shortcut for the CLI."""
import sys

from atom.autotuner.cli import main

sys.exit(main())
