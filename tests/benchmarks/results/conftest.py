# CPU-only producer tests: run pytest with --confcutdir=tests/benchmarks/results.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
