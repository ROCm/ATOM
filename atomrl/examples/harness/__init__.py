"""One-command reproduction and regression runs for the AtomRL examples.

Every example directory carries an ``examples.yaml`` that describes how to run
it, what it needs and how to judge the result. ``examples/run.sh`` reads those
files and drives Docker on one or more GPU hosts; see ``examples/harness/README.md``.
"""
