# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""One-command reproduction and regression runs for the AtomRL examples.

Every example directory carries an ``examples.yaml`` that describes how to run
it, what it needs and how to judge the result. ``examples/run.sh`` reads those
files and drives Docker on one or more GPU hosts; see ``examples/harness/README.md``.
"""
