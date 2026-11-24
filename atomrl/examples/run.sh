#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

# Run the AtomRL examples with one command and judge the results.
#
#   bash examples/run.sh list             what there is, and on what hardware
#   bash examples/run.sh 1                one example by alias or id
#   bash examples/run.sh suite            regression: unit tests + one-node smokes
#                                         (+ two-node smokes when the site has two hosts)
#
# Runs on a GPU host, or on any machine that reaches the hosts named in a site
# file (site.example.yaml). Needs Python 3.8+ with PyYAML; see README.md.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
if ! "$PYTHON" -c 'import yaml' 2>/dev/null; then
  echo "run.sh needs PyYAML for $PYTHON: $PYTHON -m pip install pyyaml" >&2
  exit 2
fi
PYTHONUNBUFFERED=1 PYTHONPATH="$(dirname "$HERE")${PYTHONPATH:+:$PYTHONPATH}" \
  exec "$PYTHON" -m examples.harness "$@"
