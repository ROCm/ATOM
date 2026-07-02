#!/usr/bin/env bash
# Run the ATOM native, non-GPU unit test suite and emit a JUnit report.
#
# Scope: pure-Python unit tests that run on a plain runner (CPU torch, no GPU,
# no aiter/MoRIIO native libs). Two paths are excluded here because they cannot
# run in this environment:
#   - tests/plugin/                            : sglang/vllm/rtpllm plugin
#       tests (next-stage work). They also install module-level sys.modules
#       stubs at import time that would leak into and break native tests.
#   - tests/entrypoints/test_openai_server.py  : integration test that spawns a
#       real ATOM server and blocks on wait_for_ready; needs a GPU + model.
#
# Other non-unit tests (P/D disaggregation) self-skip via importorskip guards
# inside the test modules, so they show up as visible SKIPs rather than errors.
#
# Env:
#   UNIT_TEST_REPORT  JUnit XML output path (default: unit-report.xml)
set -euo pipefail

REPORT="${UNIT_TEST_REPORT:-unit-report.xml}"

python -m pytest tests/ \
  --ignore=tests/plugin \
  --ignore=tests/entrypoints/test_openai_server.py \
  -rs \
  --junitxml="${REPORT}"
