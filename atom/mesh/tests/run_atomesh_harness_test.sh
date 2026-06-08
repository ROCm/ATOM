#!/usr/bin/env bash
################################################################################
# @Copyright: 2019-2026 AMD. All Rights Reserved.
# @Author   : zhen.wan@amd.com
# @Date     : 2026-05-26 09:56:21
# @Details  :
################################################################################

set -euo pipefail

LOG_FILE="${LOG_FILE:-/tmp/atomesh_harness_test.log}"
FAILED_COMMANDS=0

: > "${LOG_FILE}"

run_test() {
    printf '\n===== %s =====\n' "$*" | tee -a "${LOG_FILE}"
    "$@" 2>&1 | tee -a "${LOG_FILE}" || FAILED_COMMANDS=$((FAILED_COMMANDS + 1))
}

print_summary() {
    local total_passed
    local total_failed

    read -r total_passed total_failed < <(
        awk '
            /test result:/ {
                for (i = 1; i < NF; i++) {
                    if ($i ~ /^[0-9]+$/ && $(i + 1) == "passed;") {
                        passed += $i
                    }
                    if ($i ~ /^[0-9]+$/ && $(i + 1) == "failed;") {
                        failed += $i
                    }
                }
            }
            END {
                printf "%d %d\n", passed, failed
            }
        ' "${LOG_FILE}"
    )

    printf '\n===== Test Summary =====\n'
    printf 'Log file        : %s\n' "${LOG_FILE}"
    printf 'Total test cases: %d\n' "$((total_passed + total_failed))"
    printf 'Passed          : %d\n' "${total_passed}"
    printf 'Failed          : %d\n' "${total_failed}"
    printf 'Failed commands : %d\n' "${FAILED_COMMANDS}"

    if [[ ${total_failed} -ne 0 || ${FAILED_COMMANDS} -ne 0 ]]; then
        exit 1
    fi
}

TEST_COMMANDS=(
    # Metrics subsystem unit tests.
    "cargo test -p atomesh --lib observability::metrics"
    # Worker /engine_metrics aggregation integration tests.
    "cargo test -p atomesh --test metrics_aggregator_test -- --nocapture"
    # End-to-end Atomesh harness API test.
    "cargo test -p atomesh --test api_tests -- --nocapture --test-threads=1"
)

for test_command in "${TEST_COMMANDS[@]}"; do
    read -r -a test_args <<< "${test_command}"
    run_test "${test_args[@]}"
done

print_summary