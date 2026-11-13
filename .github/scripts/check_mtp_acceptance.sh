#!/usr/bin/env bash
# The speculative-decoding acceptance gate, for whoever runs a spec cell.
#
# gsm8k accuracy CANNOT guard a speculative cell. Verification is lossless with
# respect to the target model, so a draft head that has been broken outright --
# reading the wrong hidden states, or an output projection no checkpoint ever
# filled -- leaves every answer correct and shows up only as acceptance falling
# to the floor. This is the only gate that catches that.
#
# It lives here because it was written twice, and the copies had already
# diverged in the way that matters: the PR-side one gated the overall rate and
# not the per-position rates, so PR CI could not see a per-position collapse --
# a draft that still accepts its first token and nothing after it keeps a
# respectable overall number.
#
# `atom_oot_test.sh` records both into the result JSON, scraped from the live
# /metrics endpoint during the gsm8k run.
#
# Usage:
#   check_mtp_acceptance.sh RESULT_JSON OVERALL_THRESHOLD [PER_POS_THRESHOLDS]
#
# PER_POS_THRESHOLDS is a comma-separated list, one per draft position; absent
# or empty means the per-position rates are not gated.
#
# Reads the JSON with jq when it is there and falls back to python3: the
# nightly host runner has no python, the PR path runs inside a container that
# does, and the gate has to be the same number either way.
set -uo pipefail

result_file=${1:-}
overall_threshold=${2:-}
per_pos_thresholds=${3:-}

if [[ -z "${result_file}" || -z "${overall_threshold}" ]]; then
    echo "usage: $(basename "$0") RESULT_JSON OVERALL_THRESHOLD [PER_POS_THRESHOLDS]" >&2
    exit 2
fi
if [[ ! -f "${result_file}" ]]; then
    echo "ERROR: no results JSON for the MTP acceptance check: ${result_file}" >&2
    exit 2
fi
echo "RESULT_FILE: ${result_file}"

# read_metric overall | read_metric position <index>
#
# MTP_GATE_READER pins the reader (jq|python); the default picks whichever is
# installed. It exists so the two can be compared against each other -- a gate
# whose answer depends on which one the runner happens to have is not a gate.
read_metric() {
    local what=$1 idx=${2:-0}
    local reader=${MTP_GATE_READER:-auto}
    if [[ "${reader}" != "python" ]] && command -v jq >/dev/null 2>&1; then
        case "${what}" in
        overall)
            jq -r '.atom_ci_metadata.mtp_acceptance_overall // empty' "${result_file}"
            ;;
        position)
            jq -r ".atom_ci_metadata.mtp_per_pos_acceptance[${idx}] // empty" "${result_file}"
            ;;
        esac
    else
        python3 "${BASH_SOURCE%/*}/read_mtp_metric.py" "${result_file}" "${what}" "${idx}"
    fi
}

overall=$(read_metric overall)
if [[ -z "${overall}" ]]; then
    echo "ERROR: mtp_acceptance_overall missing -- spec-decode /metrics were not captured during the run. Treating as a failure." >&2
    exit 1
fi

fail=0
echo "MTP overall acceptance: ${overall} (threshold ${overall_threshold})"
if awk -v v="${overall}" -v t="${overall_threshold}" 'BEGIN {exit !(v < t)}'; then
    echo "FAIL: overall acceptance ${overall} < threshold ${overall_threshold}"
    fail=1
fi

if [[ -n "${per_pos_thresholds}" ]]; then
    IFS=',' read -ra thresholds <<<"${per_pos_thresholds}"
    idx=0
    for threshold in "${thresholds[@]}"; do
        position=$(read_metric position "${idx}")
        if [[ -z "${position}" ]]; then
            echo "FAIL: position ${idx} acceptance missing (expected >= ${threshold})"
            fail=1
        else
            echo "MTP position ${idx} acceptance: ${position} (threshold ${threshold})"
            if awk -v v="${position}" -v t="${threshold}" 'BEGIN {exit !(v < t)}'; then
                echo "FAIL: position ${idx} acceptance ${position} < threshold ${threshold}"
                fail=1
            fi
        fi
        idx=$((idx + 1))
    done
fi

if [[ "${fail}" -eq 1 ]]; then
    echo "MTP acceptance gate FAILED." >&2
    exit 1
fi
echo "MTP acceptance gate PASSED."
