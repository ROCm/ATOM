#!/usr/bin/env bash
# Validation-only exit receipt. The shared outer supervisor owns all descendants.
set +e
lm_eval "$@"
client_rc=$?
printf '{"exit_code":%d}\n' "$client_rc" > "$EVIDENCE/lm-eval-exit.json"
exit "$client_rc"
