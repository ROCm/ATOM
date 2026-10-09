#!/usr/bin/env bash
# Push image references, re-authenticating before each retry.
#
#   docker_push_retry.sh IMAGE [IMAGE...]
#
# The release build runs ~1h before the first push and each push is multi-GB,
# so the Docker Hub token expires mid-flight and the manifest PUT returns
# "unauthorized: authentication required". Sleeping does not fix that, hence a
# fresh login per retry; the first attempt costs nothing extra.
#
# Every call site pushes to Docker Hub, so a bare `docker login` targets the
# right registry. Credentials go in via stdin, never argv.
#
# Env: DOCKER_USERNAME, DOCKER_PASSWORD (no retry without them),
#      PUSH_ATTEMPTS (3), PUSH_BACKOFF_SECONDS (15, doubled each retry),
#      ENGINE (docker).

set -euo pipefail

ENGINE="${ENGINE:-docker}"
ATTEMPTS="${PUSH_ATTEMPTS:-3}"
BACKOFF="${PUSH_BACKOFF_SECONDS:-15}"

[ "$#" -gt 0 ] || { echo "::error::no image reference given" >&2; exit 2; }

relogin() {
    [ -n "${DOCKER_USERNAME:-}" ] && [ -n "${DOCKER_PASSWORD:-}" ] || return 1
    printf '%s' "$DOCKER_PASSWORD" \
        | "$ENGINE" login -u "$DOCKER_USERNAME" --password-stdin
}

push_one() {
    local image="$1" delay="$BACKOFF" attempt
    for attempt in $(seq 1 "$ATTEMPTS"); do
        "$ENGINE" push "$image" && return 0
        [ "$attempt" -lt "$ATTEMPTS" ] || break
        echo "::warning::push ${image} failed (${attempt}/${ATTEMPTS}); re-authenticating, retry in ${delay}s"
        # Retrying without fresh credentials just reproduces the auth failure,
        # so stop here with the reason rather than burning the attempts.
        relogin || { echo "::error::cannot re-authenticate; not retrying ${image}" >&2; return 1; }
        sleep "$delay"
        delay=$((delay * 2))
    done
    echo "::error::push ${image} failed after ${ATTEMPTS} attempts" >&2
    return 1
}

status=0
for image in "$@"; do
    push_one "$image" || status=1
done
exit "$status"
