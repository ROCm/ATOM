#!/usr/bin/env bash
# Incremental image on NODE: FROM_TAG + Dockerfile.csvfix moving /app/aiter-test to FIX_SHA (CSV-only changes,
# enforced in the Dockerfile). Tags rocm/atom-dev:m31-agx-local-<FIX short>. Never pushes. Prints "NODE TAG IMAGE_ID".
# Usage: build_csvfix_node.sh NODE FROM_TAG FIX_SHA FIX_BUNDLE   (FIX_BUNDLE: a git bundle with FIX_SHA under
#   refs/heads/, at a path NODE can read). Env: AGX_IMAGE_TAG, AGX_IMAGE_NODE_DIR, AGX_HOST_PREFIX.
set -euo pipefail
NODE=$1 FROM=$2 FIX=$3 BUNDLE=$4
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
P=${AGX_HOST_PREFIX-pit2-p03-}
TAG=${AGX_IMAGE_TAG:-rocm/atom-dev:m31-agx-local-${FIX:0:8}}
ssh -o BatchMode=yes "${P}${NODE}" bash -s <<REMOTE
set -euo pipefail
L=${AGX_IMAGE_NODE_DIR:-/var/tmp/m31agx-aiter}
C=\${L}/ctx-fix-${FIX:0:8}
rm -rf \${C} && mkdir -p \${C} \${L}/docker-config
cp ${HERE}/Dockerfile.csvfix \${C}/Dockerfile && cp ${BUNDLE} \${C}/fix.bundle
DOCKER_CONFIG=\${L}/docker-config DOCKER_BUILDKIT=1 docker build --progress=plain -t ${TAG} --build-arg FROM_IMAGE=${FROM} --build-arg FIX_SHA=${FIX} \${C} > \${L}/build-fix-${FIX:0:8}.log 2>&1 || { tail -40 \${L}/build-fix-${FIX:0:8}.log; exit 1; }
grep -E "changed vs|AOT replay of|replay gfx950|BISECT|Total time" \${L}/build-fix-${FIX:0:8}.log | sed 's/^#[0-9]* [0-9.]* //' | head -12
echo "${NODE} ${TAG} \$(docker image inspect --format '{{.Id}}' ${TAG})"
REMOTE
