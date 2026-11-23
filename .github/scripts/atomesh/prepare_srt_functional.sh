#!/usr/bin/env bash
set -euo pipefail
root="${GITHUB_WORKSPACE:?}"
runtime="${root}/.sa-srt-runtime"
source_sha=d7305542f91e966a3f9e1028fe7cf8d34bd157a2
mkdir -p "${runtime}"
if [[ ! -d "${runtime}/source/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/whx-sjtu/srt-slurm.git "${runtime}/source"
fi
git -C "${runtime}/source" fetch --depth=1 origin "${source_sha}"
git -C "${runtime}/source" checkout --detach "${source_sha}"
[[ "$(git -C "${runtime}/source" rev-parse HEAD)" == "${source_sha}" ]]
printf '%s\n' "${source_sha}" > "${runtime}/source-sha.txt"
