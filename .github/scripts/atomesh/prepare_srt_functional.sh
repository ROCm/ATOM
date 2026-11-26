#!/usr/bin/env bash
set -euo pipefail
root="${GITHUB_WORKSPACE:?}"
runtime="${root}/.sa-srt-runtime"
source_sha="$(python3 - "${SRT_FUNCTIONAL_CONTRACT:?}" <<'PY'
import json, re, sys
sha = json.load(open(sys.argv[1])).get("srtctl_sha", "df706bc365e00bea74e5babe87b4049f624341ea")
assert re.fullmatch(r"[0-9a-f]{40}", sha)
print(sha)
PY
)"
mkdir -p "${runtime}"
if [[ ! -d "${runtime}/source/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/whx-sjtu/srt-slurm.git "${runtime}/source"
fi
git -C "${runtime}/source" fetch --depth=1 origin "${source_sha}"
git -C "${runtime}/source" checkout --detach "${source_sha}"
[[ "$(git -C "${runtime}/source" rev-parse HEAD)" == "${source_sha}" ]]
printf '%s\n' "${source_sha}" > "${runtime}/source-sha.txt"
python3 - "${SRT_FUNCTIONAL_CONTRACT}" "${runtime}/source" "${RUN_DIR}" <<'PY'
import hashlib, json, pathlib, shutil, subprocess, sys
contract = json.load(open(sys.argv[1]))
if spec := contract.get("srt_experiment_patch"):
    patch = pathlib.Path(spec["path"])
    assert hashlib.sha256(patch.read_bytes()).hexdigest() == spec["sha256"]
    subprocess.run(["git", "apply", "--check", str(patch.resolve())], cwd=sys.argv[2], check=True)
    subprocess.run(["git", "apply", str(patch.resolve())], cwd=sys.argv[2], check=True)
    shutil.copyfile(patch, pathlib.Path(sys.argv[3]) / "srt-experiment.patch")
    (pathlib.Path(sys.argv[3]) / "srt-experiment-patch.json").write_text(json.dumps(spec, indent=2))
PY
router_sha="$(python3 - "${SRT_FUNCTIONAL_CONTRACT:?}" <<'PY'
import json, re, sys
contract = json.load(open(sys.argv[1]))
if contract.get("router_arm") == "baseline":
    sys.exit(0)
sha = contract["router_build"]["source_sha"]
assert re.fullmatch(r"[0-9a-f]{40}", sha)
print(sha)
PY
)"
[[ -n "${router_sha}" ]] || exit 0
git fetch --no-tags --depth=1 origin "${router_sha}"
mkdir -p "${runtime}/router-source"
git archive "${router_sha}" atom/mesh | tar -xf - -C "${runtime}/router-source"
cp .github/benchmark/pr5090-router-ab/router-Cargo.lock "${runtime}/router-source/atom/mesh/Cargo.lock"
printf '%s\n' "${router_sha}" > "${runtime}/router-source-sha.txt"
