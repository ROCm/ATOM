#!/usr/bin/env bash
set -euo pipefail
runtime="${REPO_ROOT:?}/.sa-srt-runtime"
output="${RUN_DIR:?}/router-build"
contract="${REPO_ROOT}/${ATOMESH_ENV_ATOMESH_PR5090_CONTRACT:?}"
mkdir -p "${output}"
arm="$(python3 - "${contract}" <<'PY'
import json, sys
print(json.load(open(sys.argv[1])).get("router_arm", "atomesh"))
PY
)"
if [[ "${arm}" == baseline ]]; then
  python3 "${REPO_ROOT}/.github/scripts/atomesh/pd_verify_baseline_router.py" \
    --contract "${contract}" --output "${output}" --job-id "${SRTCTL_JOB_ID:?}"
  exit 0
fi
[[ "${arm}" == atomesh ]] || { echo "Unsupported router arm: ${arm}" >&2; exit 1; }
mapfile -t build_spec < <(python3 - "${contract}" "${runtime}" <<'PY'
import hashlib, json, pathlib, re, sys
c = json.load(open(sys.argv[1]))
runtime = pathlib.Path(sys.argv[2])
b = c["router_build"]
assert (runtime / "router-source-sha.txt").read_text().strip() == b["source_sha"]
assert hashlib.sha256((runtime / "router-source/atom/mesh/Cargo.lock").read_bytes()).hexdigest() == b["lockfile_sha256"]
assert re.fullmatch(r"[0-9a-f]{40}", b["source_sha"])
assert re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", b["rust_toolchain"])
assert b["profile"] in ("release", "ci")
assert re.search(r"@sha256:[0-9a-f]{64}$", c["router_images"]["atomesh"])
for value in (c["router_images"]["atomesh"], b["source_sha"], b["rust_toolchain"], b["profile"]):
    print(value)
PY
)
[[ "${#build_spec[@]}" == 4 ]] || { echo 'Invalid router source/build contract' >&2; exit 1; }
base_image="${build_spec[0]}"
source_sha="${build_spec[1]}"
toolchain="${build_spec[2]}"
profile="${build_spec[3]}"
cache="${TMPDIR:-/tmp}/atomesh-reviewed-router-$(id -u)"
mkdir -p "${cache}/home"
echo "[router-build] source=${source_sha} rust=${toolchain} profile=${profile}"
timeout --kill-after=30s "${ATOMESH_ROUTER_BUILD_TIMEOUT_SECONDS:-1800}s" docker run --rm \
  --name "srtctl_${JOB_ID:?}_router_build" --label "srtctl.job=${SRTCTL_JOB_ID:?}" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/mesh-cache/home -e CARGO_HOME=/mesh-cache/cargo-home -e RUSTUP_HOME=/mesh-cache/rustup-home \
  -e RUSTUP_TOOLCHAIN="${toolchain}" -e ATOMESH_MESH_RUST_TOOLCHAIN="${toolchain}" \
  -e ATOMESH_MESH_SOURCE_COMMIT="${source_sha}" -e ATOMESH_MESH_SOURCE_DIRTY=false \
  -e ATOMESH_MESH_BUILD_PROFILE="${profile}" -e ATOMESH_MESH_TARGET_DIR=/mesh-cache \
  -e CARGO_BUILD_JOBS=8 \
  -v "${runtime}/router-source:/source:ro" -v "${output}:/build-output" -v "${cache}:/mesh-cache" \
  -v "${REPO_ROOT}/.github/scripts/atomesh/build_mesh.sh:/build_mesh.sh:ro" \
  --entrypoint bash "${base_image}" /build_mesh.sh /source/atom/mesh /build-output \
  > "${output}/build-container.log" 2>&1
python3 - "${contract}" "${output}" <<'PY'
import hashlib, json, pathlib, sys
c = json.load(open(sys.argv[1]))
out = pathlib.Path(sys.argv[2])
spec = c["router_build"]
m = json.loads((out / "mesh-build.json").read_text())
assert m["commit"] == spec["source_sha"] and m["profile"] == spec["profile"]
assert m["cargo"].startswith("cargo " + spec["rust_toolchain"] + " ")
assert m["lockfile_sha256"] == spec["lockfile_sha256"]
assert m["binary_sha256"] == hashlib.sha256((out / "atomesh").read_bytes()).hexdigest()
(out / "Dockerfile").write_text(
    "FROM " + c["router_images"]["atomesh"] + "\n"
    "COPY --chmod=755 atomesh /opt/atomesh-reviewed/atomesh\n"
    "ENV PATH=/opt/atomesh-reviewed:${PATH}\n"
    'LABEL io.atomesh.router.source="' + spec["source_sha"] + '"\n'
    'LABEL io.atomesh.router.binary-sha256="' + m["binary_sha256"] + '"\n'
)
# Keep the image build context to the executable and Dockerfile.
(out / ".dockerignore").write_text("*\n!atomesh\n!Dockerfile\n")
PY
if [[ -n "${GPOOL_LEASE:-}" ]]; then
  python3 "${REPO_ROOT}/.github/scripts/atomesh/pd_gpool_guard.py" router-image
fi
timeout --kill-after=30s 300s docker build --iidfile "${output}/image-id.txt" \
  -f "${output}/Dockerfile" "${output}" > "${output}/image-build.log" 2>&1
image_id="$(cat "${output}/image-id.txt")"
[[ "${image_id}" =~ ^sha256:[0-9a-f]{64}$ ]] || exit 1
docker image inspect "${image_id}" > "${output}/image-inspect.json"
timeout --kill-after=10s 60s docker run --rm --label "srtctl.job=${SRTCTL_JOB_ID}" --user "$(id -u):$(id -g)" \
  --entrypoint bash "${image_id}" -c 'set -e; binary="$(command -v atomesh)"; test "${binary}" = /opt/atomesh-reviewed/atomesh; sha256sum "${binary}"; atomesh launch --help' \
  > "${output}/runtime-check.txt" 2>&1
python3 - "${contract}" "${output}" <<'PY'
import json, pathlib, sys
c = json.load(open(sys.argv[1])); out = pathlib.Path(sys.argv[2])
m = json.loads((out / "mesh-build.json").read_text())
image_id = (out / "image-id.txt").read_text().strip()
image = json.loads((out / "image-inspect.json").read_text())[0]
check = (out / "runtime-check.txt").read_text()
assert image["Id"] == image_id
assert image["Config"]["Labels"]["io.atomesh.router.source"] == m["commit"]
assert check.split()[0] == m["binary_sha256"] and "moriio-read" in check
if c.get("kv_transport") == "moriio":
    assert "moriio-write" in check and "--vllm-moriio-peers" in check
m.update(image_id=image_id, base_image=c["router_images"]["atomesh"], runtime_verified=True)
(out / "runtime.json").write_text(json.dumps(m, indent=2) + "\n")
print("[router-build] verified " + image_id + " source=" + m["commit"])
PY
