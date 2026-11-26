#!/usr/bin/env bash
# Build plugin_poc's wheel from its checkout inside a running benchmark container, install it,
# and check on CPU that ATOM's plugin host routes the requested recipe from the installed wheel.
# Callers skip this script when no plugin_poc ref was requested, so default runs serve no plugin.
#
# The build is plugin_poc's reproducible one (benchmarks/e2e/tp4_recipe/WHEEL.md): a clean copy,
# the version stamped from the commit, pinned build tools and SOURCE_DATE_EPOCH, so a ref gives
# the same wheel bytes as the release built from it (EXPECTED_PLUGIN_WHEEL_SHA256 checks that).
set -euo pipefail

: "${CONTAINER_NAME:?CONTAINER_NAME is required}"
: "${PLUGIN_POC_PATH:?PLUGIN_POC_PATH is required}"  # checkout, relative to the workspace
: "${PLUGIN_RECIPE:?PLUGIN_RECIPE is required}"      # directory under recipes/, e.g. dsv41/tp4-c8-mono
if [[ ! "$PLUGIN_RECIPE" =~ ^[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*$ || "$PLUGIN_RECIPE" == *..* ]]; then
  echo "::error::invalid plugin recipe '$PLUGIN_RECIPE'"
  exit 1
fi

# The container mounts the workspace at /workspace and starts the server from there.
WORKSPACE="${GITHUB_WORKSPACE:-$PWD}"
DIST_PATH="${PLUGIN_POC_PATH}-dist"
RECIPE_IN_CONTAINER="/workspace/${PLUGIN_POC_PATH}/recipes/${PLUGIN_RECIPE}"

docker exec -i \
  -e SRC_DIR="/workspace/${PLUGIN_POC_PATH}" \
  -e DIST_DIR="/workspace/${DIST_PATH}" \
  -e RECIPE_DIR="${RECIPE_IN_CONTAINER}" \
  -e EXPECTED_SHA256="${EXPECTED_PLUGIN_WHEEL_SHA256:-}" \
  "${CONTAINER_NAME}" bash -l <<'IN'
set -euo pipefail
[ -d "$RECIPE_DIR" ] || { echo "::error::no recipe directory $RECIPE_DIR"; exit 1; }
G=(git -c safe.directory="$SRC_DIR" -C "$SRC_DIR")

copy=$(mktemp -d)
"${G[@]}" archive HEAD | tar -x -C "$copy"
stamp="dev$(TZ=UTC "${G[@]}" log -1 --format=%cd --date=format-local:%Y%m%d%H%M)+integ.g$("${G[@]}" rev-parse --short HEAD)"
sed -i -E "s/^version = \"([^\"]+)\"$/version = \"\1.${stamp}\"/" "$copy/pyproject.toml"
# The build tools stay out of the image's environment: its setuptools also serves JIT builds.
python3 -m venv /tmp/plugin-poc-build
/tmp/plugin-poc-build/bin/pip install -q build==1.5.0 setuptools==82.0.1 wheel==0.47.0
rm -rf "$DIST_DIR" && mkdir -p "$DIST_DIR"
SOURCE_DATE_EPOCH=$("${G[@]}" log -1 --format=%ct) \
  /tmp/plugin-poc-build/bin/python -m build --wheel --no-isolation --outdir "$DIST_DIR" "$copy"
wheel=$(ls "$DIST_DIR"/*.whl)
sha=$(sha256sum "$wheel" | cut -d' ' -f1)
echo "built $(basename "$wheel") sha256 $sha"
if [ -n "$EXPECTED_SHA256" ] && [ "$sha" != "$EXPECTED_SHA256" ]; then
  echo "::error::$(basename "$wheel") has sha256 $sha, the release says $EXPECTED_SHA256"
  exit 1
fi

# sp-qkv-provider is atom-plugin's former name; both install the same top-level packages.
python3 -m pip uninstall -y atom-plugin sp-qkv-provider >/dev/null 2>&1 || true
python3 -m pip install --no-deps "$wheel"

# FlyDSL follows AITER (the image's, or what the AITER override installed with it); a plugin states only
# the oldest FlyDSL it needs (backend.json "flydsl") and does not register below it, which the check below
# refuses. Replacing a FlyDSL can leave its dist-info (REQUESTED is not in RECORD) behind; importlib.metadata
# then reads the version as None and every FlyDSL-backed plugin declines.
purelib=$(python3 -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
for d in "$purelib"/flydsl-*.dist-info; do [ -f "$d/RECORD" ] || rm -rf "$d"; done
echo "flydsl $(python3 -c 'import importlib.metadata as m; print(m.version("flydsl"))')"

cd /workspace
HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= python3 - "$RECIPE_DIR" <<'PY'
import importlib
import importlib.metadata as m
import sys
from pathlib import Path

from atom import plugin_host, plugin_recipe

dist = m.distribution("atom-plugin")
home = Path(dist.locate_file("")).resolve()
for name in ("plugins", "pp_models"):
    try:
        path = Path(importlib.import_module(name).__file__).resolve()
    except ModuleNotFoundError:
        continue  # pp_models exists from HOOK_API 3 on
    if not path.is_relative_to(home):
        raise SystemExit(f"{name} comes from {path}, not the installed wheel")
    print(f"{name}: {dist.version} {path}")
plugin_host.configure(plugin_recipe.load(sys.argv[1]), kill_switch=False)
status = plugin_host.status()
print("plan", status["plan_id"])
for seam, info in status["seams"].items():
    print(f"  {seam}: {info['tables']}")
for slot, owners in status.get("modules", {}).items():
    print(f"  {slot}: {owners}")
PY
IN

wheel=$(ls "$WORKSPACE/$DIST_PATH"/*.whl)
sha=$(sha256sum "$wheel" | cut -d' ' -f1)
commit=$(git -C "$WORKSPACE/$PLUGIN_POC_PATH" rev-parse HEAD)
{
  echo "PLUGIN_LAUNCH_ARGS=--plugin-recipe ${RECIPE_IN_CONTAINER}"
  echo "PLUGIN_POC_SHA=${commit}"
  echo "PLUGIN_WHEEL=$(basename "$wheel")"
  echo "PLUGIN_WHEEL_SHA256=${sha}"
} >> "${GITHUB_ENV:-/dev/null}"
{
  echo "### plugin_poc"
  echo
  echo "| commit | wheel | sha256 | recipe |"
  echo "| --- | --- | --- | --- |"
  echo "| \`${commit}\` | \`$(basename "$wheel")\` | \`${sha}\` | \`${PLUGIN_RECIPE}\` |"
} >> "${GITHUB_STEP_SUMMARY:-/dev/null}"
