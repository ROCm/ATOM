# p1 milestone recipe: shared paths and helpers (sourced by every recipe script).
# Colleague-reproduction defaults: everything lives under $P1_OUT (override via env var),
# auto-detected relative to this script's own location -- no /shared_nfs path required.
RCP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
P1_OUT=${P1_OUT:-${HOME:-/tmp}/p1-repro-out}
MS=$P1_OUT                                   # small outputs: logs/, results/<TAG>/, gates/<GTAG>/
MAXB=/shared_nfs/ygan/max/bin                # previous team's post-processing scripts (point.sh); optional, internal-cluster only
NLOC=${NLOC:-$P1_OUT/nloc}                   # node-local-equivalent scratch: per-TAG HOME (torch/triton caches) and the flydsl cache
mkdir -p "$MS/logs" "$MS/results" "$MS/gates" "$NLOC" 2>/dev/null

# check_repo: on the internal cluster this enforces an exact pinned commit; for a colleague's own
# clone of the reproducible branch the commit will legitimately differ, so this is informational only.
check_repo() {
  local head dirty
  head=$(git -c safe.directory='*' -C "$REPO" rev-parse --short=9 HEAD 2>/dev/null)
  dirty=$(git -c safe.directory='*' -C "$REPO" status --short --untracked-files=no 2>/dev/null | head -5)
  echo "check_repo: $REPO @ ${head:-unknown}${dirty:+ (dirty: $dirty)} (informational -- a colleague's own clone of p1-c17-reproducible will not match the internal $ARM_REPO_EXPECT pin, that is expected)"
  return 0
}

# seed_flydsl: best-effort warm of the FlyDSL JIT compile cache from the internal cluster's pre-built
# caches. On a colleague's machine none of these source paths exist, so this silently no-ops and the
# first server start just JIT-compiles every kernel from scratch (slower first boot, same correctness).
seed_flydsl() {
  mkdir -p "$FLYDSL_RUNTIME_CACHE_DIR"
  local sd
  if [ -z "$(ls -A "$FLYDSL_RUNTIME_CACHE_DIR" 2>/dev/null)" ] && [ -n "${FLYDSL_SEED:-}" ]; then
    for sd in $FLYDSL_SEED; do
      [ -n "$(ls -A "$sd" 2>/dev/null)" ] && { cp -rn "$sd"/. "$FLYDSL_RUNTIME_CACHE_DIR"/ 2>/dev/null; echo "seed_flydsl: merged $sd"; }
    done
  fi
  [ -n "$(ls -A "$FLYDSL_RUNTIME_CACHE_DIR" 2>/dev/null)" ] || cp -rn /shared_nfs/ygan/max/flydsl_cache/. "$FLYDSL_RUNTIME_CACHE_DIR"/ 2>/dev/null
  return 0
}

# gpus_free: after the server is gone, every leased GPU must drop to <= 5% VRAM (polls up to 180 s).
gpus_free() {
  local g v t busy
  for t in $(seq 36); do
    busy=
    for g in ${HIP_VISIBLE_DEVICES//,/ }; do
      v=$(rocm-smi -d "$g" --showmemuse --csv 2>/dev/null | awk -F, 'NR==2{print $2}')
      [ "${v:-0}" -gt 5 ] && busy="$busy $g:${v}%"
    done
    [ -z "$busy" ] && { echo "### $(date -Is) GPUS_FREE ok ($HIP_VISIBLE_DEVICES)"; return 0; }
    sleep 5
  done
  echo "### $(date -Is) GPUS_FREE FAIL: still used:$busy"; rocm-smi --showpids 2>/dev/null | tail -15
  return 1
}
