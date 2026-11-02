# lib.sh: GPU helpers, sourced after config.env. Read-only; they only look at the KFD sysfs.

# KFD topology node of each GPU, in the order ROCR_VISIBLE_DEVICES counts them.
gpu_nodes() {
  local n
  for n in $(ls /sys/class/kfd/kfd/topology/nodes 2> /dev/null | sort -n); do
    grep -q '^simd_count [1-9]' "/sys/class/kfd/kfd/topology/nodes/$n/properties" 2> /dev/null && echo "$n"
  done
}

node_gpu_count() { gpu_nodes | wc -l; }

# KFD gpu_id of each GPU this run uses: GPUS as ROCR_VISIBLE_DEVICES indices, every GPU when it is empty.
gpu_ids() {
  local i=0 n
  for n in $(gpu_nodes); do
    if [ -z "${GPUS:-}" ] || [[ ",$GPUS," == *",$i,"* ]]; then cat "/sys/class/kfd/kfd/topology/nodes/$n/gpu_id"; fi
    i=$((i + 1))
  done
}

# The node's GPU target, e.g. gfx950 (MI355X) or gfx1250 (MI455X), from gfx_target_version (90500, 120500).
gpu_arch() {
  local n v
  n=$(gpu_nodes | head -1)
  [ -n "$n" ] || return 1
  v=$(awk '/^gfx_target_version/{print $2}' "/sys/class/kfd/kfd/topology/nodes/$n/properties")
  printf 'gfx%d%x%x\n' $((v / 10000)) $((v / 100 % 100)) $((v % 100))
}

# No process holds more than 1 GiB on a GPU this run uses; processes on the node's other GPUs do not count, so an
# 8-GPU MI355X shared four and four still reads idle for the half this run takes.
gpus_idle() {
  local id f v
  [ -d /sys/class/kfd/kfd/proc ] || return 1
  for id in $(gpu_ids); do
    for f in /sys/class/kfd/kfd/proc/*/vram_"$id"; do
      [ -e "$f" ] || continue
      v=$(cat "$f" 2> /dev/null) || continue
      [ "${v:-0}" -gt 1073741824 ] && return 1
    done
  done
  return 0
}
