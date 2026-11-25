#!/usr/bin/env bash
# host_facts.sh NODE...: each node's host software stack (key=value), read-only from /proc, /sys and docker: no GPU
# query, no container. Compare with README "Host software"; numbers compare only on the same driver and firmware.
for n in "$@"; do
  echo "##### ${n}"
  ssh -o BatchMode=yes "${AGX_HOST_PREFIX-pit2-p03-}${n}" bash -s <<'REMOTE'
kv() { printf '%s=%s\n' "$1" "$2"; }
. /etc/os-release
kv boot "$(uptime -s)"
kv os "${PRETTY_NAME}"
kv kernel "$(uname -r)"
kv bios "$(cat /sys/class/dmi/id/bios_version 2>/dev/null)"
kv cpu "$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs) x $(nproc --all)"
kv rocm "$(cat /opt/rocm/.info/version 2>/dev/null)"
kv amdgpu "$(cat /sys/module/amdgpu/version 2>/dev/null)"
kv gpu_vbios "$(cat /sys/class/drm/card*/device/vbios_version 2>/dev/null | sort | uniq -c | xargs)"
kv gpu_pm_fw "$(cat /sys/class/drm/card*/device/fw_version/smc_fw_version 2>/dev/null | sort | uniq -c | xargs)"
kv gpu_partition "$(cat /sys/class/drm/card*/device/current_compute_partition 2>/dev/null | sort | uniq -c | xargs) / $(cat /sys/class/drm/card*/device/current_memory_partition 2>/dev/null | sort | uniq -c | xargs)"
kv ionic "$(cat /sys/module/ionic/version 2>/dev/null)"
kv nic_fw "$(cat /sys/class/infiniband/*/fw_ver 2>/dev/null | sort | uniq -c | xargs)"
kv rdma "$(for d in /sys/class/infiniband/*; do printf '%s:numa%s ' "${d##*/}" "$(cat "$d/device/numa_node")"; done)"
kv thp "$(cat /sys/kernel/mm/transparent_hugepage/enabled)"
kv cgroup "$(stat -fc %T /sys/fs/cgroup)"
kv docker "$(docker version --format '{{.Server.Version}}' 2>/dev/null) $(docker info --format '{{json .DriverStatus}}' 2>/dev/null)"
kv python3 "$(python3 --version 2>&1)"
kv tools "amd-smi:$(command -v amd-smi >/dev/null && echo y || echo n) ss:$(command -v ss >/dev/null && echo y || echo n)"
REMOTE
done
