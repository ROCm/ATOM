#!/usr/bin/env bash
# numa_snapshot.sh NODE...: per-NUMA host memory (GiB) on each node: total, free, file cache, anon (4K), anon THP
# (= Store owners' pinned huge pages), shmem, slab, page tables, and 'other' = total - free - file - anon - anonTHP - shmem.
for n in "$@"; do
  ssh -o BatchMode=yes "${AGX_HOST_PREFIX-pit2-p03-}$n" 'for d in /sys/devices/system/node/node[0-9]*; do
      awk -v node=$(basename $d) '"'"'{v[$3]=$4} END {g=1048576; printf "%s %s total %.0f free %.0f file %.0f anon %.0f anonTHP %.0f shmem %.0f slab %.0f pagetables %.0f other %.0f\n", host, node, v["MemTotal:"]/g, v["MemFree:"]/g, v["FilePages:"]/g, (v["AnonPages:"]-v["AnonHugePages:"])/g, v["AnonHugePages:"]/g, v["Shmem:"]/g, v["Slab:"]/g, v["PageTables:"]/g, (v["MemTotal:"]-v["MemFree:"]-v["FilePages:"]-v["AnonPages:"]-v["Slab:"]-v["PageTables:"])/g}'"'"' host='"$n"' $d/meminfo; done'
done
