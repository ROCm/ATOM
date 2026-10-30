# Resource check — 2026-10-30 02:54 UTC

This task owns pending allocation **253987**, `dsv41-attention-stride`:
`amd-di / amd-di-qos`, one exclusive node, 8 MI355X, 64 CPUs, 128GiB,
maximum allowed three-day time limit. Submitted 02:29:26 UTC. The holding
script is `hold.sh` (`exec sleep infinity`). No GPU test has started.

Scheduler and idle-node state rechecked at 02:54:24 UTC: PENDING / Resources. Estimated start is
2026-10-30 04:10:41 UTC on `crsuse2-m2m-v3-049`, estimated end Nov 2
04:10:41 UTC; these are scheduler estimates, not an allocated node.

Read-only account/partition checks found associations for `amd-di` and
`amd-aifw-dev`, with their normal QoS and `amd-burst-qos`. The dev queue has
multiple pending jobs blocked by QOSGrpNodeLimit. No idle nodes are present
in amd-di, amd-spur or amd-spur-v2. Four idle nodes (194, 218, 223, 244)
are in the dedicated amd-aifw-sglang partition, outside these associations.
Keep 253987; no duplicate job was submitted and no other task was changed.

Old borrowed 248783 timed out Oct 29 19:47:12 UTC. Running 244381 and
pending 244401, 248427, 250714 belong to Kimi; do not use or cancel them.

Fresh interactive connection: slog-005, terminal 85976, at the slog prompt.
Reconnect in a later session. Once 253987 is RUNNING, verify its actual node,
Exclusive=1, all eight GPUs and idle state before entering/using the node.
Access method follows `/home/yifehuan/.codex/skills/spur-gpu-access/SKILL.md`.
