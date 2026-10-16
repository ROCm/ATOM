> The one-hour image comparison below is complete. The workflow on this
> experimental branch now runs the follow-up [focused decode attribution](decode_ablation.md).
> `image_ab.py` remains the shared image/client configuration and historical replay tool.

# TP4 C1 image comparison

This branch is a temporary investigation workflow, not the regular agentic suite.

The completed official baseline is ROCm/ATOM run **38855733134**, artifact
`tp4-c1-official-1`. Its 332 successful profiling requests cover 3590.503 seconds
of a valid 3600-second AgentX measurement. The image-native ATOM commit is
`a0527c9dbfb801e22cd29882454df6f6d21cdd91`.

The current workflow runs only the candidate image on the same runner,
`pit2-p03-g08`, GPUs 0–3. The original shared client container was cleaned up
when the candidate phase's startup guard failed. The replacement client uses
the same immutable baseline image, Python **3.11.14**, and the complete baseline
dependency snapshot in `tp4_c1_client.lock`. Installation must reproduce that
snapshot exactly. The new `plan.json` references the completed baseline run.

GPU initialization waits up to five minutes for two consecutive samples below
1 GiB per selected GPU. This tolerates asynchronous VRAM release after removing
the previous container. Persistent occupancy aborts startup; no other workload
is killed or reset. Server/model and completed-sample verification are unchanged.

The two servers use native image code, FULL graphs, TP4, DSpark5 with fixed
acceptance length 3.51, BF16 KV, FP4 index cache, identical flags and 3600-second
C1 replays. Each writes its original requests, server/client logs and sampled
GPU clocks, power, temperatures and memory. This uses a direct localhost
endpoint; SA's original measurements used AToMesh on different hosts.

The initial second phase failed before loading its server: its guard observed
208–251 GB still occupied only three seconds after the baseline container was
removed. Delayed release is consistent with this observation; the first run's
logs do not identify every residual allocation. The new wait preserves the
occupancy guard and fails if memory does not become free.
