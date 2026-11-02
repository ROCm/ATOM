# M3.1 MXFP4 stack: 1P1D CPP4->TP4 AgentX on one node (P->D gridlock repro)

Experiment harness, not for main. Runs one ATOMesh CI cell on a node you hold
(`pd_matrix.py` -> `pd_submit.sh` env -> `pd_slurm_job.sh --spur-worker`).

- `run_local_m31.sh <tag> <case> [overrides.sh]`: run ON the node; needs
  `BASE` (writable output dir) and a numeric `LOCAL_JOB_ID`. Never run two at
  once on one node: `pd_slurm_job.sh`'s pre-cleanup stops the other run's
  container.
- `overrides_common.sh`: model snapshot path, dataset cache, page-cache dir.
- `stop_run.sh <LOCAL_JOB_ID>`, `watchdog.sh <LOCAL_JOB_ID> <log dir>`.
- `timeline.py <log dir> [--every MIN]`: per-bucket prefill / Store / decode /
  NIC view (NIC columns need `nic_sampler.sh` running with the same `BASE`).
- `summarize.py`, `from_records.py`: run summaries; CI-format numbers come from
  `.github/scripts/atomesh/process_result.py <run>/slurm_job-*/benchmark_results`.
