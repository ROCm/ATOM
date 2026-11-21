# `run.sh` harness

`examples/run.sh` runs `python -m examples.harness`. It reads the `examples.yaml` next to
each example, drives Docker on one or more GPU hosts, and judges each run against the
references recorded in that file.

| Command | Does |
|---|---|
| `run.sh list` | the catalog: ids, aliases, tiers, hardware, status |
| `run.sh show <id>` | the resolved commands, per host |
| `run.sh <id>` / `run.sh run <id>` | run one example and judge it; `--full` for the long run |
| `run.sh suite [--tier …] [--only …]` | run a tier, or a subset |
| `run.sh check <id> --log <file>` | judge an existing log |
| `run.sh prepare <id>…` | download models and data, run each example's `prepare` step |
| `run.sh compare <A> <B>` | compare two runs' `summary.json` |

## Verdicts

| Verdict | Meaning |
|---|---|
| PASS | exit code 0, final step logged, no `Traceback` / `OutOfMemory` / `CUDA error` / `HSA_STATUS`, no weight-sync bucket that skipped tensors, every checked metric within tolerance |
| FAIL | any of the above violated |
| XFAIL / XPASS | the example carries a `known_issue` and failed / passed |
| SKIP | cannot run here: too few hosts, image or data missing, a capability the image lacks |
| ERROR | the environment got in the way: busy GPUs, a stalled log, a failed container |

The exit status is non-zero on any FAIL or ERROR, or if nothing completed; `--strict`
adds SKIP. Each run writes `atomrl/output/regression/<timestamp>-<label>/` with
`summary.md`, `summary.json` (verdicts, every step's metrics, image and code revisions),
`junit.xml` and the logs.

## Sites

On one GPU machine, `DATA_ROOT` is the only setting. For several hosts, or to drive GPU
hosts from another machine, write a site file ([`../site.example.yaml`](../site.example.yaml))
and pass it with `--site`, put it at `examples/site.yaml`, or name it in `$ATOMRL_SITE`.
It lists the hosts and their IPs, how to reach them (ssh by default, or any command
template), the RDMA NIC, image overrides and extra mounts. The data root, the checkout
and the mounts must be visible at the same path on every host.

Multi-node examples start one Ray cluster through the example's own launcher
(`RAY_ROLE=head` on the first host, `worker` on the others, one shared auth token), so
every raylet inherits the launcher's environment, then run the trainer as the `driver`
on the first host.

## Example layout

```
<example>/
├── README.md        overview, results, requirements, how to run
├── examples.yaml    how run.sh runs and judges it
├── run_*.sh         the launcher it wraps; usable without run.sh
├── configs/         <algorithm>_<model>_<backends>_<variant>_{smoke,longrun}.yaml
├── docker/          only when the example needs an image of its own
└── tools/           data preparation, benchmarks, exports, comparisons
```

## `examples.yaml`

```yaml
images:                       # optional; keys are global
  release: {ref: atomrl:release, atomrl_mount: /opt/atomrl/ATOM/atomrl, env: {RL_ROOT: /opt/atomrl}}
defaults: {image: release, command: bash {examples}/DAPO/run_dapo.sh}
examples:
  - id: dapo-qwen3-8b-atom-fp8
    aliases: ["4"]
    title: Qwen3-8B, FSDP2 FP8 training, ATOM FP8 rollout
    tier: t1                  # t0 | t1 | t2 | long
    nodes: 1
    requires: [models/Qwen3-8B-Base, ...]   # under the data root; checked before running
    prepare: python3 {examples}/DAPO/tools/prepare_data.py --data-root {data_root}
    compile_cache_key: atomfp8 Qwen3-8B-Base
    env: {MODE: atomfp8, TRAIN_FP8: "1", MODEL_PATH: "{data_root}/models/Qwen3-8B-Base"}
    smoke:                    # what run.sh and the suite run
      config: configs/dapo_qwen3_8b_ray_atom_fp8_4k_smoke.yaml
      steps: 3
      checks:
        metrics:              # read from the step-1 "step=1 key=value ..." log line
          - {name: rollout_corr/k3_kl, ref: 0.00399, rtol: 0.5}
          - {name: rollout_corr/kl, ref: 0.00374, band: 10}
    full: {config: configs/dapo_qwen3_8b_ray_atom_fp8_longrun.yaml, steps: 1000}
```

Placeholders such as `{data_root}`, `{config}`, `{examples}`, `{hosts[1].ip}` and
`{rdma.interface}` are filled in by `run.sh`. A metric is judged by exactly one of
`rtol` (relative to `ref`), `band` (within a factor of `ref`), `min` / `max`, or `info`
(recorded only). `known_issue` turns FAIL into XFAIL; `needs` declares capabilities
whose absence is a SKIP: `rdma` (an RDMA NIC and the site's `rdma` section) and
`atom_rdma` (an ATOM with the RDMA weight receiver: the release image's since `-261119`, or one mounted with `--atom-src`). Unknown keys are rejected, and
`atomrl/tests/unit/test_examples_harness.py` loads every `examples.yaml` and builds
every example's commands on each CI run.
