# ATOM Chat: product design and roadmap

Status: proposal (2026-11-10). Builds on the prototype in `atom/agent/`
(see `docs/atom_chat_guide.md`).

## 1. What we are building

One agent, three audiences:

| Audience | Wants | Front-end |
|---|---|---|
| Customers deploying on AMD GPUs | A correct, fast deployment the first time; no surprises on shared machines | `atom` (REPL, one-shot, guided), later web/API |
| ATOM / AITER engineers | Their workflows (benchmark, trace, debug, kernel tuning) run end to end | `atom --profile developer`, or Claude Code + MCP |
| Other agents (Claude Code, Hermes, Codex) | ATOM operations as tools | `atom mcp` |

"Robust" means these guarantees, each enforced in code rather than in a prompt:

1. **Never break what is not ours.** GPUs, ports and processes we did not lease
   are untouchable; every machine-changing action is gated by policy.
2. **Version-consistent.** Recipes, skills, flags and env vars are read from the
   ATOM/AITER *inside the image that runs*, not from whatever checkout the
   agent happens to sit in. (Already done for flags; extend to the rest.)
3. **Reproducible.** Every deployment and experiment records image digest,
   ATOM/AITER commits, exact command, env, and workload; anything can be re-run.
4. **Honest.** Numbers come from tools; explanations are labelled measured vs
   hypothesis; performance claims carry a noise band.
5. **Recoverable.** Agent crash, brain outage, or a dead trial never leaves
   leaked GPUs or a lost session; everything resumes from durable state.

Design principle: **the LLM decides, the harness executes and measures.**
Anything with a right answer (flag validation, GPU leasing, benchmarking,
parsing logs, statistics) is deterministic code; the model handles intent,
choosing among options, reading failures and profiles, and explaining.

## 2. Architecture

```
 Front-ends     atom REPL / -p / --guided     MCP server     (web, Slack)
                         │                              │
 Agent core     Session ── Brain adapters (OpenAI chat, Anthropic Messages)
                   │       Context manager (budget, summaries, resume)
                   │       Policy engine (permission modes, audit log)
                   │
 Capabilities   Tools (typed, deterministic)  Skills (ATOM + AITER SKILL.md)  Knowledge (recipes/docs)
                   │
 Execution      Runtimes: local │ docker run │ docker exec │ ssh host │ (k8s)
                Resource manager: GPU + port leases (atomic, TTL, heartbeat)
                Job runner: background jobs, logs, timeouts, drain/hang detection
                   │
 Experiments    Ledger (JSONL/SQLite) │ benchmark harness │ workload profiles │ search
```

The swarm (section 5) is a set of agents over the **same** capabilities and
execution plane; nothing in the swarm has its own way to launch a server.

## 3. Skills: ATOM + AITER

### What exists

| Repo | Skills (in the image) | Kind |
|---|---|---|
| ATOM (`/app/ATOM/.claude/skills`, `.claude/commands`) | run-atom-workload, capture-trace, debug-agent-locate-kernel, dump-bisect-debug, review-pr, atom-patterns; commands: benchmark-guide, debug-guide, add-model, ci-pr-guide, ... | deploy/bench, profile, debug, dev |
| AITER (`/app/aiter-test/.claude/skills`) | aiter-config-shape, aiter-op-test, flydsl-kernel-authoring, flydsl-kernel-code-cleanup, opus-kernel-best-practice, opus-module-build-optimization, review-pr, validate-kernel-pr | kernel tuning, kernel dev, review |

Verified: `rocm/atom-dev:nightly_202610271459` ships both sets, matched to
the ATOM (06dbf6692) and AITER (691e129) it runs. Local AITER clones on the
dev box were 2.5 months older and lacked `validate-kernel-pr` -- exactly
the skew to avoid.

### Design

- **One source of truth.** Use the repos' `SKILL.md` files as-is (same files
  Claude Code loads). No forked copies; improvements land upstream and both
  front-ends get them.
- **Load from the runtime image** (`docker run --rm ... cat`), cached per image
  digest; fall back to the host checkout with a visible skew warning.
- **Progressive disclosure.** The system prompt lists each skill's name +
  one-line description; `load_skill(name)` returns the body;
  `read_skill_file(name, path)` returns references/scripts on demand.
- **Execution mapping.** Skills were written for Claude Code ("use Bash",
  "background with `&`", "run `scripts/wait_infer_drain.sh`"). Provide:
  `exec_in_runtime(target, command, background=False)` where target is a
  deployment container (`docker exec`) or a scratch container of the same
  image with the repo + model mounted; plus `job_status` / `wait_job` /
  `job_log` for background work. A short adapter note in the prompt maps
  Claude Code tool names to ours; each skill gets an eval scenario.
- **Profiles decide which skills are visible.**
  - `customer`: deployment, benchmark, accuracy check, trace capture, failure
    diagnosis, AITER tuned-config overlay. No code editing.
  - `developer`: everything, including dump-bisect, kernel authoring and PR
    review -- these need file-editing tools; recommended front-end is
    Claude Code with ATOM Chat's MCP tools rather than re-building an editor.

## 4. Hardening the prototype

Gaps found in the current code, by priority.

**P0 -- before customers**

| Gap | Today | Fix |
|---|---|---|
| Concurrent agents | GPU/port choice is check-then-act; two agents (or swarm workers) can race | Lease manager: flock'd ledger, lease = (gpus, port, owner, ttl, heartbeat); launch requires a lease; expired leases reclaimed only if the process is gone |
| Runtime crash safety | docker log follower process; no image digest | Read logs via `docker logs` on demand; record digest; `reconcile()` on every start; idempotent stop |
| Brain failures | One attempt; stream stall hangs until timeout | Retry with backoff, stall detection, fallback brain; Anthropic Messages adapter (ATOM serves both APIs) |
| Long sessions | Character budget, old tool output truncated | Token-aware budget, rolling summary of old turns, `--resume <session>` from a persisted transcript |
| Permissions | `--yes` or ask | Modes: read-only / ask / auto-safe / auto-all; config allowlists; JSONL audit log of every action and approval |
| Version skew beyond flags | Recipes and env names from host checkout | Read recipes, skills, `envs.py` from the image; flag unknown `ATOM_*`/`AITER_*` env names; show ATOM/AITER/ROCm versions in every plan |
| Agent regressions | Unit tests only | Agent eval suite: the scenarios already run (fail-then-fix, deploy+bench+switch, guided, one-shot) as replayable tests with a scripted brain, plus a nightly real-GPU run; score success, tool errors, steps, unsupported claims |

**P1**

- `ssh` runtime for remote boxes (e.g. lab machines behind a jump host).
- Ship recipes/skills in the wheel or fetch them by ATOM version (pip installs
  have no `recipes/` today).
- Surface the server's Prometheus metrics (`docs/metrics_guide.md`) in
  status; event stream for UIs.
- Service mode: agent as an API for many users, with per-user leases.
- Multi-node / PD via atomesh recipes.

## 5. Agent swarm for "make it performant"

### Frame it as an optimization problem

- **Inputs:** workload profile (ISL/OSL distributions, concurrency, prefix
  reuse -- or a trace such as AgentX), SLA (e.g. TTFT p99, TPOT p99),
  objective (max tokens/s per GPU under SLA, or min latency), budget
  (GPU-hours, wall clock), accuracy guard (baseline from
  `.github/benchmark/models_accuracy.json`).
- **Search space, by cost:**
  1. Serving config: TP/DP/EP, `--max-num-seqs`, `--max-num-batched-tokens`,
     prefill chunk, capture sizes, block size, prefix caching, KV dtype --
     a restart per trial (minutes).
  2. Algorithms: speculative method / draft tokens, MoE backend, mono decode,
     TBO -- also a restart.
  3. Kernels: AITER tuned configs for the shapes this workload hits
     (untuned CSV -> `*_tune.py` -> tuned overlay under
     `aiter/configs/model_configs`, following aiter-config-shape) -- tens of
     minutes to hours, but reusable across runs.
  4. Code changes -- developer profile only, never for customers.

### Roles

| Role | Mostly | Does |
|---|---|---|
| Orchestrator | LLM (strongest model) | Reads recipes + workload, defines the search space and priors, picks each batch of trials, writes the final report |
| Scheduler | code | Leases GPU slices, packs trials (8 GPUs = 2 x TP4 or 4 x TP2 at once), retries, timeouts |
| Trial worker | code (LLM only on failure) | launch -> ready -> warmup -> benchmark -> collect metrics + logs -> teardown; on failure an LLM diagnoser (debug skills) classifies it as a config error or a bug |
| Profiler | LLM + capture-trace skill | For the top configs: trace, kernel breakdown (`tools/parse_trace.py`), hot kernels, untuned shapes |
| Kernel tuner | code + aiter skills | Tunes the hot shapes, builds the config overlay, runs the collision guard, hands back for re-benchmark |
| Accuracy guard | code | GSM8K (`run-atom-workload`) on finalists; rejects regressions beyond the noise band |
| Critic | LLM + statistics | Re-runs the winner A/B/A, checks the gain exceeds run-to-run noise, rejects benchmark-only knobs |

Agents communicate through the experiment ledger and a task queue (structured
JSON), not through each other's transcripts: deterministic, resumable, and
every claim traceable to a measurement.

### Search

Seed with recipe configs and past results for the same (model, GPU, image,
workload) -> coarse sweep of the highest-impact knobs -> successive halving
(short benchmark for all, long for survivors) -> local search around the
best -> noise-aware selection (repeats, A/B/A interleaving, confidence
intervals). Run-to-run noise between server restarts of +-8-11% has been
measured on DeepSeek-V4.1 on MI355X, so single-run comparisons are not
evidence.

The output is a Pareto frontier (throughput vs TTFT/TPOT) with reproducible
launch artifacts for each point; the customer picks the operating point.

### The real constraints

- **Startup dominates.** A DeepSeek-V4.1-Flash TP4 trial took 6.6-7.2 min
  to become ready; with two TP4 slots that is about 12 trials per hour. The
  search must be prior-driven, not brute force. Mitigations: persistent
  compile-cache volumes (already in place), screening configs for startup
  failures on a truncated checkpoint first (e.g. a 4-layer model starts in
  ~80 s), and a shared experience database.
- **A self-hosted brain competes with the trials.** On an 8-GPU box a TP4
  brain halves trial capacity. Options: an external/hosted endpoint for the
  orchestrator, a small TP1-TP2 brain, or a brain on another node. This is a
  product decision (cost, data residency).
- **Accuracy is part of "performant."** Every winner passes the accuracy
  guard; forced-acceptance and other benchmark-only knobs are rejected.
- **Benchmark harness.** `atom.benchmarks.benchmark_serving` supports only
  `--dataset-name random`; realistic agentic workloads need trace replay
  (AIPerf, as the AgentX recipes use). The harness wraps both behind one
  workload profile.

### The long-term asset

Every trial (model, GPU, image, workload, config -> metrics) goes into a
shared results store. Future plans start from the best known configuration,
the orchestrator explains choices with evidence, and winners can be proposed
upstream as recipe updates. That knowledge base -- not the agent loop -- is
what makes AMD deployments get better over time.

## 6. Roadmap

| Phase | Scope | Done when |
|---|---|---|
| 1. Robust core | Leases, runtime hardening, brain retries + Anthropic adapter, session persistence + audit log, permission modes, image-sourced knowledge/env validation, agent eval suite | Two agents on one box never collide; kill -9 of the agent loses nothing; eval suite green nightly on MI355X |
| 2. Skills | Skill loader (catalog, load, files), `exec_in_runtime` + background jobs, customer/developer profiles; run-atom-workload, capture-trace, GSM8K guard working through skills | "benchmark it", "trace it", "check accuracy" work end to end via the repos' own skills |
| 3. Swarm v1: `atom tune` | Workload profile + SLA -> serving/algorithm search with scheduler, ledger, accuracy guard, noise-aware selection, Pareto report | Beats the recipe default on a held-out workload by more than the measured noise band, reproducibly |
| 4. Swarm v2 | Profiler + AITER kernel-tuning lane, experience database, upstream recipe proposals, multi-node/PD | Tuned-config overlays produced and validated automatically |

### Phase 1 status (2026-11-12): done

| Item | Where | Validated |
|---|---|---|
| Machine-wide GPU/port leases | `atom/agent/leases.py`, `deploy.launch` | 8-process race: exactly one winner; two real agents racing for GPU 4 with docker: one launched, the other refused naming the holder |
| Runtime hardening | `deploy.py`: atomic ids, `--init`, per-user container names, image id, logs via `docker logs` + saved at stop, `reconcile()` releases dead servers' leases | kill -9 test below |
| Version consistency | `atom/agent/image.py`: versions, flags, env names, recipes/docs/skills (ATOM + AITER) read from the image, cached per image id | ATOM 06dbf6692 / AITER 691e1292e / ROCm 7.2.4 recorded on every plan and deployment; unknown env names flagged |
| Permission modes + audit | `policy.py`: read-only / ask / auto-safe / auto decided in code; `audit.jsonl` | read-only refuses without asking; every call and approval recorded |
| Session persistence | `session.py`: fsynced transcript, `--resume`, repair of cut-off tool calls, turn-safe compaction | agent killed mid-`wait_until_ready`: deployment, lease and transcript intact; resumed chat re-checked state and finished |
| Brain resilience | `brain.py`: typed retryable errors, backoff, stall timeout, fallback brains, Anthropic Messages client | fault injection (503 / stalled / truncated streams); Messages tool round trip against ATOM's own `/v1/messages` |
| Agent evals | `atom/agent/evals.py`, 5 scenarios scored from tool calls and records | 5/5 with DeepSeek-V4.1-Flash on MI355X; the first run exposed that `inspect_system` hid leases (fixed, checks tightened) |

Unit tests: 78, no GPU needed (`pytest tests/agent --noconftest`).

### Phase 2 status (2026-11-13): done

| Item | Where | Validated |
|---|---|---|
| Skill catalog from the image (ATOM 6 skills + 7 commands, AITER 8 skills), namespaced, progressive disclosure | `atom/agent/skills.py`; tools `list_skills`, `load_skill`, `read_skill_file` | customer profile sees 4 of 21, developer all 21 |
| Execution where ATOM runs: foreground and background jobs inside the deployment's container, process-group cancel, survive agent restarts, copy results out | `atom/agent/jobs.py`; tools `exec_in_runtime`, `job_status`, `wait_job`, `job_log`, `cancel_job`, `copy_from_runtime` | real `run_benchmark.sh` job in a container (exit 0, 15 s); cancel killed the job's tree inside the container |
| Shell policy for unattended runs: commands split into parts, read-only / workload / scratch-write parts run in auto-safe, lifecycle scripts and killers refused on execution (not on mention) | `atom/agent/shell_policy.py` | 30+ cases incl. variable indirection, `$(...)`, escaped `find` groups, `python3 -c` imports |
| Skill translation notes (server lifecycle -> lease-aware tools, Bash -> exec_in_runtime, fresh containers need no cleanup) | `skills.ADAPTER_NOTES`, system prompt | trace scenario recovered once the "clean after launch" mistake was named |
| Customer / developer profiles | `--profile` | |
| Evals: `benchmark_via_skill`, `accuracy_via_skill`, `trace_via_skill` | `atom/agent/evals.py` | reported numbers checked against the job logs; trace file checked on the host; full suite 8/8 with DeepSeek-V4.1-Flash on MI355X |

What the GPU runs found (all fixed): the first shell classifier forced
approval for the skills' own read-only poll loops and inspection one-liners,
so unattended runs stalled; a lifecycle-script *name* in an `ls` was treated
as running it; the GSM8K score regex matched the skill's `0.XXXX` doc
placeholder; dash rejects `kill -- -PGID`, so cancel signalled nothing; the
agent cleared the trace directory after launching (the skill says before).

## 7. Open decisions

1. Customer profile: deployment/tuning skills only (recommended), or also code-editing developer skills?
2. Swarm brain: hosted model for the orchestrator + local workers, or fully self-hosted (costs trial GPUs)?
3. Results store: local per machine first, or a team-wide service from the start?
4. Productization target: open-source in ROCm/ATOM, or an internal tool first?

## 8. Relation to DeepSeek's Super-agent

DeepSeek's Harness team published "Exploring Super-agent Design" and the
`dsh-superagent-prototype` (2026-10-31): a resident **Router** that talks to
people and dispatches, and isolated **Workers** that each own one task. Its
premise is that human attention (3-5 live sessions) is the bottleneck.

Same shape, different problem:

| | Super-agent | ATOM Chat swarm |
|---|---|---|
| Bottleneck | Human attention | GPU time (~7 min per V4.1 TP4 trial) and measurement noise (+-8-11%) |
| Workers | Full LLM coding sessions; cheap to parallelize | Mostly deterministic harness; each needs an exclusive GPU slice |
| Parallelism | Bounded by usefulness | Bounded by free GPUs -> lease scheduler is the core |
| Strongest model | Workers (they write the code) | Orchestrator + diagnoser; chat routing can be light |
| Output | Code / PRs a human reviews | Measurements -> repeats, noise bands, accuracy guard |
| Asset | The orchestration | Recipes, skills, and the results database |

Borrowed ideas:

- A separate coordinator instead of "one agent, delegate when appropriate":
  in DeepSeek's test a single session with a neutral prompt never delegated
  (0/5 runs). `atom tune` is therefore its own orchestrator mode, not a
  decision the chat loop makes.
- Threads/subscriptions: a trial's events (failure, hang, completion) go to
  that trial's worker/diagnoser directly, not through the orchestrator.
- Two-level trigger filter: programmatic fast path, LLM slow path -- the same
  as "harness first, LLM on failure".
- A Watchdog role for stalled work; stateless loops with state in a store
  (resume after crash); context as a cache (inline what is likely needed,
  spill the rest behind IDs such as trial/deployment ids).

Their results are preliminary (synthetic workload; p95 15.7 vs 29.5 min
against a single session, with more cache-read tokens), and multi-agent
teams can underperform single agents (CooperBench, 2026-01), so the critic
and noise-aware selection matter more than the number of agents.

Super-agent-style traffic (one router, many parallel worker sessions with
heavy prefix reuse) is also the agentic serving workload ATOM optimizes for;
it is a natural workload profile for `atom tune`.
