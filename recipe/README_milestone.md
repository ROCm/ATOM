# p1 milestone recipe: AgentX c32 A/B + accuracy gate (built by track build-int5, CPU only)

Self-contained: every script here sources only files in this directory, the two pinned trees, and READ-ONLY scripts in
/shared_nfs/ygan/max/bin (gpurun.sh, gsm8k_ix.sh, long_check.py, al_stats.py, summarize_ix.py, slim_srvmetrics.py).
Nothing writes under /shared_nfs/ygan/max except gpurun.sh's own lease/lock files (as for every lease).
Outputs: /shared_nfs/ygan/p1/milestone/{logs,results/<TAG>,gates/<GTAG>}; node-local HOME + flydsl cache under
/mnt/m2m_nobackup/ygan/p1/milestone (flydsl seeded read-only from max/flydsl_cache).

## Arms
- A (control) = int1_both_c32 EXACTLY as the previous team ran it: tree /shared_nfs/ygan/work/max-int1 @ 60870c2b4,
  int1_recipe.sh (byte copy of max/int1/recipe.sh, sha256 6ece7176...), + ATOM_PREFILL_DECODE_INTERVAL=16
  ATOM_PREFILL_DELAYER_TARGET_FILL=0.225 (chain2.sh both_c32), MONO 0, DUTY 0.3, CAP 8192, capture
  [1..8,10,12,14,16,20,24,28,32,40,48,56,64], fp8 draft KV, CONC 32 / max-num-seqs 64 / MEM 0.85 / k7 / AL 4.04.
  check_recipe.sh proves arm A's env (21 vars) and EXTRA equal the "### env"/"### extra" header of
  max/logs/server_int1_both_c32.log. Reference result: max/results/int1_both_c32 = 54668 tok/s/GPU @ ix_p90 141.1.
- B = p1/integrated5 @ 5f26263a8 (/shared_nfs/ygan/work/p1-integrated5) with every p1 flag on (env_armB.sh; exact list
  in flags_armB.txt): int1 base + round-3 7 flags + chain Y / MoE tiles / split4 / draft AR (overlay
  /shared_nfs/ygan/p1/draft-fusion/overlay on PYTHONPATH after the amdk snapshot) / fused draft ctx rope + split4_wide +
  ATOM_P1_DRAFT_BS_LADDER (production list) + mono (ATOM_MONO_ENABLE=1, ATOM_P1_MONO_MAX_ROWS=8) + scheduler DUTY 0.3,
  CAP 8192, TARGET_FILL 0.225, INTERVAL 8, finish hold LEFT 128 / MAX_MS 500 / MAX_TOKENS 400; capture
  [1..24,28,32,40,48,56,64]. Same serve.sh arguments as A.

## Files
- common.sh (paths, check_repo, seed_flydsl, gpus_free), env_armA.sh, env_armB.sh, int1_recipe.sh
- serve.sh, bench.sh, gate.sh, gate_verdict.py, ladder_stats.py: copies of max/bin/* made by mkcopies.sh; ONLY output
  paths changed (+ `git -c safe.directory` in serve.sh's log header). Exact diffs: diffs/*.diff.
- point.sh: one AgentX point (port of max/int1/point.sh): fresh server, bench.sh (aiperf inferencex-agentx-mvp, CONC,
  900 s, seed 42, same args as the previous team), stop, GPUs-free check, ladder_stats/summarize_ix.
- gate_job.sh: accuracy gate (port of max/int1/gate_job.sh): AL=real server (max-num-seqs 16), GSM8K x2 via
  max/bin/gsm8k_ix.sh (lm-eval, 1319 q, greedy, conc 16), REAL acceptance length over the GSM8K window via
  max/bin/al_stats.py on the server log, needle 20K/80K/130K via max/bin/long_check.py, verdict vs
  max/acc/baseline.json: PASS = GSM8K strict mean >= 0.9621, real AL >= 4.608, needle 3/3.
- launch.sh: starts one job as ONE capped gpurun lease in the node's p1 container (GPUS default 4, CAP default 2100 s).
- print_env.sh <A|B>: what the server gets (no GPU). check_recipe.sh: CPU validation (all pass 2026-11-19 17:5x UTC).

## Commands (from the workstation; every job is its own lease; never run two jobs on the same GPUs)
    ~/.superagent/bin/crsn 034 <<'X'
    GPUS=0,1,2,3 CAP=2100 bash /shared_nfs/ygan/p1/milestone/recipe/launch.sh gate B ms_gate_B1
    X
    # AgentX c32 pair, same node, same time window, one arm per NUMA half; then repeat with halves swapped:
    ~/.superagent/bin/crsn 034 <<'X'
    GPUS=0,1,2,3 CAP=2100 bash /shared_nfs/ygan/p1/milestone/recipe/launch.sh point A ms_A_c32_1
    GPUS=4,5,6,7 CAP=2100 bash /shared_nfs/ygan/p1/milestone/recipe/launch.sh point B ms_B_c32_1
    X
    # results
    python3 /shared_nfs/ygan/max/bin/summarize_ix.py /shared_nfs/ygan/p1/milestone/results/ms_A_c32_1 /shared_nfs/ygan/p1/milestone/results/ms_B_c32_1
    cat /shared_nfs/ygan/p1/milestone/gates/ms_gate_B1/verdict.txt
Equivalent raw client call (what bench.sh runs): aiperf profile --scenario inferencex-agentx-mvp --url http://localhost:$PORT
--endpoint /v1/chat/completions --endpoint-type chat --streaming --model minimax-m3.1 --tokenizer <model> ... --concurrency 32
--benchmark-duration 900 --random-seed 42 ... --public-dataset semianalysis_cc_traces_weka_062126 (see bench.sh).
Kill a job early: docker exec p1 pkill -f "server-port <PORT>( |$)" (point ports 18960 for GPUs 0-3, 18980 for 4-7;
gate 18940), then confirm with bash /shared_nfs/ygan/max/bin/gpustat.sh and rocm-smi that the GPUs are free.

## Caveats (read before spending GPU time)
1. The int5 prefill-step draft pin (one-line change) is validated on CPU only (cpu_pin_test + code reading: the block
   pass sets is_prefill=False and publishes its [running_bs, T] shape before running, the same path the decode-step pin
   already ran on 4 GPUs). Its first GPU contact is the gate/point; with it, the prefill-step draft runs at exactly the
   batch the production (padded) capture list gives, so exact sizes are gate-free by construction, not by measurement.
2. Mono at cap 8 is still on the GSM8K gate list (the only prior pass, ms_g1, was cap 16 on int1, 0.9625, very thin);
   draft AR is fp32-order. Gate B before counting B's AgentX numbers.
3. 30 capture sizes (vs 20) lengthen startup and take graph memory: compare the KV block count in the two server logs.
4. Mono switches itself off while another process is on the TP GPUs (GPU guard): never run anything else on them.
5. First start on a node compiles into the node-local HOME/flydsl cache: startup is longer than the reference 3-4 min.
6. CAP 2100 s is above the 30-minute lease rule on purpose (int1_both_c32 took 23 min, gate int1_g1 24 min); it needs
   the orchestrator's approval like any long lease.
