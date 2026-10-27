# Reset prefix cache between benchmark rounds

POST `/reset_prefix_cache` after the previous round has drained:

```sh
curl -fsS -X POST http://127.0.0.1:8000/reset_prefix_cache
```

Success: `{"success": true, "ranks": 1, "statistics_reset": false}`.
`ranks` counts DP schedulers (TP4/DP1 has one scheduler).

The manager fences request admission and considers a request drained only after
its terminal scheduler output, including cancelled requests. It checks every scheduler before clearing each
GPU prefix index and its state-checkpoint index. All DP schedulers must reply.
Model weights, allocated KV buffers, JIT products and CUDA graphs are retained.
This differs from the RL `clear_kv_cache` utility, which zeroes physical KV.

409 means requests are still outstanding; wait for the round to finish.
503 means the engine has not initialized. 501 means the configuration is not
supported (pipeline parallel, disaggregation or external KV/state offload).
504 means some scheduler acknowledgement did not arrive before the deadline.
500 means a reset failed. For 500/504, some indexes may already be cleared:
do not record a new benchmark round as a verified cold-cache run.

The management call is synchronous on the API event loop, like cache_stats;
it can pause HTTP processing for up to the 30-second RPC deadline. It does not
cancel or wait for active inference: such requests cause 409.

`GET /debug/cache_stats` counters stay cumulative. Validate using differences:
unique long prompt (cold), identical prompt/session (hit), reset, then identical
prompt/session again (cold). Check cached_tokens and compressed_tokens deltas,
response success and an unchanged server PID. Warm reuse must be demonstrated
before a reset can be considered validated. CPU tests do not prove GPU behavior.
