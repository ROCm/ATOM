# Agentic routing

`agentic` places each request on the worker (`url@rank`) with the lowest
prefill cost, so a long multi-turn prompt stays where its KV lives until that
worker's queue costs more than prefilling the prompt elsewhere:

```text
uncached_tokens = max(prompt_tokens - cached_tokens, 0)
cost = inflight_prefill_tokens + uncached_tokens
     + request_token_equivalent * inflight_requests
```

The minimum cost wins, ties at random. A hit saves only the prefill it covers,
with no extra bonus, as in the engine's `least_tokens` DP balancer.

| Input | Source |
| --- | --- |
| `prompt_tokens` | The request's tokens (`--model-path` or `--tokenizer-path`, prompts up to `--prepare-max-tokenize-bytes`); without them, its text bytes / 4. |
| `cached_tokens` | The KV event index (`--kv-events`): the leading prompt tokens each worker's GPU cache holds. A request the index cannot match (no tokens, media, or no candidate's events indexed) uses a text tree of past placements: the worker with the longest text match is credited its bytes / 4. The tree shares `--eviction-interval` and `--max-tree-size` (a per-worker cap) with `cache_aware`. |
| `inflight_prefill_tokens` | Tokens charged to the worker whose prefill has not finished, `mesh_worker_prefill_tokens{worker}`. |
| `inflight_requests` | `mesh_worker_requests_active{worker}`. |
| `request_token_equivalent` | `--agentic-request-token-equivalent` (512). Keep it equal to the engine's `ATOM_DP_LB_REQ_EQUIV`. |

The selected worker is charged its `uncached_tokens`, released when its P/D
prefill response completes or at a streaming response's first frame.
Non-streaming regular, gRPC and ext-proc single-worker requests are not
charged, so their prefills weigh only through `inflight_requests`. Under
idx2idx the planner selects decode first, so `agentic` chooses among the
prefill workers on its rank. For a request the index can match, `agentic`
predicts the selected worker's blocks (`on_routed`) and reports an estimated
hit ratio, reconciled with the cached tokens the engine reports.

## Branches

| Branch | Meaning |
| --- | --- |
| `hit`, `relocate`, `miss` | By the index: the selected worker holds the longest prefix; another worker holds a longer one but its queue outweighs the reuse; no candidate holds any. |
| `text_hit`, `text_relocate`, `text_miss` | The same, by the text tree. |
| `unmatched` | Neither tokens nor text. |

## Requirements and limits

- Prefill and regular pools only: configuration rejects `agentic` for decode,
  including `--policy agentic` in P/D mode without `--decode-policy`.
- DP engines need `--dp-aware` for their KV events to be followed. Messages and
  Responses requests carry `data_parallel_rank` only with the `mesh.apis` label;
  otherwise the engine picks a rank and the router charges `agentic`'s choice.
- Token routing wants `--prepare-max-tokenize-bytes` around 4 MiB and
  `--tokenizer-cache-enable-l1`; longer prompts use the text tree.
- Not covered: cache tiers below the GPU (Mooncake Store, LMCache), atomic
  selection and charge, and a session-owner layer.

```sh
atomesh launch --pd-disaggregation --prefill http://p:8000 --decode http://d:8000 \
  --model-path <model> --kv-events --prepare-max-tokenize-bytes 4194304 \
  --tokenizer-cache-enable-l1 --prefill-policy agentic --decode-policy dp_sticky
```
