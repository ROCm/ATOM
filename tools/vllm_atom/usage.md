# vLLM-ATOM Trace Parser
Parse vLLM profiler traces to extract per-layer GPU kernel breakdowns.

## Important
Please set **`VLLM_CUSTOM_SCOPES_FOR_PROFILING=1`** when capturing vLLM traces.

## Requirements
```bash
pip install openpyxl ijson # ijson is optional, used to reduce peak memory
```

## Usage
```bash
python parse_vllm_trace.py [options]
```
Traces must be captured with `VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` to include the user annotation markers the parser relies on.
| Option | Default | Description |
|---|---|---|
| `--layer N` | `8` | The transformer layer index to extract stats from. |
| `--percentile P` | `50` | Use the P-percentile batches to obtain breakdown. |
| `--eager-trace PATH` | - | Companion eager trace for extracting host module attributes. For graph-captured batches like decode workloads in `FULL_AND_PIECEWISE` settings, kernel executions in a graph are packed into a single graph replay event, losing per-kernel CPU attribution. An eager trace can be provided to establish device-side kernels with the host-side kernel launches. |
| `--output-prefix PREFIX` | - | Prefix for the output xlsx filename. The output file will be "{PREFIX}_trace_breakdown.xlsx". |

## Capture vLLM-ATOM profile traces
Profile traces of vLLM-ATOM can be collected the same way as profiling vLLM, through adding the `--profiler-config` argument to the `vllm serve` command:

```bash
export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1

vllm serve --model your/model \
    ...
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/path/to/traces/", "torch_profiler_with_stack": true}'
```

To provide companion eager mode traces, collect separate eager traces of the same settings:

```bash
export VLLM_CUSTOM_SCOPES_FOR_PROFILING=1

vllm serve --model your/model \
    ...
    --enforce-eager \
    --profiler-config '{"profiler": "torch", "torch_profiler_dir": "/path/to/eager_traces/", "torch_profiler_with_stack": true}'
```

Then pass the eager trace to the parse through the `--eager-trace` argument.