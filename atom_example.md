# Atom

A lightweight vLLM implementation built from scratch.

## Installation

```bash
git clone https://github.com/valarLip/atom.git
cd atom
```

## Example

```bash
python example.py --model meta-llama/Meta-Llama-3-8B
```

The default level is 3(running with torch compile), the model we support now is Qwen, Llama, Mixtral, you can set the model name or path using --model

## Perf running

```bash
python bench_test.py --model Qwen/Qwen3-0.6B
```

You can get the performance detail:
```bash
Total: 4096tok, Time: 0.31s, Throughput: 13333.85tok/s
```