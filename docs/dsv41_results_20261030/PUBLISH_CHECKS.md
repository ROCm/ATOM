# Publication checks — 2026-10-30

The requested publication includes ATOM only. Its main target is
`124c3b0d1636919f03513c33d3e414d2112d2cb0`. The original local HEAD is
`96405d8609abdb0e9be594b47dcd4a492e81c6d4`; the original remote feature tip is
`f12ea9a63777d05c9f5a53fd8bb2b8a144a11c9b`. Local backup branches retain both.

The remote-only commit's three-file MoE padding correction and tests are
retained. Its old BF16 dashboard environment change is omitted: the current
FP8 catalog and opt-in owner entry remain the deployment contract.

Rebase completed without conflicts. `git range-diff` reports all ten original
project commits patch-equivalent on the new main. The imported padding fix is
now `e878f5a21`; the initial publication commit is `4a8726d05`. The original
dashboard conflict was resolved by retaining the current FP8 catalog.

Completed checks:

| Check | Result |
|---|---|
| Public profile / GPU-preflight guard / owner-entry CPU tests | 8 passed |
| Owner EP contract | 1 passed |
| Owner lifecycle / graph retirement order | 2 passed |
| Owner MoE route/expert boundary after rebase | 4 passed |
| Owner deployment admission | 6 passed |
| Source-executed padding CPU smoke checks | 4 passed: TP4 intermediate576 → 640 for alignment128; alignment256 → 768; zero alignment refused; runner receives padded method dimensions |
| Python syntax for files changed from main | 103 files passed |
| Black / Ruff for imported padding code, its tests and A0 snapshot | 13 files passed |
| Published evidence and A0 source hashes | 11 result records + 14 source files equal original bytes and recorded SHA256 |
| New summary / index Markdown links | 41 local targets exist |
| Whitespace / conflict check | `git diff --check origin/main HEAD` passed |
| Rebase ancestry | main is an ancestor; all ten earlier project patches preserved |

The 21 unittest cases can be run without Torch:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests/tools -p 'test_dsv41_*.py'
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests/models/deepseek_v41 -p test_owner_ep.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests/models/deepseek_v41 -p test_owner_lifecycle.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests/models/deepseek_v41 -p test_owner_moe_rebase.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests/models/deepseek_v41 -p test_owner_service_contract.py
```

The padding smoke check extracted the actual factory and initializer AST into
an isolated namespace with host/runtime dependencies stubbed. It exercised
dimension selection and runner arguments; it did not launch a kernel or test
tensor arithmetic. The retained `tests/model_ops/test_comm_fused_moe.py` tests
require Torch/pytest, neither installed on this host, and were not run here.

The first Ruff pass identified five import-order differences because the
snapshot moved from outside ATOM into its source tree. The snapshot bytes were
preserved. Its narrow `ruff.toml` extends the repository configuration and
classifies `atom` as the external dependency it was in the original harness;
no lint rules were disabled. Black was checked with target Python3.12 to match
the local interpreter.

Native interface review retained upstream's TP pad-row routing change, new
Linear scale dtype/output-padding guards, shared Engram mapping helper and
compiled FlyDSL dispatch. The frozen AITER dependency has `_run_compiled`.
This is source compatibility review, not a runtime or performance certificate.
The old A0 launcher remains pinned to the original runtime; a new experiment
revision must qualify the rebased interfaces before any future GPU execution.

No GPU benchmark, full-model accuracy evaluation or historical reviewer was
run during publication. Old service/numerical results retain their original
code/dependency/configuration scope. A0 remains GPU unrun and is not enabled.
The branch is pushed with an exact lease against the original remote feature
tip, preserving protection against concurrent remote updates.
