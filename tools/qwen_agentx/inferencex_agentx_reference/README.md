# InferenceX reference validators

These files are unchanged copies from SemiAnalysisAI/InferenceX revision
`77a21c0aa849d1bd3cdcb23ee33554ebeb4d88e4`:

- `inferencex-e2e/infx/results/agentic/validate_agentic_result.py`
- `inferencex-e2e/infx/results/agentic/analyze_benchmark_distributions.py`

Source: https://github.com/SemiAnalysisAI/InferenceX/tree/77a21c0aa849d1bd3cdcb23ee33554ebeb4d88e4/inferencex-e2e/infx/results/agentic

The request-error validator is one check. The runner additionally checks the
AgentX scenario's `metadata.submission_valid`. These scripts do not implement
InferenceX power collection or its complete submission audit.
