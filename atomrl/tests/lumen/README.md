# Tests

## Running Tests

```bash
# Run the complete suite in a ROCm GPU image, from the ATOM repository root.
python -m pytest atomrl/tests/lumen -v

# CPU-safe coverage used by CI.
python -m pytest \
  atomrl/tests/lumen/patches/test_registry.py \
  atomrl/tests/lumen/patches/test_cli.py \
  atomrl/tests/lumen/models/test_launcher_parity.py \
  atomrl/tests/lumen/models/test_training_stack_policy.py -v
```
