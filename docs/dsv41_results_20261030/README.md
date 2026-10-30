# Retained result snapshots

These are exact copies of selected completed records from the external
`validation/dashboard_whole_20261029/` workspace. [MANIFEST.json](MANIFEST.json)
records each original path, byte count and SHA256. Copying did not rerun a
reviewer, benchmark, numerical check or GPU experiment.

Read the [optimization summary](../dsv41_optimization_summary.md) for deployment,
denominators, failed candidates, accuracy boundaries and code versions.
The records describe historical measurements, not the rebased Git revision.
They contain references to source files, rank outputs and binaries retained
outside Git; this selected set is not a self-contained reproduction bundle.

JSON files ending in `.gz` preserve their original compressed bytes. For example:

```bash
python -m gzip -d < docs/dsv41_results_20261030/fp8_owner_service_r3/REVIEW.json.gz
```

Or load a selected record with Python's `gzip.open(path, "rt")` and `json.load`.
Do not run historical reviewers against the rebased checkout. No model
weights, caches, large traces or compiled GPU binaries are included.
