# 455-CI results

One file per test session, `<YYYYMMDD>-result.md`, named by the session date: Saturday runs MLPerf Endpoint, Wednesday runs 8K/1K. Each holds the test image and environment, the results against the previous session, the findings, each problem's next step and the core error log of each failure. `recipes/455-CI/tools/make_report.py` writes them from the data in `recipes/455-CI/`; see [`455-CI-guide.md`](../455-CI-guide.md) §5. [`example.md`](example.md) shows the format with made-up numbers.

| Session | Scenario | Image | ATOM / aiter | Result |
|---|---|---|---|---|
| [2026-11-14](20261114-result.md) | MLPerf Endpoint | `gfx1250-atom-ci455-20261114` | `9fec99786` / `b6f1ff6b7` | 0/4 passed, ❌ 4 |
