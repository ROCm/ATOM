# 455-CI results

One file per test session, `<YYYYMMDD>-result.md`, named by the session date: Saturday runs MLPerf Endpoint, Wednesday runs 8K/1K. Each holds the test image and environment, the results against the previous session, the findings, each problem's next step and the core error log of each failure. `recipes/455-CI/tools/make_report.py` writes them from the data in `recipes/455-CI/`; see [`455-CI-guide.md`](../455-CI-guide.md) §5. [`example.md`](example.md) shows the format with made-up numbers.

| Session | Scenario | Image | ATOM / aiter | Result |
|---|---|---|---|---|
