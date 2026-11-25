# AgentX cancellation-drain reproduction

This CPU-only diagnostic uses unchanged methods from agentx-harness 1.0.14.
It distinguishes a missing cancellation acknowledgement from an acknowledgement
that has been counted while a future DAG join still blocks phase completion.

Download and extract the pinned wheel without installing the benchmark client:

```sh
python -m pip download --no-deps --python-version 3.12 agentx-harness==1.0.14 --dest wheels
python -m zipfile -e wheels/agentx_harness-1.0.14-py3-none-any.whl extracted
python .github/benchmark/pr5090-router-ab/diagnostics/repro_agentx_cancel_drain.py extracted/aiperf result.json
```

Run with Python 3.12. The wheel SHA256 is
`43707f5377ef8095d9afec60aab1ecc8782aa878d022e528332a0b793d56ceaf`.
The script checks all five source-file hashes before loading their ASTs.
No GPU, server, credentials or installed AgentX dependencies are needed.

| Case | Acknowledgements | Future join | Drain timeout |
|---|---:|---|---|
| No future join | All five, asynchronous | Absent | No |
| Observed boundary state | All five, asynchronous | Remains | Yes, despite zero in-flight credits |
| Join drains after returns | All five, asynchronous | Removed with its drain notification | No |
| Returns processed before the cancellation call returns | All five, synchronous | Remains | No, existing `need <= 0` branch bypasses the gate |
| A return is actually missing | Four of five | Absent | Yes, with one in-flight credit |

Each case runs ten times and asserts identical results. The local timer is 50ms
instead of the live ten seconds. Counting, return callbacks, branch predicates,
interception and phase-drain logic are real methods; transport, metadata and
publication use in-memory fixtures. The test starts from the observed future-join
state. It does not reconstruct the entire DAG or prove how that state originated.
Removing the join is a diagnostic control, not a proposed unconditional cleanup fix.

The PP/DEP C80 run and the reused SA TP C80 run both retained a future join at
turn 74 of trace `578354185933aedcdc6c17fbfbfcfb48876b`. The PP/DEP run counted five
cancellation returns, then reported a drain timeout. The SA run had no outstanding
wire requests when it entered cancellation and took the `need <= 0` bypass.
The source methods inspected in upstream master
`18015a2d77a6077bee9750c6929a889d52c7df54` are unchanged from the pinned wheel.

This is a client completion-state diagnostic. It does not explain the separate
Mooncake PP3 transfer timeout with ten producer sender workers, alter the measured
GPU runs, or change their SA metric calculations.
