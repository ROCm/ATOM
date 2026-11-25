# Mooncake #5840 validation

Candidate: https://github.com/kvcache-ai/Mooncake/pull/5840 at
`8babd5a9b3ecfe0df78379ca0f1f143a391edd44`. Only its three production
files are backported onto v0.3.14 (`e1ea5d31bb5e46ff2df9f90e7c2fbaf52fcdee8c`).
This is an existing upstream Draft PR, not a new submission or a confirmed fix.

Both wheels were built on the jump host CPU using the original worker filesystem,
identical TE-only CMake configuration, GNU 11.4.0 and HIP 7.15.26333. Both passed
10 endpoint state tests and a Python3.12 import. Store/etcd/Rust/EP are disabled;
HIP, HIP DMA-BUF, multi-protocol and metrics remain enabled. The rebuilt baseline
must reproduce before improvement can be attributed to the patch.

The two contracts replace only the producer wheel. D remains on the original
fixed image. P=PP4, D=DP4+EP4, C80, D max-num-seqs20/rank, DSpark7/capture160,
sender10/20 async tasks, native timeout variables unset. P order0,1,3,2 is retained
as a reproduction control. Natural order still needs validation after a pass.

Each test acquires and releases its own 8-GPU lease. Stop at the first native
failure and preserve the failed exit status. Otherwise require protocol/real-KV
checks, original AgentX warmup and 900s observation; this is not an SA sweep point.

The staging helper verifies wheel and source hashes, builds offline, and verifies
the installed engine, connector and original vLLM kernel hashes. It adds the same
bounded Python registration/failure logging to both producers. It never compiles
or loads the old C++ interposer or runs the fixed-offset GDB decoder.

Binary artifacts are in the task-owned shared directory named by the contracts.
`manifest.json` records wheel/engine hashes and build limitations. Run
`python -m unittest discover -s .github/scripts/atomesh -p 'test_mooncake_rebuild.py'`
and the adjacent `test_timeout_probe.py` before dispatch.
