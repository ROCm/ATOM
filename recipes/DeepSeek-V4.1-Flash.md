# DeepSeek-V4.1-Flash

Implementation is in progress. The configuration/reference foundation does not
yet provide a runnable ATOM model. Serving support will be enabled only after
native weights, all 40 layers, Engram and runtime state pass their gates.

The reference revision is `dba1be0a40aa45a94ad051997016db3960a90277`.
The initial target is native ATOM on a single node of 8 MI355X GPUs.
Engram builds on ROCm/ATOM PR #2185.

| Stage | Deliverable | Status |
|---|---|---|
| P00 | Pinned checkpoint, numerical oracle and input fixtures | Complete: 48 shards, 96085 tensors, 7 checks passed |
| P01 | Nested configuration, CSA2 topology and format schema | Complete |
| P02 | Native FP8/FP4 weights and kernel interfaces | Pending |
| P03 | Single-Pass mHC, MoE semantics and complete Engram | Pending |
| P04 | Full-layer exact text inference | Pending |
| P05 | Paging, batching and request-state lifecycle | Pending |
| P06 | Chat, tools and reasoning-effort protocol | Pending |
| P07 | Vision and image requests | Pending |
| P08 | Multimodal chunking and embedding lifetime | Pending |
| P09 | Packed cache, native kernels and graph execution | Pending |
| P10 | V4.1 DSpark and accepted-prefix state commit | Pending |
| P11 | Candidate-only indexing, Engram residency and fusion | Pending |
| P12 | Optional CED decoder replay | Pending |
| P13 | Optional encoder replay and persistent global cache | Pending |
| P14 | Validated distributed/deployment combinations | Pending |

The full-layer path is the numerical and performance baseline. CED and bounded
replay are approximate modes with separate quality gates and cache provenance.
Model math, cache/state ownership, Engram lookup/runtime, preprocessing, and
serving protocols are separate responsibilities.

See `tests/models/deepseek_v41/README.md` for reference validation. Accuracy will
use `lm_eval`; no model accuracy or performance result is claimed yet.

The first attention implementation reuses V4 BF16 sparse-attention kernels.
V4.1 cache values retain their reference quantize/dequantize semantics before
BF16 storage; native packed attention remains a later performance stage.

P01 registers the configuration and attention family. Model/backend execution
registration follows P04/P05 when those implementations are usable; selecting
CSA2 currently fails explicitly rather than invoking the incompatible V4 backend.
