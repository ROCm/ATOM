### Inconclusive -- not enough data

A/A on MI308, commit `ccd383435`

| Model | Levels &le; -2.0% | Total Tput | TPOT | TTFT |
|---|---|---|---|---|
| DeepSeek-V4-Flash | insufficient | | | |

<details>
<summary>Per-level breakdown</summary>

Italic levels are measured but not judged (c < 64).

| Model | Concurrency | Total Tput | TTFT | TPOT | Drift |
|---|---|---|---|---|---|
| DeepSeek-V4-Flash | 64 | +0.3% | -1.3% | -0.4% | +0.2% |
| | **median of judged** | insufficient | | | |

</details>


> Identical code measures 0.27%, individual levels within 1.8 points of that. The verdict takes the family median for that reason.

No model reported enough judged levels. This is **not** a pass -- treat it as no signal.

Coverage gaps: 1 model(s) reported too few judged levels.


Advisory. This check does not block merge.