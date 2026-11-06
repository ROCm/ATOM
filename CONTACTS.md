<!-- Maintained by the ATOM team. Please open an issue or ask a listed contact to change this page. -->

# ATOM — Area Contacts

> **DRAFT** — this page is being trialed and may change. Feedback is welcome in an issue.

This page tells contributors who to contact for a pull request to ATOM. Each area has a contact and one or more backups. If a change touches several areas, start with the area of the main change. The contact makes sure your pull request gets a reviewer; they may hand the review to someone else on the team.

## How to get your pull request reviewed

1. Open the pull request against `main` and make it ready for review (see [Pull Request Process](CONTRIBUTING.md#pull-request-process)).
   Pull requests into other branches, drafts, and titles marked WIP or DO NOT MERGE are not picked up.
2. A reviewer is usually assigned within one working day after the pull request is ready.
   You can also mention the contact for the area your change touches (see the table below);
   if it touches several areas, mention the contact of the main one.
3. Most pull requests get a first review within a few working days. The exact time depends on the size and
   risk of the change and on current project priorities; fixes for regressions and release blockers come first.
4. If nobody has responded for about a week, mention the area contact, then the backup contact.

## Areas

| Area | Paths | Contact | Backup(s) |
|---|---|---|---|
| Engine core (scheduler, model runner, KV block management) | `atom/model_engine/` | Honglie Yi ([@yhl-amd](https://github.com/yhl-amd)) | Yiting Wang ([@yitingw1](https://github.com/yitingw1)) |
| Context parallelism (DCP / PCP) | `atom/distributed/dcp*`, `atom/distributed/pcp*`, `atom/model_ops/dcp*` | Yiting Wang ([@yitingw1](https://github.com/yitingw1)) | Lirong Zhang ([@ZhangLirong-amd](https://github.com/ZhangLirong-amd)) |
| Speculative decoding / MTP | `atom/spec_decode/`, `atom/models/*_mtp.py` | Lirong Zhang ([@ZhangLirong-amd](https://github.com/ZhangLirong-amd)) | Honglie Yi ([@yhl-amd](https://github.com/yhl-amd)) |
| Attention backends | `atom/model_ops/attention*`, `atom/model_ops/*mla*` | Jiayun Yu ([@jiayyu](https://github.com/jiayyu)) | Haonan Wang ([@HaonanWang98](https://github.com/HaonanWang98)) |
| MoE / expert parallelism (incl. WideEP) | `atom/model_ops/moe*`, `atom/model_ops/fused_moe*` | Jiaoliang Yu ([@JiaoliangYu](https://github.com/JiaoliangYu)) | Lirong Zhang ([@ZhangLirong-amd](https://github.com/ZhangLirong-amd)) |
| PD disaggregation / KV transfer / offload / mesh | `atom/kv_transfer/`, `atom/mesh/` | Lingpeng Jin ([@valarLip](https://github.com/valarLip)) | Honglie Yi ([@yhl-amd](https://github.com/yhl-amd)), Yajie Zhang ([@Jasen2201](https://github.com/Jasen2201)) |
| Model enablement | `atom/models/`, `atom/model_loader/`, `atom/model_config/`, `atom/multimodal/` | Jiayun Yu ([@jiayyu](https://github.com/jiayyu)) | Haonan Wang ([@HaonanWang98](https://github.com/HaonanWang98)) |
| API server / entrypoints | `atom/entrypoints/`, `atom/structured_output/` | Honglie Yi ([@yhl-amd](https://github.com/yhl-amd)) | Lingpeng Jin ([@valarLip](https://github.com/valarLip)) |
| vLLM / SGLang plugins | `atom/plugin/` | Hattie Wu ([@wuhuikx](https://github.com/wuhuikx)) | Lingpeng Jin ([@valarLip](https://github.com/valarLip)) |
| CI, Docker and recipes | `.github/`, `docker/`, `recipes/` | Xin Huang ([@gyohuangxin](https://github.com/gyohuangxin)) | Zufa Yu ([@zufayu](https://github.com/zufayu)) |
| Overall coordination, and anything not listed above | — | Lingpeng Jin ([@valarLip](https://github.com/valarLip)) | Zufa Yu ([@zufayu](https://github.com/zufayu)) |

## Community reviewers

None yet. Contributors who want to help review are welcome — see [REVIEWERS.md](REVIEWERS.md).
