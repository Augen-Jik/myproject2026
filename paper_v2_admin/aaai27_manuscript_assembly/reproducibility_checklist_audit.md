# AAAI-27 reproducibility checklist audit

This audit records the evidence behind the answers in
`official_checklist/ReproducibilityChecklist.tex`. The checklist source itself
contains only the response tokens required by the official template.

## Answer map

| Item | Answer | Evidence / boundary |
|---|---|---|
| 1.1 | yes | Method Sec. 3 gives the factorization, L0/L1/L2 definitions, and four-stage STAG-Route pipeline. |
| 1.2 | yes | Claims are explicitly scoped in the method, robustness, reconciliation, and limitations sections. |
| 1.3 | no | Related work is cited, but references are not explicitly marked as pedagogical replication background. |
| 2.1 | no | The paper is empirical and does not claim theorem/proof contributions. |
| 2.2--2.8 | NA | Not applicable after 2.1=no. |
| 3.1 | yes | The work uses the novel STC-Bench benchmark. |
| 3.2 | yes | The paper motivates the method-agnostic discriminative stress construction and held-out topology sets. |
| 3.3 | partial | The 2,000-instance benchmark, oracle artifacts, splits, hashes, and data card are in the T4 package, not in a conventional paper data appendix. |
| 3.4 | partial | Release materials are prepared, but the repository does not yet contain an explicit research-use license. |
| 3.5--3.6 | NA | The evaluated datasets are introduced by this work rather than imported from prior literature. |
| 3.7 | yes | The paper and `DATA_CARD.md` describe the synthetic benchmark and why a discriminative construction is needed. |
| 4.1 | yes | The paper reports computational experiments. |
| 4.2 | no | Final settings are documented, but complete development sweep ranges and selection criteria are not stated for every hyperparameter. |
| 4.3 | partial | Preprocessing/generation code is in the artifact package, not a conventional code appendix. |
| 4.4 | partial | Analysis and frozen-output rebuild code are present; the full competitive-table script still contains placeholder runner calls. |
| 4.5 | partial | Public release is intended, but an explicit free research-use license is not yet present. |
| 4.6 | partial | Core code is commented, but not every implementation step links back to a paper section. |
| 4.7 | yes | The canonical model and interface ablations use seed 1337; deterministic inference is recorded in the T4 materials. |
| 4.8 | partial | `environment.yml` and `requirements.lock` pin software, and run records identify an RTX 4080 with 32 GB; CPU/OS details are incomplete in the paper. |
| 4.9 | partial | CR, CS, travel time, grounding, schema, and transfer metrics are described, but not every metric receives a fully formal definition and motivation. |
| 4.10 | no | Sample counts are given, but the run count is not stated for every reported method/result. |
| 4.11 | yes | The paper reports a paired 95% CI, medians, tail proportions, bucket breakdowns, and per-map results. |
| 4.12 | no | The submission does not claim inferential significance from an appropriate hypothesis test. |
| 4.13 | partial | Canonical STAG hyperparameters are in `MODEL_CARD.md`; final settings are not exhaustively listed for every baseline. |

## T4 package inventory used

- `DATA_CARD.md`: STC-Bench semfix_full v1, n=2,000, split/provenance notes.
- `MODEL_CARD.md`: canonical Qwen2.5-1.5B-Instruct LoRA adapter, training recipe, prompt contract, and SHA256.
- `artifact_manifest.json` plus `tools/verify_manifest.py`: measured hashes, sizes, and roles.
- `requirements.lock` and `environment.yml`: pinned software environment.
- `scripts/reproduce_offline.sh`: GPU-free manifest verification and frozen-output table rebuild.
- `scripts/reproduce_main_table.sh`: path verification is executable; GPU inference and SUMO runner calls are currently documented placeholders, which is why checklist item 4.4 is `partial` rather than `yes`.

## Release actions needed to upgrade partial answers

1. Add an explicit research-use license for the benchmark and source code.
2. Replace the placeholder GPU/SUMO commands in `reproduce_main_table.sh` with the canonical executable runners.
3. Add CPU, OS, CUDA, SUMO, and exact library versions to the paper or supplement.
4. State run counts and final hyperparameters for every baseline.
