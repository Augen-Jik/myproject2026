# Final Method Notes

## Why The Main Method Is `Sparse-LoRA-v2`
- `gat_v2_final_positioning.md` explicitly locks the paper-facing method name to `Sparse-LoRA-v2` and rejects any default method name that contains `GAT`.
- In the non-GAT mainline, the sparse model remains the strongest practical choice: it keeps the low inference cost of the sparse branch and avoids over-claiming an unstable GAT gain.
- The old `Sparse-LoRA` table label is therefore frozen to the paper name `Sparse-LoRA-v2` in the final outputs.

## Why `stage4_fix` Is The Final Upstream Baseline
- `stage4_fix` is the last sparse checkpoint that remained stable on the held-out guard slices and was never invalidated by later evidence.
- The upstream decision page already marked it as the safest default baseline before the final micro-patch attempt.
- The later patch branch did not produce a safe replacement: `stage4b_fix` had no visible movement, `stage4c_fix` showed only tiny gains with new false positives, and `stage4d_micro` failed to transfer at all.
- The final upstream freeze therefore stays at `/root/autodl-tmp/model_merged_sparse_v2_stage4_fix`.

## Why Patch SFT Enters Negative Results
- The patch branch reached diminishing returns on the held-out natural distribution rather than a clean upgrade path.
- The decisive evidence is the sequence of failed targeted interventions:
- `stage4b_fix`: `29 / 29` unchanged.
- `stage4c_fix`: only `1 / 29` targeted movements and `8` new normal-only false positives.
- `stage4d_micro`: `0 / 18 no_anchor -> anchor` plus `8 / 8` hard-negative false positives.
- Because the final patch branch does not beat `stage4_fix` safely, it is preserved only as negative-results material and is excluded from the paper mainline.

## Why GAT Moves To The Appendix
- `gat_v2_final_positioning.md` and `gat_v2_release_notes.md` both conclude that GAT-v2 is not ready for the paper main table.
- The locked reason is not “GAT never helps anywhere,” but “there is no stable, paper-safe global gain that justifies making GAT the default method.”
- The GAT-v2 summary remains worse than the sparse baseline on the main validation slices, so GAT can only be kept as appendix / exploratory analysis.
- The allowed appendix naming stays: `Sparse-LoRA-v2 + Gated-GAT-v2` and `Sparse-LoRA-v2 + Always-GAT-v2`; neither may replace `Sparse-LoRA-v2` as the default paper method name.

## What The PPO Baseline Means In The Final Story
- PPO quick baseline was available and has been included as a coverage-only RL row. Its current result comes from the quick graph-wrapper protocol, so undefined fields are intentionally left blank instead of being fabricated.
- The PPO row is positioned as RL category coverage, not as a tuned PPO claim and not as evidence that PPO is the new main method.
- The row is therefore explicitly labeled `quick baseline, untuned`, and any fields that are not defined under that quick protocol remain blank on purpose.

## Locked Table Policy
- `final_main_results.csv` contains no GAT mainline row.
- Exploratory patch models do not enter the final main table.
- GAT rows may appear only in appendix/exploratory ablation bookkeeping.
- `Travel Time (s)`, `model_infer_time_s`, `route_solve_time_s`, and derived `Planning Time (s)` are the unified timing fields across the frozen outputs.
- If a historical analytical proxy must be retained, it must be named `analytical_travel_time_s_legacy` rather than `true_travel_time`.
