# Final Negative Results Summary

## Locked Scope
- This page freezes the patch-SFT branch as negative-results-only material.
- `sparse_upstream_decision_page.md` recorded the pre-`stage4d_micro` decision to allow one last narrow patch attempt.
- `final_patch_negative_results.md` is the later and higher-priority conclusion after that final attempt; it closes the patch branch and re-freezes the sparse upstream at `stage4_fix`.

## Patch Variants Moved Out Of The Mainline
- `stage4b_fix`: no effective held-out movement. The targeted `diff_29` stayed `29 / 29` unchanged, with `no_anchor_to_anchor_count = 0 / 29`.
- `stage4c_fix`: only `1 / 29` targeted cases moved, while adding `8` new normal-only false positives.
- `stage4d_micro`: the final narrow patch still failed to transfer. It achieved `0 / 18 no_anchor -> anchor` on the target bucket and triggered `8 / 8` hard-negative false positives.

## Why These Rows Stay In Negative Results
- The branch never produced a stable held-out gain large enough to replace `stage4_fix`.
- The later patch rounds introduced side effects instead of a safe upstream upgrade.
- The final locked outcome is therefore: `stage4_fix` remains the sparse upstream; `stage4b_fix`, `stage4c_fix`, and `stage4d_micro` are retained only as negative-results evidence.

## Paper Placement Policy
- Exploratory patch models do not enter `final_main_results.csv`.
- Exploratory patch models do not enter `final_ablation_results.csv` as mainline-eligible rows.
- Patch SFT is discussed only as a negative-results branch that reached diminishing returns on the held-out natural distribution.
