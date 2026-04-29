# Legacy Metric Cleanup Notes

## Scope

- Cleanup date: `2026-04-09`
- Audit file: `/root/autodl-tmp/results/legacy_metric_cleanup_audit.csv`
- Targeted main-evidence areas:
  - `/root/autodl-tmp/results/paper_ready`
  - `/root/autodl-tmp/results/final_tables`
  - defense-related notes under `/root/autodl-tmp/results/paper_ready` and `/root/autodl-tmp/results/maps`
  - result-display explanation text in `/root/autodl-tmp/code/app_fixed.py`

## Important Directory Note

- `/root/autodl-tmp/results/defense` does not currently exist.
- Defense-facing cleanup was therefore applied to:
  - `/root/autodl-tmp/results/paper_ready/defense_summary.md`
  - `/root/autodl-tmp/results/maps/final_folium_defense_notes.md`

## Old Fields Replaced

- `Avg SUMO Time (s)` -> `Travel Time (s)`
- `SUMO Time (s)` -> `Travel Time (s)`
- `Avg Inference Time (s)` / `Inference Time (s)` -> `model_infer_time_s`
- `Avg Planning Time (ms)` / `Planning Time (ms)` -> `route_solve_time_s`
- Main-evidence `Planning Time (s)` is now recomputed using the current protocol:
  - `Planning Time (s) = model_infer_time_s + route_solve_time_s`
- `ΔSUMO Time (s)` -> `ΔTravel Time (s)`
- `ΔInference Time (s)` -> `Δmodel_infer_time_s`
- `sumo_time_delta` -> `travel_time_delta`

## Old Fields Retained Only As Legacy

- `analytical_travel_time_s_legacy`
  - This is the only allowed retained form of historical `true_travel_time`.
  - It must not be interpreted as live SUMO `Travel Time (s)`.
- `travel_time` and `infer_time` aliases in `/root/autodl-tmp/code/app_fixed.py`
  - Kept only for backward-compatible readers/UI consumers.
  - Canonical evidence fields remain `travel_time_s`, `model_infer_time_s`, `route_solve_time_s`, and `planning_time_s`.
- `planning_ms`, `plan_ms`, `time_s`, `sumo_time`, `true_travel_time` in old raw artifacts
  - Retained only inside historical run outputs, legacy evaluators, or mapping documentation.

## Historical Results No Longer Recommended As Main Paper Evidence

- `/root/autodl-tmp/code/results/eval_summary.csv`
- `/root/autodl-tmp/results/eval_summary.csv`
- `/root/autodl-tmp/results/runs/**/metrics.csv`
- `/root/autodl-tmp/results/runs/**/eval_report.txt`
- `/root/autodl-tmp/results/runs/**/summary.json`
- `/root/autodl-tmp/results/runs/**/config_snapshot.json`
- `/root/autodl-tmp/results/runs/**/compare_results.json`
- `/root/autodl-tmp/code/evaluate.py` outputs based on `true_travel_time`
- `/root/autodl-tmp/code/.ipynb_checkpoints/**`

These materials are preserved for traceability and provenance only. They should not be cited as the main evidence in the paper or defense because they still contain legacy fields, mixed timing semantics, or analytical travel-time proxies.

## Files Unified To The Current Protocol

- Main tables:
  - `/root/autodl-tmp/results/paper_ready/main_results.csv`
  - `/root/autodl-tmp/results/paper_ready/main_results.md`
  - `/root/autodl-tmp/results/paper_ready/main_results.tex`
  - `/root/autodl-tmp/results/paper_ready/final_main_results.csv`
  - `/root/autodl-tmp/results/paper_ready/final_ablation_results.csv`
  - `/root/autodl-tmp/results/final_tables/method_comparison.csv`
  - `/root/autodl-tmp/results/final_tables/ablation_study.csv`
- Supporting appendix / narrative files:
  - `/root/autodl-tmp/results/paper_ready/appendix_full_table.csv`
  - `/root/autodl-tmp/results/paper_ready/defense_summary.md`
  - `/root/autodl-tmp/results/paper_ready/experiment_analysis.md`
  - `/root/autodl-tmp/results/paper_ready/delta_analysis.md`
  - `/root/autodl-tmp/results/paper_ready/gat_case_analysis.md`
  - `/root/autodl-tmp/results/paper_ready/final_method_notes.md`
  - `/root/autodl-tmp/results/final_tables/field_name_mapping.md`
  - `/root/autodl-tmp/results/final_tables/table_notes.md`
  - `/root/autodl-tmp/results/maps/final_folium_defense_notes.md`

## Non-Negotiable Rules After Cleanup

- Do not write `true_travel_time` as SUMO-measured travel time.
- Do not use `planning_ms` / `plan_ms` directly as paper-table main fields.
- If a historical field must be retained, label it explicitly as legacy.

## Final Judgment

- Current paper / defense main-evidence metric protocol unified: `Yes`
- Unified main-evidence time fields:
  - `Travel Time (s)`
  - `model_infer_time_s`
  - `route_solve_time_s`
  - `Planning Time (s)`
- Allowed legacy-only retained field:
  - `analytical_travel_time_s_legacy`
