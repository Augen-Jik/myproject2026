# Sim Tuning Risk Check

## Scope

- Checked the live SUMO evaluation chain centered on `code/app_fixed.py`, `code/apply_scene_profile_to_sumo.py`, `code/sim_scene_profiles.py`, and `code/sim_eval_protocol.py`.
- Also scanned legacy evaluation scripts and defense-facing result notes to see whether old implicit variables or old metric definitions can still leak into paper/defense materials.

## must_fix_before_defense

- `Planning Time` / `Travel Time` are still misdocumented in defense-facing notes. `results/final_tables/table_notes.md:23-24` says planning time excludes LLM inference and says `true_travel_time` can be treated as SUMO travel time; `results/final_tables/field_name_mapping.md:7-8` repeats the same mapping. This conflicts with the current canonical protocol in `code/sim_eval_protocol.py:75-82` and `results/sim_eval_protocol_notes.md:93-95`, and it also conflicts with `code/evaluate.py:241-246`, where `true_travel_time` is only an analytical proxy. This is the clearest remaining fairness-risk item because it can make environment-side travel metrics look better or worse for the wrong reason.

- Legacy result bundles can still be mistaken for protocol-compliant evidence. Files such as `results/eval_summary.csv`, `results/compare_results.json`, and multiple `results/runs/*/metrics.csv` entries still expose old fields like `planning_ms`, `plan_ms`, `time_s`, `sumo_time`, `true_travel_time`, and `warmup_excluded=true`, but they do not carry the current `sim_eval_protocol`, `scene_reroute_policy`, or full scene snapshot now written by `code/app_fixed.py:2738-2754`. If any of those old tables are cited in the paper or defense without relabeling or regeneration, the audience can no longer tell whether seed, reroute, warm-up, and timing definitions were identical.

## acceptable_with_note

- The speed-factor mapping is still defined in two places: `code/sim_scene_profiles.py:34-44` and `code/apply_scene_profile_to_sumo.py:21-30`. The live injector uses the centralized canonical map and then merges profile overrides through `code/apply_scene_profile_to_sumo.py:137-148`, so there is no observed runtime mismatch today. Still, this duplication is a maintenance drift risk and should be called out if anyone edits the profiles later.

- A generic runtime API for manual TLS mutation still exists in `code/sumo_runner.py:134-147`. The active experiment chain does not use it; the live chain instead applies fixed scene-owned TLS programs through `code/apply_scene_profile_to_sumo.py:286-302` and records the selected TLS profile in `code/app_fixed.py:2746-2750`. The residual risk is not current unfairness, but that an ad hoc script could bypass the audited scene protocol if someone uses `sumo_runner.py` directly.

- Different scenes do use different reroute thresholds and periods, for example `code/sim_scene_profiles.py:284-305`, `code/sim_scene_profiles.py:307-330`, and `code/sim_scene_profiles.py:345-377`. In the current chain this is explicit, because reroute is scene-owned, injected from one entry point in `code/apply_scene_profile_to_sumo.py:160-203`, triggered periodically in `code/apply_scene_profile_to_sumo.py:433-463`, and persisted to results in `code/app_fixed.py:2750-2754`. The remaining note is only about older result files that predate this snapshotting.

- Historical outputs still contain abbreviated warm-up metadata such as `warmup_excluded=true` without the exact second counts. The current chain has fixed and recorded `warmup_seconds`, `evaluation_start_time`, `evaluation_end_time`, and `vehicle_depart_window` in `code/sim_eval_protocol.py:72-90` and uses them in `code/app_fixed.py:881-906`, but archived results from older runs should be treated as legacy unless they are re-exported under the new protocol.

## no_issue

- No unfixed random seed was found in the live SUMO chain. Scene profiles carry fixed `simulation_seed` values in `code/sim_scene_profiles.py:216-377`, and SUMO is launched with `--seed` in `code/sim_scene_profiles.py:477-492`.

- Warm-up and evaluation windows are fixed and recorded in the live chain. The protocol is built from the scene profile in `code/sim_eval_protocol.py:72-90`, the simulator warms up to `max(warmup_seconds, evaluation_start_time)` in `code/app_fixed.py:881-885`, and the run stops at the fixed evaluation end in `code/app_fixed.py:904-906`.

- Reroute triggering is fixed rather than model-driven in the live chain. The canonical protocol fixes `reroute_trigger_mode` to `periodic` in `code/sim_eval_protocol.py:10-13` and `code/sim_eval_protocol.py:86-89`, while the actual runtime reroute check is scene-controlled in `code/apply_scene_profile_to_sumo.py:433-463`.

- Live TLS changes are no longer hidden. The scene injector applies fixed TLS programs/phases in `code/apply_scene_profile_to_sumo.py:286-302`, and the result record stores the scene TLS profile in `code/app_fixed.py:2746-2750`.

- The current live chain no longer mixes `Planning Time` and `Travel Time` at runtime. `code/app_fixed.py:2693-2705` computes planning time as inference plus route solving, `code/app_fixed.py:2761-2767` stores canonical planning/travel fields separately, and `code/app_fixed.py:1499-1550` normalizes old display records through `code/sim_eval_protocol.py:98-134`.

当前仿真主链路已经基本足够支撑论文和答辩，但答辩前必须统一旧文档中对 `Planning Time` 和 `true_travel_time` 的口径，并避免把未协议化的历史结果当作主证据。
