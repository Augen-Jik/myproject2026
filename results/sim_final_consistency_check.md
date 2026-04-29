# Final Consistency Check for Simulation Mainline

## Scope

Checked inputs:

- `/root/autodl-tmp/code/app_fixed.py`
- `/root/autodl-tmp/code/sim_scene_profiles.py`
- `/root/autodl-tmp/code/apply_scene_profile_to_sumo.py`
- `/root/autodl-tmp/code/sim_eval_protocol.py`
- `/root/autodl-tmp/results/legacy_metric_cleanup_notes.md`
- `/root/autodl-tmp/results/paper_ready/final_main_results.csv`
- `/root/autodl-tmp/results/defense/final_defense_onepage.md` if present

Actual defense-material check result:

- `/root/autodl-tmp/results/defense/final_defense_onepage.md` is not present in the current workspace.
- Current defense files inspected instead:
  - `/root/autodl-tmp/results/defense/simulation_setup_onepage.md`
  - `/root/autodl-tmp/results/defense/simulation_setup_qa.md`

## 1. Protocolized time fields on the mainline

Status: PASS

Mainline evidence and code paths are aligned to the protocolized timing semantics.

- `final_main_results.csv` uses the canonical paper fields:
  - `Travel Time (s)`
  - `model_infer_time_s`
  - `route_solve_time_s`
  - `Planning Time (s)`
- `app_fixed.py` explicitly computes:
  - `route_solve_time_s`
  - `model_infer_time_s`
  - `planning_time_s = model_infer_time_s + route_solve_time_s`
  - `travel_time_s` as the SUMO trip duration result
- `sim_eval_protocol.py` defines:
  - `Planning Time (s)` as model inference plus route solving only
  - `Travel Time (s)` as the realized SUMO trip duration only

Observed legacy compatibility behavior:

- `app_fixed.py` still retains `travel_time` and `infer_time` as backward-compatible aliases for UI/history readers.
- These aliases are not used as the canonical paper-table fields.
- This is acceptable as long as paper and defense evidence continue to cite only the protocolized fields above.

## 2. Whether environment variables are injected from scene profile

Status: PASS

The current mainline consistently uses `scene_profile` as the source of environment-side simulation control.

- `app_fixed.py` resolves the active scene profile before simulation.
- `SUMOSimulationEngine.run_simulation(...)` constructs `SUMOSceneProfileApplier(config, scene_profile)`.
- `apply_scene_profile_to_sumo.py` uses the scene profile to build SUMO command options and apply runtime window effects.
- `sim_scene_profiles.py` centralizes exogenous environment parameters, including traffic controls, demand, incidents, closures, seed, warm-up, and evaluation window.

Conclusion:

- Environment-side changes are injected from the scene profile rather than scattered ad hoc in the app layer.

## 3. Whether reroute is uniformly controlled by scene configuration

Status: PASS

Reroute is scene-controlled, not model-controlled.

- `sim_scene_profiles.py` stores reroute parameters in the scene definition:
  - `reroute_enabled`
  - `reroute_period`
  - `reroute_threshold_factor`
  - `reroute_threshold_constant`
- `apply_scene_profile_to_sumo.py` converts those scene fields into reroute behavior and periodic checks.
- `sim_eval_protocol.py` fixes `reroute_trigger_mode = periodic`.
- `app_fixed.py` records the scene reroute policy into the result payload.

Conclusion:

- The current chain is consistent with the intended rule that reroute belongs to the environment/scenario protocol.

## 4. Whether legacy fields are still being used as primary evidence in documents

Status: PASS WITH NOTE

Primary evidence files checked in this pass do not treat legacy fields as the main evidence basis.

- `final_main_results.csv` is already migrated to protocolized headers.
- Current defense files use `Planning Time (s)` semantics and do not present `planning_ms`, `plan_ms`, or `Avg SUMO Time` as the main evidence field names.
- `legacy_metric_cleanup_notes.md` correctly states that legacy fields may remain for compatibility but should not be cited as main evidence.

Note:

- `legacy_metric_cleanup_notes.md` contains historical cleanup context saying the defense directory did not exist at that earlier cleanup step.
- This statement is now time-specific background, not a current evidence claim.
- It does not create a paper/defense metric conflict by itself.

## 5. Whether defense materials still contain metric conflicts

Status: PASS

No active defense-material metric conflict was found in the currently present defense files.

- `simulation_setup_onepage.md` clearly separates:
  - model-side responsibilities
  - environment-side responsibilities
  - `Planning Time (s)` versus `Travel Time (s)`
- `simulation_setup_qa.md` uses the same split and keeps reroute, speed, TLS, and incident changes under scene control rather than model control.
- No checked defense file presents `true_travel_time` as realized SUMO travel time.
- No checked defense file presents `planning_ms` or `plan_ms` as the primary defense-table metric.

Boundary condition:

- Because `final_defense_onepage.md` does not exist, this check applies to the currently available defense materials in `/root/autodl-tmp/results/defense`.

## Overall judgment

The current mainline is internally consistent on the points that matter for paper submission and defense presentation:

- protocolized time fields are the canonical evidence fields
- environment variables are injected through scene profiles
- reroute is controlled by the scene protocol
- legacy fields remain only as compatibility artifacts, not as the recommended evidence basis
- current defense materials do not show an active metric-definition conflict

Residual caution:

- If a new defense summary file is created later, it should continue using only:
  - `Planning Time (s)`
  - `Travel Time (s)`
  - `model_infer_time_s`
  - `route_solve_time_s`
- Any retained historical aliases should be marked `legacy` and should not be elevated back into headline evidence tables.

READY_FOR_PAPER_AND_DEFENSE
