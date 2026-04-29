# SUMO Scene Profile Unification Notes

## 1. Related files discovered in the current project

- `/root/autodl-tmp/code/build_urban_net.py`
  - Generates the SUMO road network, default lane counts, base speeds, TLS type, and the original route/flow template.
  - The historical flow template is defined around the `flows = [...]` block and was the closest thing to a demand preset.
- `/root/autodl-tmp/code/app_fixed.py`
  - Starts SUMO through TraCI, injects the evaluation vehicle, and used to derive environment speeds directly from temporary scene parsing.
  - This is now the runtime entry that resolves and applies a fixed `scene_profile`.
- `/root/autodl-tmp/code/sumo_runner.py`
  - Exposes TraCI helpers for reading edge state and mutating traffic-light state.
  - Important for review because TLS mutation exists here and should remain outside model control.
- `/root/autodl-tmp/code/scenarios.py`
  - Contains UI scenario presets and scene taxonomy.
  - It now maps each preset to a fixed `scene_profile`.
- `/root/autodl-tmp/code/config.py`
  - Runtime config bootstrap.
  - It now exposes `DEFAULT_SCENE_PROFILE` and the notes path.
- `/root/autodl-tmp/SUMO/config/my_config.sumocfg`
  - Base SUMO config file.
- `/root/autodl-tmp/SUMO/routes.rou.xml`
  - Currently empty in the checked workspace, so background demand is now generated per scene profile at runtime instead of being implicitly read from this file.
- `/root/autodl-tmp/code/sim_scene_profiles.py`
  - New unified scene configuration layer added for this task.

## 2. What is now decided by the scene configuration layer

The following parameters are defined by `SimSceneProfile` in `/root/autodl-tmp/code/sim_scene_profiles.py`:

- `scene_name`
- `speed_profile`
- `congestion_level_to_speed_factor`
- `incident_edges`
- `blocked_edges`
- `lane_reduction_edges`
- `tls_profile_name`
- `tls_fixed_program_id`
- `reroute_enabled`
- `reroute_period`
- `reroute_threshold_factor`
- `reroute_threshold_constant`
- `demand_profile_name`
- `depart_rate`
- `simulation_seed`
- `warmup_seconds`
- `evaluation_start_time`
- `evaluation_end_time`

In the current runtime, these scene-config parameters concretely control:

- Background demand generation through a per-scene runtime route file.
- Fixed SUMO random seed through `--seed`.
- Fixed TLS program assignment through `tls_fixed_program_id`.
- Fixed edge speed/capacity penalties derived from incidents, blockages, and lane reductions.
- Fixed warmup and evaluation window before the planner vehicle is injected.
- A scene-owned reroute policy object. This policy is fixed in the profile even when the current evaluator keeps the planner vehicle route unchanged after selection.

Predefined scene templates currently include:

- `normal_baseline`
- `simple_local`
- `directional_asymmetry`
- `core_blockage`
- `propagation_range`
- `compound_disaster`
- `temporal_switch`
- `anti_truncation_eval`

## 3. What is decided by the model

The model is only allowed to decide:

- Edge weight / cost estimation for planning.
- The final route selected by the path-planning algorithm from those costs.

The model is not allowed to directly modify:

- TLS program selection
- Vehicle demand injection profile
- SUMO random seed
- Exogenous speed restrictions
- Incident lists
- Blocked edges
- Lane reductions

In other words: red lights, speed limits, closures, accidents, demand, and simulation randomness are environment variables, not model outputs.

## 4. Why the two must be decoupled

- Reproducibility: different models must face the same fixed environment, otherwise travel-time comparisons are not meaningful.
- Fairness: if a model can change TLS, seed, or demand, it can improve results by altering the testbed rather than improving route choice.
- Causal clarity: the planner should only be judged on whether it estimates costs well and chooses a good path under a given environment.
- Failure analysis: when the environment is fixed, bad outcomes are easier to attribute to parsing, weighting, smoothing, or routing errors.
- Experimental hygiene: scene difficulty should come from the scenario profile, not from hidden model-side control over exogenous variables.

## 5. Practical runtime rule after this change

- Scene profile resolves first.
- SUMO environment is instantiated from the scene profile.
- The model predicts only planning weights.
- The routing algorithm selects a path from those weights.
- SUMO evaluates that chosen path under the fixed exogenous environment.
- The planner vehicle keeps the route chosen by the planner; reroute-related fields belong to the scene profile and are not exposed to model outputs.

This is the intended control boundary for all path-planning experiments in this project.
