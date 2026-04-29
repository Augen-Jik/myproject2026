# SUMO Scene Injection Audit

## 1. Entry scripts found in the current project

The current SUMO runtime chain is centered on these files:

- `/root/autodl-tmp/code/app_fixed.py`
  - Main UI/runtime entry.
  - Starts SUMO, injects the planner vehicle, runs the simulation loop, and reads `tripinfo` to get travel time.
- `/root/autodl-tmp/code/apply_scene_profile_to_sumo.py`
  - New unified scene injection entry added in this task.
  - Owns speed injection, incident/blockage/lane-reduction injection, TLS profile application, and scene-controlled reroute behavior.
- `/root/autodl-tmp/code/sim_scene_profiles.py`
  - Declares exogenous scene profiles and route-demand generation.
- `/root/autodl-tmp/code/sumo_runner.py`
  - Generic SUMO helper wrapper. Not the main experiment entry.
- `/root/autodl-tmp/code/build_urban_net.py`
  - Generates the network and the historical base flow library used by runtime route generation.

## 2. What is now truly injected into SUMO

The following scene parameters are now applied to the live SUMO process through `/root/autodl-tmp/code/apply_scene_profile_to_sumo.py`:

- `simulation_seed`
  - Injected through SUMO CLI `--seed`.
- `depart_rate`
  - Injected by generating a per-scene runtime `.rou.xml` and scaling flow periods.
- `incident_edges`
  - Injected at runtime by lowering lane speeds and adapting edge travel time / effort inside the evaluation window.
- `blocked_edges`
  - Injected at runtime by forcing a near-zero lane speed, disallowing all common vehicle classes on affected lanes, and assigning a huge edge travel time / effort inside the evaluation window.
- `lane_reduction_edges`
  - Injected at runtime by closing surplus lanes beyond the configured remaining-lane count, reducing surviving-lane speed, and raising travel time / effort inside the evaluation window.
- `congestion_level_to_speed_factor`
  - Injected through a centralized mapping in `/root/autodl-tmp/code/apply_scene_profile_to_sumo.py`.
  - Supported canonical levels are `severe`, `moderate`, `mild`, `relief`, and `normal`, with aliases such as `light`, `heavy`, and `blocked`.
- `tls_profile_name`
  - Injected through a minimal TLS registry in `/root/autodl-tmp/code/apply_scene_profile_to_sumo.py`.
  - Current concrete profiles are `static_program_0`, `static_program_0_phase_0`, `static_program_0_phase_2`, `static_all_intersections`, and `all_red_lock`.
- `tls_fixed_program_id`
  - Injected through TraCI `trafficlight.setProgram(...)`.
- `warmup_seconds`, `evaluation_start_time`, `evaluation_end_time`
  - Injected by gating when scene effects become active and when they are restored.
- `reroute_enabled`
  - Injected through one unified scene-owned reroute entry.
- `reroute_period`
  - Injected through scene-owned periodic reroute checks.
- `reroute_threshold_factor`
  - Injected through the threshold test `current_remaining_tt > alt_tt * factor + constant`.
- `reroute_threshold_constant`
  - Injected through the same threshold test.

## 3. What is still only a configuration placeholder or partial implementation

- `demand_profile_name`
  - Currently used as scene metadata and audit labeling.
  - The actual SUMO demand change is driven by `depart_rate` over a shared base flow library, not by multiple fully distinct OD templates yet.
- `speed_profile.base_speed_factor`
  - Present in config, but the current injector applies scene effects only to affected edges rather than globally rescaling every edge in the network.
- `tls_profile_name`
  - Already has a real minimal implementation, but it is still a small registry.
  - If future scenes require per-intersection custom phase tables or corridor-specific offsets, those would need additional profile entries.
- `speed_profile` as a rich object
  - The current injector uses `minimum_edge_speed_mps`, blocked/lane-reduction behavior, and the centralized congestion mapping.
  - More granular per-edge custom speed curves are not implemented yet.

## 4. Reroute ownership

`reroute` is now controlled by the scene profile, not by the model.

Specifically:

- The model does not set `reroute_enabled`.
- The model does not set `reroute_period`.
- The model does not set `reroute_threshold_factor`.
- The model does not set `reroute_threshold_constant`.
- The model still only provides edge-cost estimates for planning and the initial path choice.
- Any later route revision caused by reroute is triggered by the scene-owned SUMO injector under fixed exogenous rules.

## 5. Local verification performed

I ran local smoke tests after wiring the injector:

- Speed/window test:
  - On a shortened `propagation_range` profile, lane `C2R0_N_0` changed from `16.67` m/s before the evaluation window to `0.35` m/s inside the window, then returned to `16.67` m/s after the window.
- TLS test:
  - A `static_program_0_phase_2` profile kept the inspected traffic light on phase `2` during the test window.
- Reroute test:
  - A vehicle with initial route `('R0C1_E', 'C2R0_N', 'C2R1_N', 'C2R2_N', 'C2R3_N')` was rerouted under the scene-owned policy to `('R0C1_E', 'R0C2_E', 'C3R0_N', 'C3R1_N', 'C3R2_N', 'R3C2_W', 'C2R3_N')`.

These checks show that speed restrictions, evaluation-window gating, TLS profile injection, and scene-controlled reroute are all now reaching the live SUMO runtime.
