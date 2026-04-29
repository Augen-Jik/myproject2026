# SIM Evaluation Protocol Notes

## 1. Unified protocol file

The unified evaluation protocol is defined in:

- `/root/autodl-tmp/code/sim_eval_protocol.py`

It standardizes these fields:

- `planning_time_definition`
- `travel_time_definition`
- `warmup_seconds`
- `evaluation_start_time`
- `evaluation_end_time`
- `reroute_trigger_mode`
- `reroute_period`
- `simulation_seed_policy`
- `num_eval_seeds`
- `vehicle_depart_window`

## 2. Required interpretation of the metrics

- `Planning Time (s)` only counts model inference time plus route-solving time.
- `Travel Time (s)` only counts the vehicle trip duration inside SUMO.
- These two are now intentionally separated and must not be written into a single mixed field.

In the current app pipeline:

- `model_infer_time_s` = model inference only
- `route_solve_time_s` = graph search / path solving only
- `planning_time_s` = `model_infer_time_s + route_solve_time_s`
- `travel_time_s` = live SUMO trip duration for the evaluated vehicle

## 3. Current project behavior

### Reroute trigger

Current reroute is triggered by a fixed periodic mode.

Concretely:

- The trigger mode is fixed as `periodic` in `/root/autodl-tmp/code/sim_eval_protocol.py`.
- The live runtime uses `/root/autodl-tmp/code/apply_scene_profile_to_sumo.py`.
- `SUMOSceneProfileApplier.on_simulation_step()` checks simulation time on every step.
- When the simulation is inside the evaluation window, `_maybe_run_periodic_reroute(...)` fires according to the fixed `reroute_period`.

So the current project does not use model-driven reroute triggering.

### Seed policy

Current seed fixing is done as follows:

- Each scene profile provides a fixed `simulation_seed`.
- SUMO is started with `--seed <scene_profile.simulation_seed>`.
- The evaluation protocol fixes the seed policy as `scene_profile_fixed_seed_single_run`.
- The current protocol fixes `num_eval_seeds = 1`.

This means a batch is reproducible because every scene uses a fixed, declared seed policy rather than an implicit or model-dependent seed.

### Warm-up and evaluation window

Current warm-up and evaluation window are defined by the scene profile and recorded by the evaluation protocol:

- `warmup_seconds`
- `evaluation_start_time`
- `evaluation_end_time`
- `vehicle_depart_window`

Operationally:

- SUMO is started first.
- The simulation is stepped until `max(warmup_seconds, evaluation_start_time)`.
- The evaluated vehicle departs deterministically at that moment.
- Scene effects and scene-controlled reroute are only active inside `[evaluation_start_time, evaluation_end_time]`.

So warm-up and evaluation are fixed exogenous protocol settings, not something inferred from the model output.

## 4. Legacy field mapping

The new protocol gives explicit mapping for older timing/travel fields:

- `planning_ms` -> `planning_time_ms`
- `plan_ms` -> `planning_time_ms`
- `time_s` -> `planning_time_s`
- `infer_time` -> `model_infer_time_s`
- `sumo_time` -> `travel_time_s`
- `travel_time` -> `travel_time_s`
- `true_travel_time` -> `analytical_travel_time_s_legacy`

Important note:

- `true_travel_time` in `/root/autodl-tmp/code/evaluate.py` is an analytical proxy computed from a synthetic weight formula.
- It is not the same thing as live SUMO `travel_time_s`.
- Therefore it must not be merged into the canonical SUMO travel-time field.

## 5. Why the protocol must stay outside model understanding

These protocol elements must not be directly influenced by the model:

- reroute trigger mode
- reroute period
- random seed policy
- warm-up duration
- evaluation window
- vehicle depart window

The reason is simple:

- If the model can change these, it can change the test conditions instead of improving planning quality.
- That would turn environment volatility into fake model gains.
- Fixed protocol settings ensure that observed improvements come from better inference or route choice, not from easier simulation timing, luckier seeds, or delayed departures.

In short: protocol stability is necessary to claim that a planning improvement is real rather than an artifact of environment fluctuation.
