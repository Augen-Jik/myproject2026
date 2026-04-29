# Final Freeze Notice

- Freeze time: `2026-04-24 14:04:28 UTC`
- Freeze directory: `/root/autodl-tmp/results/final_frozen_20260424_completion_aware`
- Latest run path: `/root/autodl-tmp/results/runs/20260424T135739Z__experiment_mode__compare_multi_method_live_scene_profile__method_comparison_v2`
- Latest run metrics source: `/root/autodl-tmp/results/runs/20260424T135739Z__experiment_mode__compare_multi_method_live_scene_profile__method_comparison_v2/metrics.csv`
- Frozen latest run metrics copy: `/root/autodl-tmp/results/final_frozen_20260424_completion_aware/latest_run_metrics.csv`
- Final metrics path: `/root/autodl-tmp/results/final_frozen_20260424_completion_aware/metrics.csv`
- Aggregate table path: `/root/autodl-tmp/results/final_frozen_20260424_completion_aware/method_comparison_final.csv`

## completion_status distribution

- `arrived`: 29
- `truncated_at_evaluation_end`: 7

## Truncated Runs by method

- `Dijkstra`: 1
- `Rule-A*`: 0
- `DQN`: 3
- `GCN-Weight`: 1
- `Sparse-LoRA`: 1
- `Sparse-LoRA+GAT`: 1

## stop_count fix

- `stop_count` no longer reads the non-existent `tripinfo.stopCount` field.
- It is computed from speed transitions in the sampled vehicle trajectory.
- A stop is counted when `last_speed >= 0.1` m/s and current `speed < 0.1` m/s.

## unfinished trip fix

- Trips with `arrival = -1` or `vaporized = end` are no longer marked as arrived.
- Scene-level metrics include `vehicle_arrived`, `simulation_truncated`, and `completion_status`.
- Records whose `travel_time_s` is close to the evaluation horizon are not silently treated as real arrivals.

## writing rule

Future paper writing uses this directory as the only result source. Old runs and old `paper_ready` tables must not be used for final tables or reported runtime metrics.
