# Runtime Metrics Audit

## stop_count fix

- `stop_count` no longer reads the non-existent `tripinfo.stopCount` field.
- `stop_count` is computed from speed transitions in the sampled vehicle trajectory.
- A stop is counted when `last_speed >= 0.1` m/s and current `speed < 0.1` m/s.

## unfinished trip fix

- Trips with `arrival = -1` or `vaporized = end` are no longer marked as arrived.
- Scene-level metrics include `vehicle_arrived`, `simulation_truncated`, and `completion_status`.
- Records whose `travel_time_s` is close to the evaluation horizon are no longer silently treated as real arrivals; they remain visible as truncated/incomplete runs.

## current final run

- Run directory: `/root/autodl-tmp/results/runs/20260424T135739Z__experiment_mode__compare_multi_method_live_scene_profile__method_comparison_v2`
- Run metrics: `/root/autodl-tmp/results/runs/20260424T135739Z__experiment_mode__compare_multi_method_live_scene_profile__method_comparison_v2/metrics.csv`
- Final metrics copy: `/root/autodl-tmp/results/final_frozen_20260424_completion_aware/metrics.csv`
- Final aggregate table: `/root/autodl-tmp/results/final_frozen_20260424_completion_aware/method_comparison_final.csv`
- Completion-aware stats table: `/root/autodl-tmp/results/final_frozen_20260424_completion_aware/completion_aware_stats.csv`

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

## consistency checks

- Total scene-level metrics: 36
- `positive_wait_zero_stop`: 0
- `simulation_truncated=true`: 7

## truncated audit

- Path: `/root/autodl-tmp/results/final_frozen_20260424_completion_aware/truncated_runs_audit.csv`
- Rows: 7
