# LLM Prompt Baselines Notes

## Entry Points

- `CoT-Qwen`: `compare/run_compare.py` -> `COT_PROMPT_TMPL` + `llm_infer()` + `parse_weights()` + `astar_route()`.
- `R1-Raw`: `compare/run_compare.py` -> `llm_infer_r1()` + `parse_weights()` + `astar_route()`.
- Live evaluation protocol reused from `compare/run_live_scene_profile_main_table.py` via `standard_scene_cases()` and `app_fixed.sumo_engine.run_simulation()`.

## Compatibility

- Both methods can be attached to the six standard scenes without changing Sparse-LoRA / GAT / DQN / PPO code.
- `planning_time` in this batch counts only prompt inference + downstream A* planning; model loading is excluded.
- `signal_ratio` is applicable through `compute_edge_metrics()`.
- `anchor_coverage` is left blank because these prompt baselines do not emit anchor-level structures in the current evaluation chain.

## Scene-Level Results

| Method | Scene | Travel Time (s) | Planning Time (s) | Constraint Rate (%) | Signal Ratio (%) | Anchor Coverage |
|---|---|---:|---:|---:|---:|---|
| CoT-Qwen | normal_baseline | 406.700 | 4.413914 | 100.00 | 3.100 |  |
| CoT-Qwen | simple_local | 706.700 | 16.247672 | 50.00 | 3.100 |  |
| CoT-Qwen | directional_asymmetry | 898.000 | 3.030694 | 14.29 | 3.100 |  |
| CoT-Qwen | core_blockage | 2160.100 | 5.647573 | 25.00 | 3.100 |  |
| CoT-Qwen | propagation_range | 2113.600 | 8.738162 | 33.33 | 3.100 |  |
| CoT-Qwen | compound_disaster | 2400.100 | 8.674889 | 18.18 | 3.100 |  |
| R1-Raw | normal_baseline | 235.800 | 18.599233 | 100.00 | 0.000 |  |
| R1-Raw | simple_local | 356.000 | 16.523592 | 0.00 | 0.000 |  |
| R1-Raw | directional_asymmetry | 898.000 | 24.438466 | 14.29 | 0.000 |  |
| R1-Raw | core_blockage | 2160.100 | 49.642500 | 12.50 | 39.600 |  |
| R1-Raw | propagation_range | 1964.200 | 29.360208 | 83.33 | 46.900 |  |
| R1-Raw | compound_disaster | 2400.100 | 14.483521 | 0.00 | 19.800 |  |

## Mean Rows Added To Main Table

| Method | Category | Travel Time (s) | Planning Time (s) | Constraint Rate (%) | Signal Ratio (%) |
|---|---|---:|---:|---:|---:|
| CoT-Qwen | LLM-Prompt | 1447.533333 | 7.792151 | 40.133333 | 3.100000 |
| R1-Raw | LLM-Prompt | 1335.700000 | 25.507920 | 35.020000 | 17.716667 |
