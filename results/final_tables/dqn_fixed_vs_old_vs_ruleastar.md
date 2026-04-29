# DQN Fixed vs Old DQN vs Rule-A*

## Old DQN Issue Summary

- `compare/run_compare.py`: `method_dqn()` first calls `method_rule_astar()` and then trains/follows the rule-derived weights instead of producing an independent RL output.
- `compare/run_live_scene_profile_main_table.py`: `_legacy_baseline_method(..., method_name='DQN')` calls `_legacy_rule_weights(case)` and returns those same weights as `final_weights` for `constraint_rate` evaluation.
- Because live `constraint_rate` is computed from `final_weights` rather than from the finally chosen path, old DQN and Rule-A* share the same per-scene constraint behavior by construction.
- Reference scene-level old metrics loaded from: `/root/autodl-tmp/results/runs/20260410T022825Z__experiment_mode__compare_multi_method_live_scene_profile__method_comparison_v2/metrics.csv`

## Scene-Level Comparison

| Scene | New DQN Travel (s) | New DQN Planning (s) | New DQN Constraint (%) | Selected Candidate | Old DQN Travel (s) | Old DQN Constraint (%) | Rule-A* Travel (s) | Rule-A* Constraint (%) |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| 场景N normal_baseline 正常通行 | 235.8 | 0.000980 | 100.00 | Dijkstra | 1280.0 | 100.0 | 406.7 | 100.0 |
| 场景A 黄河路单向拥堵 | 706.7 | 0.000693 | 100.00 | Rule-A* | 906.6 | 100.0 | 706.7 | 100.0 |
| 场景E 方向不对称拥堵 | 1336.0 | 0.000567 | 85.71 | GCN-Weight | 1877.7 | 85.71 | 1778.0 | 85.71 |
| 场景D 中央核心节点封锁 | 1927.1 | 0.000319 | 100.00 | GCN-Weight | 2160.1 | 37.5 | 1273.7 | 37.5 |
| 场景G 传播范围外溢 | 2001.6 | 0.000327 | 100.00 | GCN-Weight | 1852.1 | 50.0 | 2113.6 | 50.0 |
| 场景C 多路段复合拥堵 | 2400.1 | 0.000348 | 81.82 | Dijkstra | 2400.1 | 45.45 | 2400.1 | 45.45 |

## Acceptance Check

- New DQN reuses Rule-A* weights as its own output: `no`
- Six standard scenes evaluated: `True`
- Scene-level comparison against old DQN and Rule-A* generated: `yes`

## WARNING Check

- No full six-scene `constraint_rate` identity with Rule-A* was detected.
