# PPO Standalone Note

## PPO 原始数字

来源文件：`results/ppo_baseline/ppo_quick_summary.csv`

- `method`: `PPO (quick baseline, untuned)`
- `Travel Time (s)`: `11.25`
- `Planning Time (s)`: `0.001429`
- `success rate`: `1.0`
- `average reward`: `8.75`
- `failure rate`: `0.0`
- `timeout rate`: `0.0`
- `training_steps`: `20000`
- `eval_episodes`: `30`
- `train_wall_time_s`: `23.486`
- `device`: `cpu`
- `table_ready`: `yes`

## 评估协议说明

- 当前 PPO 数字来自 `ppo_quick_summary.csv` 对应的简化快速基线环境。
- 该环境不是本次主表使用的 `scene-profile-fixed` live SUMO 协议。
- 因此，PPO 行不具备与主表 6 场景 live run 完全同口径的可比性。
- 尤其是主表要求的 `simulation_seed / warmup_seconds / evaluation_start_time / evaluation_end_time` 四个协议凭据，PPO 快速基线结果中并未提供同口径场景级记录。

## 论文使用建议

- 建议将 PPO 结果作为脚注、独立说明框，或 appendix 中的 coverage-only baseline 说明。
- 不建议将该 PPO 行直接并入最终主表 `method_comparison_final.csv`。
- 若论文正文需要提到 PPO，应明确标注其评估环境为“simplified quick baseline”而非 `scene-profile-fixed` live SUMO protocol。
