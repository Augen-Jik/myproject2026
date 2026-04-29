# 实验表格说明文件

## 冻结状态

- 本页对应 2026-04-11 写作冻结版本。
- 当前阶段不再新增 baseline、消融或训练任务。
- 允许的后续动作仅限于表格引用、章节叙事、附录组织与文字润色；不再改动任何实验数值。

## 最终表格清单

### 主表
- `results/final_tables/method_comparison_final.csv`
- `results/paper_ready/table1_method_comparison_final.tex`
- `results/paper_ready/table1_final_preview.md`

### 消融表
- `results/final_tables/ablation_study.csv`
- `results/paper_ready/table2_ablation.tex`
- `results/paper_ready/table2_final_preview.md`

### 附录表
- `results/final_tables/method_comparison_appendix.csv`
- `results/paper_ready/tableA1_ppo_appendix.tex`

## 主表最终分组与顺序

- `Classical`: `Dijkstra`, `Rule-A*`
- `GNN`: `GCN-Weight (Kipf & Welling)`, `DCRNN-inspired`
- `RL`: `DQN-RouteSelector`
- `LLM-Prompt`: `CoT-Qwen`, `R1-Raw`
- `Ours`: `Sparse-LoRA`
- `Extended`: `Sparse-LoRA+GAT`

说明：
- `PPO` 不再保留在主表，而是冻结到附录表 `tableA1_ppo_appendix.tex`。
- `Sparse-LoRA+GAT` 仍然保留结果，但在论文叙事中归为 `Extended`，避免把图平滑扩展误写成核心主方法本体。

## 消融表最终顺序

- `Qwen-LoRA`
- `LoRA+GAT`
- `No-SpatioTemporal`
- `Sparse-LoRA`
- `Sparse-LoRA+GAT`

说明：
- `LoRA+GAT` 保留为旧 dense+GAT 对照，不作为主方法组成部分。
- `No-SpatioTemporal` 是本轮新增且已冻结的真实时空输入消融：唯一变化是去掉结构化时空字段注入。

## 写作引用口径

- `Travel Time (s)` 是 SUMO 中的实际 trip duration。
- `Planning Time (s)` 统一定义为 `model_infer_time_s + route_solve_time_s`。
- `PPO` 仅可引用为 appendix-only quick baseline，不能表述为与主表同协议的 live rerun。
- `DQN-RouteSelector` 必须表述为 repaired candidate-route selection policy，不复用 rule-based weights 作为自身输出。
- `DCRNN-inspired` 必须表述为 inspired baseline，而不是完整 DCRNN reproduction。
- `MAPPO` 已明确暂缓，见 `results/final_tables/mappo_scope_note.md`。
