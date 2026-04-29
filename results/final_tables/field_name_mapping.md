# 字段名映射文件（Field Name Mapping）

## 当前协议主字段

| 当前协议字段 | 中文统一名称 | 历史字段 / 混用表述 | 说明 |
|--------------|--------------|---------------------|------|
| `Travel Time (s)` | `行程耗时（秒）` | `SUMO Time (s)`、`Avg SUMO Time (s)`、`travel_time`、`sumo_time` | 只表示 live SUMO 中被评估车辆的实际行程时间。 |
| `model_infer_time_s` | `模型推理耗时（秒）` | `Avg Inference Time (s)`、`infer_time`、`infer_ms` | 只表示模型推理本身，不含图搜索求解，不含 SUMO 行驶。 |
| `route_solve_time_s` | `路径求解耗时（秒）` | `Planning Time (ms)`、`planning_ms`、`plan_ms` | 只表示图搜索 / 路径求解部分。旧毫秒字段不能直接当成 `Planning Time (s)`。 |
| `Planning Time (s)` | `总规划耗时（秒）` | 历史材料中常误把 `Planning Time (ms)` 直接改名得到 | 当前协议定义为 `model_infer_time_s + route_solve_time_s`。 |
| `analytical_travel_time_s_legacy` | `历史解析行程代理（秒）` | `true_travel_time` | 历史解析代理值，仅可作为 legacy 保留；不是 live SUMO `Travel Time (s)`。 |
| `Constraint Rate (%)` | `约束满足率（%）` | `Avg Constraint Rate (%)`、`constraint_rate` | 路径满足交通约束的比例。 |
| `Signal Ratio (%)` | `信号边覆盖率（%）` | `Avg Signal Ratio (%)`、`signal_ratio` | 路径中被赋予显式信号/异常权重的边比例。 |
| `Parsed Edge Ratio (%)` | `解析边覆盖率（%）` | `parsed_edge_ratio`、`parsed_ratio` | LLM 成功解析并赋权的边比例。 |

---

## 主证据文件映射关系

### `main_results.csv` / `main_results.md` / `main_results.tex`

| 历史字段 | 当前字段 | 处理 |
|---------|---------|------|
| `Avg SUMO Time (s)` | `Travel Time (s)` | 直接重命名 |
| `Avg Inference Time (s)` | `model_infer_time_s` | 直接重命名 |
| `Avg Planning Time (ms)` | `route_solve_time_s` | 换算为秒（`/1000`） |
| — | `Planning Time (s)` | 新增派生列：`model_infer_time_s + route_solve_time_s` |

### `appendix_full_table.csv`

| 历史字段 | 当前字段 | 处理 |
|---------|---------|------|
| `SUMO Time (s)` | `Travel Time (s)` | 直接重命名 |
| `Inference Time (s)` | `model_infer_time_s` | 直接重命名 |
| `Planning Time (ms)` | `route_solve_time_s` | 换算为秒（`/1000`） |
| — | `Planning Time (s)` | 新增派生列：`model_infer_time_s + route_solve_time_s` |

### `method_comparison.csv` / `final_main_results.csv`

| 旧列名 | 当前列名 | 处理 |
|--------|---------|------|
| `Inference Time (s)` | `model_infer_time_s` | 直接重命名 |
| 旧 `Planning Time (s)` | `route_solve_time_s` | 纠正其真实语义：旧值实际是 route solve only |
| — | `Planning Time (s)` | 重新按协议计算 |

### `ablation_study.csv` / `final_ablation_results.csv`

| 旧列名 | 当前列名 | 处理 |
|--------|---------|------|
| `Inference Time (s)` | `model_infer_time_s` | 直接重命名 |
| 旧 `Planning Time (s)` | `route_solve_time_s` | 纠正其真实语义：旧值实际是 route solve only |
| — | `Planning Time (s)` | 重新按协议计算 |

---

## Legacy 原始结果文件说明

| Legacy 字段 | 只能映射到 | 说明 |
|-------------|------------|------|
| `planning_ms` / `plan_ms` | `route_solve_time_s` | 仅表示图搜索求解耗时；不能直接当成主表 `Planning Time (s)`。 |
| `time_s` | 视脚本而定，通常需人工审计 | 在历史脚本中曾混用为规划或推理时间，不能自动当主证据字段。 |
| `infer_time` | `model_infer_time_s` | 历史推理耗时别名。 |
| `travel_time` / `sumo_time` | `Travel Time (s)` | 仅在 live SUMO 语境中成立；需结合协议文件确认。 |
| `true_travel_time` | `analytical_travel_time_s_legacy` | 解析代理；不得再写成 SUMO 实测行程时间。 |

---

## 明确禁止的旧口径

- 不得把 `true_travel_time` 解释为 live SUMO `Travel Time (s)`。
- 不得把 `planning_ms` / `plan_ms` 直接作为论文主表 `Planning Time (s)`。
- 历史结果若必须保留，字段名或注释中必须显式标注 `legacy`。
