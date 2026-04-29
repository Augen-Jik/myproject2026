# UI Metrics Refactor Notes

## 目标

把 Streamlit 主界面的结果展示区重构成答辩可快速讲解的结构，重点解决：

- 模型理解结果、规划结果、仿真结果混在一起，不利于现场解释
- `Planning Time (s)` 与 `Travel Time (s)` 视觉上没有明确分层
- 兼容旧结果读取时，legacy 字段名有误导风险

## 本轮改动

### 1. 结果区重组为 A / B / C / D 四块

在 `/root/autodl-tmp/code/app_fixed.py` 中将结果展示重构为：

- A. 模型理解结果
  - `Sparse Anchors`
  - `scene_profile`
  - 关键异常边摘要
  - exploratory 选项状态（如 GAT）

- B. 规划结果
  - 路径概览
  - 边权 / 代价关键变化
  - `route_solve_time_s`
  - route solve 细节（详细模式下展开）

- C. 时间指标
  - `Planning Time (s)`
  - `model_infer_time_s`
  - `route_solve_time_s`
  - `Travel Time (s)`

- D. 环境配置摘要
  - `TLS profile`
  - `reroute policy`
  - `seed`
  - `warmup / evaluation window`

### 2. 新增 canonical display payload

新增了结果展示适配层：

- `_build_result_display_payload(...)`
- `_safe_float(...)`
- `_render_metric_panel(...)`

作用：

- 所有时间字段先通过 `normalize_eval_time_fields(...)` 做协议归一化
- UI 只消费 canonical 字段
- 不把 `planning_ms / plan_ms / sumo_time / time_s / true_travel_time` 直接展示给用户

### 3. 明确分开 Planning Time 与 Travel Time

时间区改为两张独立卡片：

- `Planning Time (s)`：模型推理 + 路径求解
- `Travel Time (s)`：SUMO actual travel time

同时保留子项：

- `model_infer_time_s`
- `route_solve_time_s`

这样答辩时可以直接解释：

- 模型多久理解输入并给出边代价
- 规划器多久把代价转成路径
- 车辆在 SUMO 环境里实际跑了多久

### 4. 增加两种显示级别

侧边栏新增：

- `答辩简洁模式`
- `调试详细模式`

区别：

- 答辩简洁模式：保留 A/B/C/D 主结论块，隐藏冗长表格与附加图表
- 调试详细模式：展开 route 明细、异常映射、协议快照、速度曲线、甘特图

## 结果

当前主结果展示区已经满足答辩快速说明需求，尤其是：

- 模型输出和规划结果不再混讲
- `Planning Time (s)` 与 `Travel Time (s)` 不再视觉混写
- 环境配置被单独归位，能明确说明哪些是 exogenous scene profile，哪些是模型控制变量

## 使用建议

- 答辩默认使用：`答辩简洁模式`
- 若老师追问解析细节、协议映射或路径构成，再切换到：`调试详细模式`

## 备注

- 当前结果页的展示级别切换是 sidebar 控件；若切换后需要刷新当前结果块，重新运行一次规划即可。
