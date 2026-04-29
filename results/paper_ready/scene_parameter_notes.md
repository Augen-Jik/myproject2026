# Scene Parameter Notes

## 用途

`scene_parameter_table.csv` 汇总了当前 SUMO 测试链路中 8 个场景 profile 的关键环境参数，来源于：

- `code/sim_scene_profiles.py`
- `code/sim_eval_protocol.py`
- `code/apply_scene_profile_to_sumo.py`

这份表可以直接作为论文方法设置或答辩说明中的“场景参数总表”。

## 核心结论

不同测试场景下，系统确实会动态调整以下环境变量：

- 拥堵速度映射与边速度限制
- 红绿灯固定 program / phase
- 事故边、封闭边、车道缩减
- 背景车流需求强度
- reroute 周期与阈值
- 随机种子、warm-up、评估窗口

但这些调整都由场景配置固定控制，不受模型理解或模型输出直接影响。

## 场景层固定控制什么

场景层负责固定外生环境变量，包括：

- `speed_profile` 与 `congestion_level_to_speed_factor`
- `incident_edges`
- `blocked_edges`
- `lane_reduction_edges`
- `tls_profile_name` 与 `tls_fixed_program_id`
- `demand_profile_name` 与 `depart_rate`
- `reroute_enabled`、`reroute_period`、`reroute_threshold_factor`、`reroute_threshold_constant`
- `simulation_seed`
- `warmup_seconds`
- `evaluation_start_time`
- `evaluation_end_time`
- `vehicle_depart_window`

在运行时，这些变量会被真正注入 SUMO：

- 场景进入 evaluation window 后，会对事故边施加固定降速，对封闭边施加固定封闭/极高通行代价，对缩减边施加固定车道压缩和 travel-time penalty。
- 红绿灯使用预先绑定的固定 TLS profile，不允许模型临时改灯。
- reroute 采用固定的 `periodic` 触发模式，周期和阈值来自 scene profile，而不是来自模型。
- 背景车流由 demand profile 和 `depart_rate` 固定生成。
- 每个场景都使用固定 seed policy 和固定 simulation seed。

## 模型层只负责什么

模型层只负责：

- 边权 / 代价估计
- 路径选择

模型不允许直接修改：

- TLS program 或 phase
- 事故 / 封闭 / 车道缩减状态
- 车流注入强度
- reroute 周期与阈值
- 随机种子
- warm-up 与评估窗口

## 为什么必须这样解耦

这样做的原因是保证：

- 公平性：不同方法在同一组固定环境条件下比较，避免把环境波动误判成模型改进。
- 可复现性：seed、需求、信号灯、事故状态和评估窗口都能被重复恢复。
- 方法边界清晰：论文里可以明确说明“模型解决的是边权估计与路径规划问题”，而不是把环境控制也混进模型能力。
- 结果解释稳定：如果某个方法 travel time 更优，可以更有把握地归因于路径决策，而不是红绿灯、车流或随机种子的变化。

## 表格解读建议

- `speed_profile` 列记录基础速度参数和 canonical congestion-to-speed 因子。
- `incident_setting`、`blocked_edges`、`lane_reduction` 列记录场景内被动态调整的事故/封闭/容量约束。
- `tls_profile_name` 记录该场景绑定的固定信号灯控制方案。
- `demand_profile` 记录背景流量模板及 `depart_rate`。
- `reroute_period` 与 `reroute_threshold_factor` 记录场景自带 reroute 规则。
- `seed_policy`、`warmup_seconds`、`evaluation_window` 记录复现实验口径。

## 答辩可直接使用的结论

当前已经形成“模型层 / 环境层”清晰分工。

这份表可直接用于论文方法设置或答辩说明。
