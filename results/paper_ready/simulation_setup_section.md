# Simulation Setup

## 场景配置层

本文将 SUMO 测试环境中的外生条件统一收敛到 `scene profile` 配置层，由 `/root/autodl-tmp/code/sim_scene_profiles.py` 中的 `SimSceneProfile` 负责声明。该配置层用于在实验开始前固定每个测试场景的环境状态，从而避免不同方法在不一致的交通条件下被比较。当前实现中，场景配置统一管理以下参数：速度配置与拥堵等级到速度因子的映射、事故边、封闭边、车道缩减边、固定信号灯方案、需求强度、reroute 规则、随机种子以及 warm-up 与 evaluation window。项目当前预定义了 `normal_baseline`、`simple_local`、`directional_asymmetry`、`core_blockage`、`propagation_range`、`compound_disaster`、`temporal_switch` 和 `anti_truncation_eval` 共 8 个场景 profile，用于覆盖局部拥堵、方向不对称干扰、核心路段封锁、传播型拥堵和复合灾害等不同测试条件。

在运行时，`/root/autodl-tmp/code/apply_scene_profile_to_sumo.py` 将上述场景配置注入 live SUMO 进程。具体而言，背景需求通过按场景生成的 runtime `.rou.xml` 文件写入，其中共享基础 flow library 但按 `depart_rate` 缩放发车周期；事故边通过降低车道最高速度并同步更新 edge travel time / effort 实现；封闭边通过将速度压至近零、禁止常见车辆类别通行并赋予极高 travel time / effort 实现；车道缩减通过关闭多余车道、降低保留车道速度并提高通行代价实现；信号灯状态则通过固定 `tls_profile_name` 和 `tls_fixed_program_id` 的方式在 TraCI 层统一施加。因而，不同测试场景下，拥堵速度、红绿灯和事故状态确实会动态调整，但这些调整由场景配置固定控制，不受模型理解直接影响。

此外，场景配置还固定控制场景自有的 reroute 机制。若某个 profile 启用 reroute，则其触发周期 `reroute_period` 以及阈值参数 `reroute_threshold_factor`、`reroute_threshold_constant` 均由场景层给定，并由 SUMO 注入器在仿真过程中统一执行。也就是说，reroute 规则是环境的一部分，而不是模型可学习或可改写的自由度。

## 模型层与环境层分工

本文在设计上严格区分模型层与环境层的责任边界。模型层只负责两项决策：其一，基于输入场景信息输出边权/路径代价估计；其二，在这些代价之上完成路径选择。除边权估计与路径选择之外，模型不直接修改任何 SUMO 环境变量，尤其不允许控制红绿灯 program 或 phase、事故/封闭/车道缩减状态、背景车流注入强度、随机种子、warm-up 时长、evaluation window，以及 reroute 的触发规则。

环境层则负责固定控制上述所有外生变量，并保证它们在一次实验中以确定方式进入 SUMO。这样的解耦有三方面意义。第一，公平性：不同方法必须在同一组环境条件下进行比较，否则 travel-time 改善可能仅来自测试条件变化而非规划能力提升。第二，可复现性：固定种子、固定需求、固定信号灯和固定评估窗口使得实验结果能够被重复恢复。第三，方法边界清晰：论文可以明确说明模型解决的是“边权估计与路径规划”问题，而不是把交通信号控制或环境操纵混入模型能力。基于这一边界，若某方法在 `Travel Time (s)` 上取得优势，则更容易将收益归因于更合理的代价建模与路径选择，而非外生环境被模型隐式改写。

## 评估协议

统一评估协议由 `/root/autodl-tmp/code/sim_eval_protocol.py` 定义，并在当前实验链路中强制采用。按照该协议，`Planning Time (s)` 的定义为模型推理时间与路径求解时间之和，即

`Planning Time (s) = model_infer_time_s + route_solve_time_s`，

其中 `model_infer_time_s` 仅统计模型推理本身，`route_solve_time_s` 仅统计图搜索/路径求解过程；SUMO 启动、warm-up、仿真推进以及车辆在 SUMO 中的实际行驶时间均不计入 `Planning Time (s)`。相对应地，`Travel Time (s)` 仅表示被评估车辆在 SUMO 中从出发到到达的 trip duration，是本文用于衡量闭环路径质量的核心效果指标。两者在定义上被明确分离，不再允许写入混合字段。

warm-up 与 evaluation window 同样由场景配置固定指定。实际执行时，SUMO 首先启动并进入 warm-up 阶段，随后仿真推进至 `max(warmup_seconds, evaluation_start_time)`，被评估车辆在该时刻确定性出发；场景效果以及场景控制的 reroute 仅在 `[evaluation_start_time, evaluation_end_time]` 窗口内生效。当前协议中，`reroute_trigger_mode` 被固定为 `periodic`，即由注入器在仿真步循环中按固定周期检查是否满足 reroute 条件，而非由模型输出触发。随机性控制方面，协议固定 `seed policy = scene_profile_fixed_seed_single_run`，即每个场景使用其 profile 中声明的固定 `simulation_seed` 启动 SUMO，并在当前协议下采用单次固定种子评估（`num_eval_seeds = 1`）。因此，warm-up、evaluation window、reroute 触发方式和随机种子都属于外生协议设置，而不是模型可利用的自由变量。

## 场景参数表说明

为便于论文方法设置部分集中呈现环境参数，`/root/autodl-tmp/results/paper_ready/scene_parameter_table.csv` 已汇总 8 个场景 profile 的关键设置，包括速度映射、TLS profile、事故与封闭边、车道缩减、需求强度、reroute 参数、seed policy、warm-up 时长和 evaluation window。该表可直接作为论文中的“场景参数总表”或方法设置附表引用。与之配套的 `/root/autodl-tmp/results/paper_ready/scene_parameter_notes.md` 则给出了字段解释及答辩表述建议。实践上，正文可在方法设置小节中先概述场景配置层与协议边界，再将具体参数细节统一指向 `scene_parameter_table.csv`，从而兼顾主文简洁性与实验设置的可追溯性。
