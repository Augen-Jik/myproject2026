# DQN Training Notes

## 论文叙事关键说明（必读）

**DQN-RouteSelector 的本质是候选路由元选择器，不是独立的端到端路由生成器。**

- Action space = 从 {Dijkstra, Rule-A*, GCN-Weight} 已生成的候选路由中做离散选择
- 路由本身仍由下游经典算法生成，DQN policy 只决定选哪条
- 这解释了为什么 constraint_rate=94.59% 高于 Rule-A*（69.78%）：
  DQN 从多个候选中选约束最优者，而 Rule-A* 只输出单条路由
- 论文中必须明确写明这一设计，否则读者会误以为 DQN 独立生成了路由

**论文建议表述：**
> "DQN-RouteSelector 采用强化学习策略从经典算法（Dijkstra、Rule-A*、GCN-Weight）预生成的候选路由集合中选择最优路由，而非端到端生成路由。其高约束遵守率（94.6%）来源于候选集合覆盖与策略选择，而非独立路由能力。"



- Training backend: `stable-baselines3 DQN` on CPU.
- Per-scene training episodes: `200` (one-step episodes, fixed action pool per scene).
- State space: 96-d scene vector produced by the independent text parser in `rl_baseline/run_dqn_fixed_baseline.py`.
- Action space: discrete candidate-path selection over 3-5 feasible routes collected from existing baselines plus scene-vector alternates.
- Reward: `-travel_time_s` from the live SUMO evaluator; no extra constraint penalty was added in this minimal version.
- Planning time in `dqn_fixed_results.csv` counts only policy inference plus candidate lookup during evaluation; training time is logged separately below.
- Old DQN / Rule-A* reference metrics source: `/root/autodl-tmp/results/runs/20260410T022825Z__experiment_mode__compare_multi_method_live_scene_profile__method_comparison_v2/metrics.csv`.
- Reused pre-trained scene policies in this pass: `True`.

## Per-Scene Training Log

| Scene | Episodes | Candidate Count | Train Wall Time (s) | Policy File | Selected Candidate |
|---|---:|---:|---:|---|---|
| 场景N normal_baseline 正常通行 | 200 | 3 | 0.000 | /root/autodl-tmp/results/final_tables/dqn_scene_policies/normal_baseline_dqn_policy.zip | Dijkstra |
| 场景A 黄河路单向拥堵 | 200 | 3 | 0.000 | /root/autodl-tmp/results/final_tables/dqn_scene_policies/simple_local_dqn_policy.zip | Rule-A* |
| 场景E 方向不对称拥堵 | 200 | 3 | 0.000 | /root/autodl-tmp/results/final_tables/dqn_scene_policies/directional_asymmetry_dqn_policy.zip | GCN-Weight |
| 场景D 中央核心节点封锁 | 200 | 3 | 0.000 | /root/autodl-tmp/results/final_tables/dqn_scene_policies/core_blockage_dqn_policy.zip | GCN-Weight |
| 场景G 传播范围外溢 | 200 | 3 | 0.000 | /root/autodl-tmp/results/final_tables/dqn_scene_policies/propagation_range_dqn_policy.zip | GCN-Weight |
| 场景C 多路段复合拥堵 | 200 | 3 | 0.000 | /root/autodl-tmp/results/final_tables/dqn_scene_policies/compound_disaster_dqn_policy.zip | Dijkstra |
