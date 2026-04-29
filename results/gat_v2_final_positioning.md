# GAT-v2 Final Positioning

本文件作为当前版本的正式定位结论，优先级高于旧的 exploratory / clean-run 中间记录。

## Official Decision

- 主推荐方法名：`Sparse-LoRA-v2`
- `GAT-v2` 不进入论文主结果表
- 论文主方法名不再使用任何 `GAT` 变体作为默认方法名
- 若附录中保留 GAT 结果，推荐名称：`Sparse-LoRA-v2 + Gated-GAT-v2`
- `Always-GAT-v2` 仅用于附录 / 消融

## Evidence Summary

- 当前没有一个场景能稳定体现 GAT 的条件性收益
- 不建议写成“全局稳定提升”
- `propagation_range`：没有明确价值
- `compound_disaster`：只有边际价值，不够稳

## Appendix Policy

- GAT 结果仅以 exploratory comparison 形式保留在附录
- 附录对比方法限定为：
  - `Sparse-LoRA-v2`
  - `Sparse-LoRA-v2 + Gated-GAT-v2`
  - `Sparse-LoRA-v2 + Always-GAT-v2`
- 其中 `Sparse-LoRA-v2 + Gated-GAT-v2` 是唯一建议保留的 GAT 附录名称
- `Sparse-LoRA-v2 + Always-GAT-v2` 只作为消融参考，不作主张

## Locked Outcome

- GAT 已从论文主链路降级为附录 / 探索性分析
- 论文主方法名锁定为 `Sparse-LoRA-v2`

## 关键数值确认（Live SUMO 6场景固定协议，seeds 101-505）

数据来源：`final_tables/method_comparison_final.csv`（冻结，2026-04-17 验证）

| Variant | Travel Time (s) | Constraint Rate (%) | Signal Ratio (%) |
|---------|----------------|---------------------|-----------------|
| Sparse-LoRA | **1190.25** | 70.28 | 9.18 |
| Sparse-LoRA+GAT | **1190.25** | 70.28 | **100.0** |

**结论：GAT 对行程时间零贡献（完全相同：1190.25s），对 signal_ratio 有显著影响（9.18 → 100.0）。**

附录中必须明确标注：
- "图增强扩展变体的改进仅体现在信号比（signal_ratio）指标，不改善行程时间或约束遵守率"
- 不得将 signal_ratio 提升表述为"路径规划质量提升"，应说明其含义（路网边被预测权重覆盖的比例）
- 行程时间无改善意味着 GAT 的图平滑在 SUMO 实际行驶中未产生可观测的路由效益
