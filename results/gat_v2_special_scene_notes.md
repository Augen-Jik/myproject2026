# GAT-v2 Special Scene Notes

- propagation_range 是否有明确价值: 没有明确价值。 sparse/main dense MAE=`0.0`/`0.1902`， changed-edge MAE=`0.0`/`0.0`。
- compound_disaster 是否有明确价值: 有边际价值，但不够稳。 sparse/main dense MAE=`0.0043`/`0.1912`， changed-edge MAE=`0.0636`/`0.0693`。
- 哪个场景最能体现 GAT 的条件性收益: 当前没有一个场景能稳定体现 GAT 的条件性收益。
- 是否建议论文主文强调“特殊场景更有价值”而不是“全局稳定提升”: 不建议把当前版本写成“全局稳定提升”。若保留 GAT，只能作为探索性附录，不宜在主文强推。
