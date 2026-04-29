本次 clean run 主证据表明，Sparse-LoRA+GAT 在 6 个标准场景上的平均 `Travel Time (s)` 最低，为 503.47；Sparse-LoRA-v2 为 511.15；Rule-A* 为 513.55，说明规则基线本身已经较强。因此，本工作的结论不是“学习方法全面碾压传统方法”，而是“在强规则基线存在的前提下，稀疏输出结合图补全能够取得条件性收益”。

从时延拆分看，Sparse-LoRA-v2 与 Sparse-LoRA+GAT 的平均 `model_infer_time_s` 分别约为 1.47 和 1.48，显著低于 Qwen-LoRA 的 15.86 和 R1-LoRA 的 31.49；对应 `route_solve_time_s` 仅在 `10^-4` 秒量级。因此，协议定义下的 `Planning Time (s)` 主要由模型推理构成，而不是由图搜索本身主导。

从约束一致性看，Sparse-LoRA-v2 与 Sparse-LoRA+GAT 的平均约束符合率均为 68.23%，高于 Rule-A* 的 61.57。因此，稀疏方案在保持较低 `Planning Time (s)` 的同时，仍能维持较高的路况约束一致性。需要强调的是，GAT 的收益具有场景依赖性，在部分场景中能够改善 `Travel Time (s)`，在部分场景中则主要起到补全作用而非显著降时。
