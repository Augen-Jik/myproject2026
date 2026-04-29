# Final Experiment Summary

## Main Table

The frozen main table is organized into six groups: `Classical`, `GNN`, `RL`, `LLM-Prompt`, `Ours`, and `Extended`.

- `Sparse-LoRA` is the core main-method row kept under `Ours`.
- `Sparse-LoRA+GAT` is retained under `Extended` because it adds graph smoothing on top of the sparse mainline rather than defining the core method claim by itself.
- Among the protocol-matched main-table rows, `Sparse-LoRA` and `Sparse-LoRA+GAT` remain the strongest travel-time results at `1190.25 s`, clearly below the classical, GNN, RL, and prompt-only baselines.
- `DQN-RouteSelector` is the repaired RL baseline and remains in the table because it is now a valid independent candidate-route selector rather than a copy of `Rule-A*`.
- `CoT-Qwen` and `R1-Raw` stay in `LLM-Prompt` to provide citation-ready prompt baselines with a traceable live rerun chain.

## Ablation Table

The frozen ablation narrative is:

1. `Qwen-LoRA` is the dense LoRA starting point.
2. `LoRA+GAT` is an old dense+GAT control retained only for completeness.
3. `No-SpatioTemporal` removes structured spatiotemporal prompt fields while keeping the same Sparse-LoRA model, parser, planner, and metrics.
4. `Sparse-LoRA` shows the gain from the sparse structured formulation.
5. `Sparse-LoRA+GAT` shows the extra graph-smoothing extension after the sparse mainline is established.

Core reading:
- `No-SpatioTemporal` worsens travel time from `511.15 s` to `542.33 s` relative to `Sparse-LoRA`, supporting the claim that structured spatiotemporal input helps the sparse pipeline.
- `Sparse-LoRA+GAT` further improves travel time to `503.47 s`, but this is treated as an extension rather than the core method definition.

## Spatiotemporal Input Effect

The `No-SpatioTemporal` row isolates the prompt-input question with minimal scope change. Its only difference from `Sparse-LoRA` is whether structured fields such as road, direction, range, time, propagation, and conflict are injected into the prompt. The drop from `Sparse-LoRA` to `No-SpatioTemporal` supports the paper's claim that explicit spatiotemporal structuring contributes to better downstream routing quality.

## Why PPO Stays In The Appendix

The available PPO result comes from `results/ppo_baseline/ppo_quick_summary.csv`, which is a simplified quick-baseline environment rather than the six-scene `scene-profile-fixed` live SUMO protocol. It is therefore still useful as coverage material, but it is not protocol-matched enough to sit in the frozen main table. The correct placement is Appendix Table A1.

## Why MAPPO Is Deferred

`MAPPO` remains out of scope for the frozen write-up because the repository still lacks a clear multi-agent decomposition, per-agent observation spaces, joint or shared-policy action design, cooperative reward specification, and a compatible multi-agent SUMO wrapper. That makes MAPPO citation-relevant but not a lowest-risk submission item in the current codebase.

## Why GAT Is Treated As Extended

The GAT component is helpful and worth reporting, but it sits on top of the sparse mainline rather than defining the mainline itself. For writing clarity, the paper should anchor the core contribution on `Sparse-LoRA` and present `Sparse-LoRA+GAT` as an extended graph-enhanced variant with additional upside but a narrower claim.
