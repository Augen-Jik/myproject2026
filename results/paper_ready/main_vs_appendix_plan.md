# Main Vs Appendix Plan

## Main Body

The main body should keep only the material needed to support the central claim efficiently.

- Main method description: `Sparse-LoRA` as the core sparse prompt-to-anchor-to-route pipeline.
- Main comparison table: `results/final_tables/method_comparison_final.csv` rendered by `results/paper_ready/table1_method_comparison_final.tex`.
- Main ablation table: `results/final_tables/ablation_study.csv` summarized through `results/paper_ready/table2_ablation.tex`.
- Key ablation story: dense baseline -> No-SpatioTemporal -> Sparse-LoRA -> Sparse-LoRA+GAT.
- Simulation protocol summary: `results/paper_ready/simulation_setup_section.md` plus `scene_parameter_table.csv` as the environment-setting reference.

## Appendix

The appendix should carry material that is useful, citation-ready, or reproducibility-oriented, but not necessary for the core narrative.

- PPO quick baseline: `results/final_tables/method_comparison_appendix.csv` and `results/paper_ready/tableA1_ppo_appendix.tex`.
- GAT extension explanation: the `Extended` row in Table 1 plus `results/final_tables/table_notes.md` for its narrative placement.
- MAPPO scope note: `results/final_tables/mappo_scope_note.md`.
- Negative-results chain: `results/paper_ready/final_negative_results_summary.md` and `results/sparse_upstream_decision_page.md`.
- Scene-level prompt-baseline and ablation details: `results/final_tables/llm_prompt_baselines_scene_results.csv`, `results/final_tables/no_spatiotemporal_notes.md`, and `results/paper_ready/appendix_full_table.csv` when detailed evidence is needed.
