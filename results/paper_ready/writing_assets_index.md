# Writing Assets Index

## Introduction

- `results/paper_ready/final_experiment_summary.md`: frozen headline takeaways for motivating the paper's contribution.
- `results/paper_ready/final_negative_results_summary.md`: useful for framing what was tried and deliberately excluded from the final claim.
- `results/final_tables/method_comparison_final.csv`: final high-level comparison evidence.

## Method

- `code/inference.py`: main inference entry points, including the sparse prompt chain.
- `code/sparse_utils.py`: structured spatiotemporal prompt formatting logic.
- `results/final_tables/no_spatiotemporal_notes.md`: minimal-scope explanation of the spatiotemporal-input ablation.
- `results/paper_ready/simulation_setup_section.md`: environment-model boundary and protocol definitions.
- `results/paper_ready/scene_parameter_table.csv`: scene configuration reference table.

## Experiments

- `results/final_tables/method_comparison_final.csv`: frozen main table data.
- `results/paper_ready/table1_method_comparison_final.tex`: main table LaTeX.
- `results/paper_ready/table1_final_preview.md`: markdown preview of the frozen main table.
- `results/final_tables/ablation_study.csv`: frozen ablation scene-level table.
- `results/paper_ready/table2_ablation.tex`: ablation LaTeX.
- `results/paper_ready/table2_final_preview.md`: markdown preview of the frozen ablation table.
- `results/final_tables/table_notes.md`: canonical interpretation notes for group labels, appendix placement, and naming constraints.

## Appendix

- `results/final_tables/method_comparison_appendix.csv`: appendix-only PPO quick baseline row.
- `results/paper_ready/tableA1_ppo_appendix.tex`: PPO appendix table.
- `results/final_tables/ppo_standalone_note.md`: protocol caveat for PPO.
- `results/final_tables/mappo_scope_note.md`: why MAPPO is deferred.
- `results/final_tables/dcrnn_inspired_notes.md`: conservative naming note for the DCRNN-inspired baseline.
- `results/final_tables/llm_prompt_baselines_notes.md`: prompt-baseline rerun notes.
- `results/paper_ready/appendix_full_table.csv`: larger appendix data asset if full detail is needed.
