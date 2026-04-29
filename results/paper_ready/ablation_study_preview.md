# Ablation Study Final Preview

Source: frozen `ablation_study.csv`; Sparse-LoRA-v2 is the final main method, GAT rows are exploratory controls.

| Variant           | Variant (CN)                     |   Travel Time (s) |   Planning Time (s) |   Constraint Rate (%) |   Signal Ratio (%) |   Parsed Edge Ratio (%) |
|:------------------|:---------------------------------|------------------:|--------------------:|----------------------:|-------------------:|------------------------:|
| Qwen-LoRA         | Qwen-LoRA（密集 LoRA 基线）      |           591.95  |            15.8584  |               47.4167 |           34.7167  |                50       |
| LoRA+GAT          | Qwen-LoRA+GAT（探索性参考）      |           574.417 |            15.7851  |               47.4167 |           64.25    |                50       |
| No-SpatioTemporal | 无时空增强输入                   |           542.333 |             1.17328 |               50.9833 |            9.55    |                10.25    |
| Sparse-LoRA       | Sparse-LoRA-v2（最终主方法）     |           511.15  |             1.47508 |               68.2333 |            9.38333 |                 9.38333 |
| Sparse-LoRA+GAT   | Sparse-LoRA-v2+GAT（附录探索性） |           503.467 |             1.48174 |               68.2333 |           61.45    |                 9.38333 |
