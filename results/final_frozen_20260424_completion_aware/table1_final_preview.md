# Method Comparison Final Preview

> Completion-aware live SUMO aggregation; Truncated Runs are retained as incomplete runs and are not counted as arrivals.

## Classical

| Method | Travel Time (s) | Time Loss (s) | Waiting Time (s) | Stop Count | Completion Rate (%) | Truncated Runs |
|---|---:|---:|---:|---:|---:|---:|
| Dijkstra | 1276.12 | 931.59 | 914.68 | 3.83 | 83.3 | 1 |
| Rule-A* | 1257.12 | 898.70 | 880.12 | 4.17 | 100.0 | 0 |

## GNN

| Method | Travel Time (s) | Time Loss (s) | Waiting Time (s) | Stop Count | Completion Rate (%) | Truncated Runs |
|---|---:|---:|---:|---:|---:|---:|
| GCN-Weight | 1471.90 | 1159.07 | 1135.15 | 5.00 | 83.3 | 1 |

## RL

| Method | Travel Time (s) | Time Loss (s) | Waiting Time (s) | Stop Count | Completion Rate (%) | Truncated Runs |
|---|---:|---:|---:|---:|---:|---:|
| DQN | 1796.57 | 1215.91 | 1177.88 | 7.50 | 50.0 | 3 |

## Ours

| Method | Travel Time (s) | Time Loss (s) | Waiting Time (s) | Stop Count | Completion Rate (%) | Truncated Runs |
|---|---:|---:|---:|---:|---:|---:|
| Sparse-LoRA | 1257.33 | 989.79 | 969.88 | 4.33 | 83.3 | 1 |

## Extended

| Method | Travel Time (s) | Time Loss (s) | Waiting Time (s) | Stop Count | Completion Rate (%) | Truncated Runs |
|---|---:|---:|---:|---:|---:|---:|
| Sparse-LoRA+GAT | 1257.33 | 989.79 | 969.88 | 4.33 | 83.3 | 1 |
