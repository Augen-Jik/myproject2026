# Completion-Aware Stats

Source: frozen `metrics.csv`; truncated/incomplete runs are retained in all-run means and excluded only from the arrived-only travel-time column.

| Method          |   Total Runs |   Arrived Runs |   Truncated Runs |   Completion Rate (%) |   Mean Travel Time All Runs (s) |   Mean Travel Time Arrived Only (s) | Truncated Scenarios   |
|:----------------|-------------:|---------------:|-----------------:|----------------------:|--------------------------------:|------------------------------------:|:----------------------|
| Dijkstra        |            6 |              5 |                1 |                 83.33 |                         1276.12 |                             1099.32 | D                     |
| Rule-A*         |            6 |              6 |                0 |                100    |                         1257.12 |                             1257.12 |                       |
| DQN             |            6 |              3 |                3 |                 50    |                         1796.57 |                             1353.03 | D|G|C                 |
| GCN-Weight      |            6 |              5 |                1 |                 83.33 |                         1471.9  |                             1334.26 | G                     |
| Sparse-LoRA     |            6 |              5 |                1 |                 83.33 |                         1257.33 |                             1028.78 | C                     |
| Sparse-LoRA+GAT |            6 |              5 |                1 |                 83.33 |                         1257.33 |                             1028.78 | C                     |

## Overall Completion Status

| completion_status           |   count |
|:----------------------------|--------:|
| arrived                     |      29 |
| truncated_at_evaluation_end |       7 |
