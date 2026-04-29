# DCRNN-inspired Notes

- Core design: forward/backward random-walk diffusion aggregation over the fixed directed 96-edge graph.
- Diffusion depth: `K=2`.
- Seed weights: reuse the existing `method_gcn_weight()` output as the initial graph signal, then diffuse over directed edge-neighborhoods.
- This baseline is DCRNN-inspired and only reuses diffusion-style graph aggregation on a fixed graph; it is not a full DCRNN reproduction.

## Scene-Level Results

| Scene | Travel Time (s) | Planning Time (s) | Constraint Rate (%) | Signal Ratio (%) |
|---|---:|---:|---:|---:|
| normal_baseline | 235.800 | 0.001630 | 100.00 | 77.100 |
| simple_local | 706.700 | 0.001560 | 100.00 | 81.200 |
| directional_asymmetry | 1336.000 | 0.001365 | 71.43 | 85.400 |
| core_blockage | 1927.100 | 0.001427 | 37.50 | 83.300 |
| propagation_range | 2113.600 | 0.003107 | 16.67 | 77.100 |
| compound_disaster | 2400.100 | 0.003095 | 18.18 | 81.200 |

## Duplicate Check

- No full six-scene identity with `GCN-Weight (Kipf & Welling)` was detected.
