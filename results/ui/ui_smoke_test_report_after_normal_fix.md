# UI Smoke Test Report After normal_baseline Fix

- Date: 2026-04-09 UTC
- App: `/root/autodl-tmp/code/app_fixed.py`
- Method: `streamlit.testing.v1.AppTest` regression smoke test after the normal_baseline UI fix
- Covered scenes: normal_baseline, simple_local, directional_asymmetry, propagation_range, compound_disaster + cross-scene stale-cache check

## Scenario Summary

| scene | scene_profile refresh | env params | model result | time metrics | map switch | defense mode | result record | notes |
|---|---|---|---|---|---|---|---|---|
| normal_baseline | PASS | PASS | PASS | PASS | PASS | PASS | PASS | planning=1.11 s; travel=198.40 s; run=10.20s; detected_scene_type=normal_baseline; selection_mode=preset_locked |
| simple_local | PASS | PASS | PASS | PASS | PASS | PASS | PASS | planning=1.09 s; travel=318.70 s; run=9.30s; detected_scene_type=simple_local; selection_mode=preset_locked |
| directional_asymmetry | PASS | PASS | PASS | PASS | PASS | PASS | PASS | planning=1.50 s; travel=1334.70 s; run=42.47s; detected_scene_type=directional_asymmetry; selection_mode=preset_locked |
| propagation_range | PASS | PASS | PASS | PASS | PASS | PASS | PASS | planning=1.15 s; travel=1852.10 s; run=168.46s; detected_scene_type=propagation_range; selection_mode=preset_locked |
| compound_disaster | PASS | PASS | PASS | PASS | PASS | PASS | PASS | planning=1.78 s; travel=2400.10 s; run=365.95s; detected_scene_type=compound_disaster; selection_mode=preset_locked |

## Per-Scene Notes

### normal_baseline

- env panel: scene_profile=`normal_baseline`; scene_type=`normal_baseline`; tls=`static_program_0`; reroute=`disabled` (period=60s); seed=`101`; warmup=`120 s`; evaluation=`(120, 1800)`
- expected env: `tls static_program_0`, `reroute disabled` (period=60s), `seed 101`, `warmup 120 s`, `evaluation 120-1800 s`, counts={'incident': 0, 'blocked': 0, 'lane': 0}
- results: scene_profile=`normal_baseline`; scene_type=`normal_baseline`; detected_scene_type=`normal_baseline`; selection_mode=`preset_locked`; locked=True
- timing: Planning Time=`1.11 s` (model_infer_time_s=1.11 · route_solve_time_s=0.00); Travel Time=`198.40 s` (SUMO actual travel time)
- maps/defense: default={'planned_route_map_view': 'Real Map (Folium)', 'roadnet_overview_map_view': 'Real Map (Folium)'}; map switch ok=True; defense ok=True; debug expand ok=True
- binding note: 当前为显式预设：scene_profile 将锁定到该 preset，不再按文本内容重判。 | 当前 scene_profile 由预设场景锁定为 `normal_baseline`，不会再按文本内容重判。

### simple_local

- env panel: scene_profile=`simple_local`; scene_type=`simple_local`; tls=`static_program_0`; reroute=`disabled` (period=60s); seed=`111`; warmup=`180 s`; evaluation=`(180, 2100)`
- expected env: `tls static_program_0`, `reroute disabled` (period=60s), `seed 111`, `warmup 180 s`, `evaluation 180-2100 s`, counts={'incident': 2, 'blocked': 0, 'lane': 1}
- results: scene_profile=`simple_local`; scene_type=`simple_local`; detected_scene_type=`simple_local`; selection_mode=`preset_locked`; locked=True
- timing: Planning Time=`1.09 s` (model_infer_time_s=1.09 · route_solve_time_s=0.00); Travel Time=`318.70 s` (SUMO actual travel time)
- maps/defense: default={'planned_route_map_view': 'Real Map (Folium)', 'roadnet_overview_map_view': 'Real Map (Folium)'}; map switch ok=True; defense ok=True; debug expand ok=True
- binding note: 当前为显式预设：scene_profile 将锁定到该 preset，不再按文本内容重判。 | 当前 scene_profile 由预设场景锁定为 `simple_local`，不会再按文本内容重判。

### directional_asymmetry

- env panel: scene_profile=`directional_asymmetry`; scene_type=`directional_asymmetry`; tls=`static_program_0_phase_0`; reroute=`disabled` (period=60s); seed=`202`; warmup=`180 s`; evaluation=`(180, 2100)`
- expected env: `tls static_program_0_phase_0`, `reroute disabled` (period=60s), `seed 202`, `warmup 180 s`, `evaluation 180-2100 s`, counts={'incident': 7, 'blocked': 0, 'lane': 1}
- results: scene_profile=`directional_asymmetry`; scene_type=`directional_asymmetry`; detected_scene_type=`directional_asymmetry`; selection_mode=`preset_locked`; locked=True
- timing: Planning Time=`1.50 s` (model_infer_time_s=1.50 · route_solve_time_s=0.00); Travel Time=`1334.70 s` (SUMO actual travel time)
- maps/defense: default={'planned_route_map_view': 'Real Map (Folium)', 'roadnet_overview_map_view': 'Real Map (Folium)'}; map switch ok=True; defense ok=True; debug expand ok=True
- binding note: 当前为显式预设：scene_profile 将锁定到该 preset，不再按文本内容重判。 | 当前 scene_profile 由预设场景锁定为 `directional_asymmetry`，不会再按文本内容重判。

### propagation_range

- env panel: scene_profile=`propagation_range`; scene_type=`propagation_range`; tls=`static_program_0_phase_2`; reroute=`enabled` (period=45s); seed=`404`; warmup=`240 s`; evaluation=`(240, 2400)`
- expected env: `tls static_program_0_phase_2`, `reroute enabled` (period=45s), `seed 404`, `warmup 240 s`, `evaluation 240-2400 s`, counts={'incident': 3, 'blocked': 3, 'lane': 1}
- results: scene_profile=`propagation_range`; scene_type=`propagation_range`; detected_scene_type=`propagation_range`; selection_mode=`preset_locked`; locked=True
- timing: Planning Time=`1.15 s` (model_infer_time_s=1.15 · route_solve_time_s=0.00); Travel Time=`1852.10 s` (SUMO actual travel time)
- maps/defense: default={'planned_route_map_view': 'Real Map (Folium)', 'roadnet_overview_map_view': 'Real Map (Folium)'}; map switch ok=True; defense ok=True; debug expand ok=True
- binding note: 当前为显式预设：scene_profile 将锁定到该 preset，不再按文本内容重判。 | 当前 scene_profile 由预设场景锁定为 `propagation_range`，不会再按文本内容重判。

### compound_disaster

- env panel: scene_profile=`compound_disaster`; scene_type=`compound_disaster`; tls=`static_program_0_phase_2`; reroute=`enabled` (period=30s); seed=`505`; warmup=`300 s`; evaluation=`(300, 2700)`
- expected env: `tls static_program_0_phase_2`, `reroute enabled` (period=30s), `seed 505`, `warmup 300 s`, `evaluation 300-2700 s`, counts={'incident': 5, 'blocked': 4, 'lane': 2}
- results: scene_profile=`compound_disaster`; scene_type=`compound_disaster`; detected_scene_type=`compound_disaster`; selection_mode=`preset_locked`; locked=True
- timing: Planning Time=`1.78 s` (model_infer_time_s=1.78 · route_solve_time_s=0.00); Travel Time=`2400.10 s` (SUMO actual travel time)
- maps/defense: default={'planned_route_map_view': 'Real Map (Folium)', 'roadnet_overview_map_view': 'Real Map (Folium)'}; map switch ok=True; defense ok=True; debug expand ok=True
- binding note: 当前为显式预设：scene_profile 将锁定到该 preset，不再按文本内容重判。 | 当前 scene_profile 由预设场景锁定为 `compound_disaster`，不会再按文本内容重判。

## must_fix_before_defense

- none

## should_fix_if_time

- 本轮未发现新的 should_fix_if_time 级别 UI 缺陷；若答辩前还有时间，建议在真实浏览器中再手点一轮 Folium iframe 兼容性。

## acceptable

- normal_baseline：headless AppTest 中 Folium 视图会自动回退到 Plotly；UI 已给出 loading / fallback 提示，切到 Plotly 后结果保持一致。
- simple_local：headless AppTest 中 Folium 视图会自动回退到 Plotly；UI 已给出 loading / fallback 提示，切到 Plotly 后结果保持一致。
- directional_asymmetry：headless AppTest 中 Folium 视图会自动回退到 Plotly；UI 已给出 loading / fallback 提示，切到 Plotly 后结果保持一致。
- propagation_range：headless AppTest 中 Folium 视图会自动回退到 Plotly；UI 已给出 loading / fallback 提示，切到 Plotly 后结果保持一致。
- compound_disaster：headless AppTest 中 Folium 视图会自动回退到 Plotly；UI 已给出 loading / fallback 提示，切到 Plotly 后结果保持一致。
- 跨场景切换缓存保护通过：切换场景但未重新运行时，会提示旧结果已停止展示、隐藏旧原始输出，并禁用导出。

## Cross-Scene Cache Check

- dirty warning=True; raw output hidden=True; export disabled=True

## Verdict

- normal_baseline 已可从 UI 正确进入，且不再误归到 simple_local。
- 其余 4 个回归场景与跨场景切换均通过最小 smoke test。
- 结论：`UI_READY_FOR_DEFENSE`
