# UI Smoke Test Report

- Date: 2026-04-09 UTC
- App: `/root/autodl-tmp/code/app_fixed.py`
- Method: `streamlit.testing.v1.AppTest` headless smoke test + cross-scene stale-cache check
- Covered scenes: normal_baseline, simple_local, directional_asymmetry, propagation_range, compound_disaster

## Scenario Summary

| scene | access | scene_profile refresh | env params | model result | time metrics | map switch | defense mode | notes |
|---|---|---|---|---|---|---|---|---|
| normal_baseline | FAIL | FAIL | FAIL | PASS | PASS | PASS | FAIL | planning=1.06 s; travel=266.50 s; run=11.24s; map-rerender=0.76s; displayed scene_profile=simple_local |
| simple_local | PASS | PASS | PASS | PASS | PASS | PASS | PASS | planning=1.09 s; travel=318.70 s; run=9.10s; map-rerender=0.52s |
| directional_asymmetry | PASS | PASS | PASS | PASS | PASS | PASS | PASS | planning=1.59 s; travel=1334.70 s; run=38.89s; map-rerender=0.56s |
| propagation_range | PASS | PASS | PASS | PASS | PASS | PASS | PASS | planning=1.11 s; travel=1852.10 s; run=160.54s; map-rerender=1.00s |
| compound_disaster | PASS | PASS | PASS | PASS | PASS | PASS | PASS | planning=1.83 s; travel=2400.10 s; run=360.17s; map-rerender=0.56s |

## Per-Scene Notes

### normal_baseline

- scene_profile=`simple_local`; tls=`static_program_0`; reroute=`disabled` (period=60s); seed=`111`; warmup=`180 s`; evaluation=`(180, 2100)`
- expected env=`tls static_program_0`, `reroute disabled` (period=60s), `seed 101`, `warmup 120 s`, `evaluation 120-1800 s`
- incident / blocked / lane summary=`2 条: R3C0_E=severe, R3C1_E=heavy` / `0 条` / `1 条: R3C0_E→1`; expected counts={'incident': 0, 'blocked': 0, 'lane': 0}
- model result cards ok=True; route cards ok=True; Planning Time=`1.06 s`; Travel Time=`266.50 s`
- map default={'planned_route_map_view': 'Real Map (Folium)', 'roadnet_overview_map_view': 'Real Map (Folium)'}; Folium warning before switch=True; Plotly switch ok=True; Defense Mode ok=False; Debug expand ok=True

### simple_local

- scene_profile=`simple_local`; tls=`static_program_0`; reroute=`disabled` (period=60s); seed=`111`; warmup=`180 s`; evaluation=`(180, 2100)`
- expected env=`tls static_program_0`, `reroute disabled` (period=60s), `seed 111`, `warmup 180 s`, `evaluation 180-2100 s`
- incident / blocked / lane summary=`2 条: R3C0_E=severe, R3C1_E=heavy` / `0 条` / `1 条: R3C0_E→1`; expected counts={'incident': 2, 'blocked': 0, 'lane': 1}
- model result cards ok=True; route cards ok=True; Planning Time=`1.09 s`; Travel Time=`318.70 s`
- map default={'planned_route_map_view': 'Real Map (Folium)', 'roadnet_overview_map_view': 'Real Map (Folium)'}; Folium warning before switch=True; Plotly switch ok=True; Defense Mode ok=True; Debug expand ok=True

### directional_asymmetry

- scene_profile=`directional_asymmetry`; tls=`static_program_0_phase_0`; reroute=`disabled` (period=60s); seed=`202`; warmup=`180 s`; evaluation=`(180, 2100)`
- expected env=`tls static_program_0_phase_0`, `reroute disabled` (period=60s), `seed 202`, `warmup 180 s`, `evaluation 180-2100 s`
- incident / blocked / lane summary=`7 条: R4C0_E=severe, R4C1_E=severe, R4C2_E=heavy, R4C3_E=heavy, ... (+3)` / `0 条` / `1 条: R4C1_E→1`; expected counts={'incident': 7, 'blocked': 0, 'lane': 1}
- model result cards ok=True; route cards ok=True; Planning Time=`1.59 s`; Travel Time=`1334.70 s`
- map default={'planned_route_map_view': 'Real Map (Folium)', 'roadnet_overview_map_view': 'Real Map (Folium)'}; Folium warning before switch=True; Plotly switch ok=True; Defense Mode ok=True; Debug expand ok=True

### propagation_range

- scene_profile=`propagation_range`; tls=`static_program_0_phase_2`; reroute=`enabled` (period=45s); seed=`404`; warmup=`240 s`; evaluation=`(240, 2400)`
- expected env=`tls static_program_0_phase_2`, `reroute enabled` (period=45s), `seed 404`, `warmup 240 s`, `evaluation 240-2400 s`
- incident / blocked / lane summary=`3 条: C4R1_N=moderate, C4R2_N=moderate, R1C2_E=light` / `3 条: C2R0_N, C2R1_N, C2R2_N` / `1 条: C4R1_N→1`; expected counts={'incident': 3, 'blocked': 3, 'lane': 1}
- model result cards ok=True; route cards ok=True; Planning Time=`1.11 s`; Travel Time=`1852.10 s`
- map default={'planned_route_map_view': 'Real Map (Folium)', 'roadnet_overview_map_view': 'Real Map (Folium)'}; Folium warning before switch=True; Plotly switch ok=True; Defense Mode ok=True; Debug expand ok=True

### compound_disaster

- scene_profile=`compound_disaster`; tls=`static_program_0_phase_2`; reroute=`enabled` (period=30s); seed=`505`; warmup=`300 s`; evaluation=`(300, 2700)`
- expected env=`tls static_program_0_phase_2`, `reroute enabled` (period=30s), `seed 505`, `warmup 300 s`, `evaluation 300-2700 s`
- incident / blocked / lane summary=`5 条: R3C0_E=heavy, R3C1_E=heavy, C4R2_N=moderate, C1R1_S=severe, ... (+1)` / `4 条: C2R1_N, C2R1_S, C2R2_N, C2R2_S` / `2 条: C4R1_N→1, R3C2_E→1`; expected counts={'incident': 5, 'blocked': 4, 'lane': 2}
- model result cards ok=True; route cards ok=True; Planning Time=`1.83 s`; Travel Time=`2400.10 s`
- map default={'planned_route_map_view': 'Real Map (Folium)', 'roadnet_overview_map_view': 'Real Map (Folium)'}; Folium warning before switch=True; Plotly switch ok=True; Defense Mode ok=True; Debug expand ok=True

## must_fix_before_defense

- normal_baseline 目前不能从 UI 直接选到。当前场景列表没有 baseline 预设；用自定义“正常通行”文本测试时，scene_profile 仍刷新成 simple_local，环境参数也落到了 simple_local（seed=111 / warmup=180 / eval=180-2100），而不是 normal_baseline（seed=101 / warmup=120 / eval=120-1800）。

## should_fix_if_time

- 本轮 headless smoke test 未发现新的 should_fix_if_time 级别 UI 缺陷；若答辩前还有时间，建议补一轮真实浏览器手动点检 Folium iframe 的现场兼容性。

## acceptable

- simple_local：在 headless AppTest 中，默认 Folium 视图会出现组件序列化警告并自动回退到 Plotly；UI 已给出 loading / fallback 提示，手动切到 Plotly 后告警消失，Planning/Travel 结果保持一致。
- directional_asymmetry：在 headless AppTest 中，默认 Folium 视图会出现组件序列化警告并自动回退到 Plotly；UI 已给出 loading / fallback 提示，手动切到 Plotly 后告警消失，Planning/Travel 结果保持一致。
- propagation_range：在 headless AppTest 中，默认 Folium 视图会出现组件序列化警告并自动回退到 Plotly；UI 已给出 loading / fallback 提示，手动切到 Plotly 后告警消失，Planning/Travel 结果保持一致。
- compound_disaster：在 headless AppTest 中，默认 Folium 视图会出现组件序列化警告并自动回退到 Plotly；UI 已给出 loading / fallback 提示，手动切到 Plotly 后告警消失，Planning/Travel 结果保持一致。
- propagation_range / compound_disaster：reroute 场景在终端日志里出现了 SUMO `Route replacement failed` 告警，但 AppTest 未把它传播成页面异常；结果卡片、时间指标、地图切换和 Defense Mode 都正常通过。本轮按 UI 层面记为 acceptable，若答辩时会同时投屏终端，可考虑额外压低这类日志噪声。
- 跨场景切换的缓存保护正常：切到新场景但未重新运行时，会提示上一轮缓存结果已停止展示、隐藏旧原始输出，并禁用导出按钮。

## Cross-Scene Cache Check

- dirty warning=True; raw output hidden=True; export disabled=True

## Verdict

- UI 还没有完全达到答辩展示标准。
- 主要阻塞项是 `normal_baseline` 无法从当前 UI 正确进入；其余 4 个目标场景在结果区、时间口径、地图主备切换和 Defense Mode 上均通过 smoke test。
