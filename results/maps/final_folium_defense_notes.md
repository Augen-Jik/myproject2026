# Final Folium Defense Notes

## Search Result
- Reused Folium core from `/root/autodl-tmp/code/viz.py`.
- UI integration points found in `/root/autodl-tmp/code/app_fixed.py`.
- Historical HTML reference found at `/root/autodl-tmp/results/codex_folium_smoke.html`.

## Final Defense View
- Output HTML: `/root/autodl-tmp/results/maps/final_folium_defense_view.html`
- Scenario used for acceptance: `场景 2 / 花园路施工封闭`
- `analytical_travel_time_s_legacy` under scenario truth: `12.6` s
- Key highlighted edges: `C4R0_N, C4R0_S, C4R1_N, C4R1_S, C4R2_N, C4R2_S, C4R3_N, C4R3_S`

## Map Configuration
- Map center point: `34.782135, 113.672960`
- Initial zoom level: `14`
- Final viewport behavior: `fit_bounds` to `((34.76971, 113.6635699), (34.79456, 113.69699))` with OSM reference bounds `(34.76971, 113.64893, 34.79456, 113.69699)`

## Acceptance Judgment
- 是否大致落在郑州金水区: `是`。当前中心点约为 `34.782135, 113.672960`，路名也对应农业路、黄河路、花园路、未来路等金水区语境。
- 是否存在明显偏移/镜像/方向颠倒: `未见明显整体偏移、镜像或整幅方向颠倒`。SUMO 边与本地 OSM 参考道路可大致重合，答辩展示层面可接受。
- 备注: 上述 `analytical_travel_time_s_legacy` 是历史解析代理，不是 live SUMO `Travel Time (s)`，不建议作为论文/答辩主证据引用。edge id 中的 `E/W/N/S` 更接近抽象网格编码，不建议在答辩时把它直接解释为严格罗盘方向；本次最终图已优先展示道路名、路段和权重信息。
- 是否需要进一步修正: `否`。

No further alignment fix needed before defense.
