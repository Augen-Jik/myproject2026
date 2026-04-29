# stage4c Remaining Repair Priority

## Final Recommendation
- `DO_ONE_LAST_MICRO_PATCH`
- Rationale: There is exactly one concentrated bucket still worth a final micro patch: single-edge `LEVEL=畅通` no-anchor failures in short/plain or prefixed wrappers. Everything else is already in range-boundary or low-ROI territory.

## What To Fix If Only One Last Micro Patch Is Allowed
- Best bucket: `single_edge_changed_relief__short_plain_or_prefixed`
- Why this one: it covers `18` cases, or `36.00%` of all remaining hard cases, with an upper bound of `17.14` percentage points on held-out `simple_local` strict-fail cases if fully fixed.
- Scope discipline: keep it strictly on `single-edge + LEVEL=畅通 + phrase in {畅通无阻, 车流顺畅/顺畅} + structure in {short_plain, prefixed_short}`, and add explicit hard negatives for `交通基本正常` / `保持正常通行`.

## Buckets That Should Stop Now
- Low ROI buckets: `relief_normal_short_range__no_anchor`, `anomaly_plus_normal_side_event__3plus_edge_range_boundary`, `single_edge_changed_relief__planner_long_tail`, `multi_edge_relief__no_anchor`.
- Risky buckets that are not suitable for a tiny last patch: `anomaly_plus_normal_side_event__2edge_range_boundary`.

## Paper-Closure Readout
- Current state: remaining hard cases=`50`, remaining single-edge=`19`, remaining no-anchor=`28`, remaining anchor-but-range-incomplete=`22`.
- Upstream baselines are already stable enough for paper closure on everything except the one concentrated single-edge relief bucket. If that bucket does not move immediately after one micro patch, the next action should be to freeze and write up.

## Why The Ranking Looks Like This
- Policy support for relief-focused last patch: `main_gap_mentions_changtong_shunchang=True`, `multi_edge_must_cover_full_range=True`.
- Collateral risk already visible: normal-only false-positive anchors=`8` with phrase mix `{'交通基本正常': 6, '保持正常通行': 2}`.
- changed-edge denominator inferred from postmortem buckets: `105` held-out `val_normal/simple_local` changed samples.
