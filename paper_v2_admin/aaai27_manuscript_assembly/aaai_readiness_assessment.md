# AAAI-27 readiness assessment — STAG-Route

Date: 2026-06-16. Scope: honest gap analysis against AAAI main-track bar, with a prioritized
optimization plan. Companion to `next_action_decision.md` and `camera_ready_todo.md`.

## 1. Where the paper is strong (keep, lead with these)
- **Novel, crisp framing.** "Output-interface view of language-grounded routing" (dense per-edge vs
  typed sparse anchors) is a genuine reframe, not an incremental tweak. This is the paper's spine.
- **P1 representational adequacy = STRONG.** Nine-method closed-loop competitive table; STAG rank-1 on
  CS (0.96) and Avg TT; Dense-LoRA double-failure (CS 0.02 + collapse) with paired bootstrap CIs
  (excess-detour reduction 961.74s, CI excludes 0). Substrate validated (CS-0 planners).
- **Mechanistic, not just leaderboard.** L0/L1/L2 granularity isolates *typed structure* (not sparsity)
  as load-bearing; P5 field ablation (direction→localization, type→conflict); verifier normalizes not
  repairs. This is the kind of "why" AAAI reviewers reward.
- **Honesty is a feature.** Compositional-OOD frontier (45–55%) reported and reconciled; CR↔TT
  trade-off disclosed; DCRNN read by bucket with sensor-invariance ablation; scope-locked claims
  (frozen hierarchy, OC-1..6). Hard to attack on overclaim.
- **Reproducibility hygiene.** Canonical adapter pinned (c92bdc87), single canonical scorer imported
  (no reimplementation drift), method-agnostic benchmark construction.

## 2. Reviewer attack surfaces (ranked by risk to acceptance)

### R1 (highest) — Closed-loop transfer — RESOLVED 2026-06-22 (T6/T6b)
~~Closed-loop CS 0.96 is on ONE SUMO net.~~ **Done.** T6 (`results/T6_20260619T074306Z/`) ran the
faithful held-out closed-loop protocol on TWO unseen nets (M2_GridM |E|=194, M3_GridL |E|=398, n=60
each, discriminative precheck 60/60): STAG 119/120 (M2 59/60, M3 60/60; unconditional CS 0.992,
CR 0.992), Dense 0/120 (Θ(|E|) enumerate-all + truncation collapse). T6b
(`results/T6b_20260619T133300Z/`) frame-position fallout audit PASS — top-2 concentration 20%/13.3%,
an order of magnitude below the ≥90% by-construction trip-wire; CS independent of frame position. The
biggest "future work" is now headline evidence (§7.6). Supersedes the "behavioral transfer not shown"
caution in `aaai27_r1b_tier2_closed_loop/` and `aaai27_r4_cross_topology_offline/` (those predate T6).

### R2 — Not-run baselines (DQN, Graph WaveNet, LLM-A*)
Reported "not-run-with-reason." DCRNN + DynamicRouteGPT-adapted are run (good), but AAAI reviewers
often want at least one *learned-control* baseline actually executed.
- **Fix:** run DQN on the same closed loop (env build flagged in `p0_dqn_dcrnn_env_build`). One real RL
  baseline closes most of this gap. LLM-A* is the next most-expected.

### R3 — DynamicRouteGPT fairness
Backbone-matched substitution (Qwen2.5-1.5B vs original Llama3-8B) AND fine-tuned-vs-zero-shot. Doubly
caveated (footnotes a–c). Defensible but pushable.
- **Fix (optional):** add an original-scale or fine-tuned DRGPT variant, or move DRGPT fully to a
  "scoped, backbone-matched" appendix and keep the main claim on DCRNN (the structural-separation story
  is stronger and cleaner anyway).

### R4 — External validity of STC-Bench
Self-constructed benchmark, single city-grid family, Chinese road text. Reviewers question generality.
- **Fix:** (a) report results on ≥1 distinct topology family already in the freeze (M2/M3/M5 exist);
  (b) state benchmark-construction validity (method-agnostic selection, oracle audit) prominently;
  (c) commit to public release.

### R5 — Closed-loop n=150
30/bucket. Bootstrap CIs mitigate, but expanding to n=300–500 would harden headline CIs cheaply
(generation is the only cost; oracle/checker already exist).

### R6 — Unwritten core sections
Method formalization, related-work differentiation, and intro narrative are not yet drafted (only the
contribution list is frozen). For AAAI these are make-or-break for clarity and positioning.

## 3. Prioritized plan (impact × cost)

| # | action | impact | cost | gate |
|---|---|---|---|---|
| P0 | **Camera-ready full-frame** (L0 + backbone 2000) — IN PROGRESS | locks table numbers | ~5–6h GPU | running now |
| P0 | Write method + related-work + intro narrative (into skeleton TODO stubs) | clarity/positioning | writing | after skeleton eyeballed |
| **P1** | **Tier-2 closed-loop on ≥2 held-out nets** (R1) | **highest acceptance lift** | net build + GPU | mapping contract ready |
| P2 | Run DQN closed-loop baseline (R2) | removes "no RL baseline" | env build + GPU | env scaffold exists |
| P2 | Expand closed-loop to n≈300 (R5) | harden CIs | GPU only | checker exists |
| P3 | Distinct-topology-family offline results (R4) | external validity | GPU only | M2/M3/M5 in freeze |
| P3 | LLM-A* baseline (R2) / DRGPT scale variant (R3) | completeness | GPU + eng | optional |

## 4. Honest verdict
The paper is **methodologically sound and well-scoped** — the framing, mechanism story, and honesty
are above the bar. What separates "solid workshop / borderline main-track" from "confident main-track
accept" is **R1 (Tier-2 closed-loop transfer)** and **R2 (one real RL baseline)**. Neither changes any
existing claim; both convert current "future work" into evidence. If the calendar only allows one, do
**Tier-2**. Everything else (R3–R5) is hardening, not gating.
