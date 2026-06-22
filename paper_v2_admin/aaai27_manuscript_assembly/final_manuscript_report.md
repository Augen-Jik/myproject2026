# final_manuscript_report.md — FINAL_MANUSCRIPT_EDITS_LIMITATIONS_SWAP_AND_INTEGRITY

Date: 2026-06-19. Task class: pure text assembly + reference verification. No experiments, no
recomputed numbers, no frozen scope/number edits. Working dir: `/root/autodl-tmp/`.

## A. Limitations swap — DONE

The skeleton inputs limitations via `section7_assembled.tex` →
`../aaai27_section7_abstract_intro_frozen/limitations.tex`. The closed-loop paragraph in that file
was the OLD open-future-work version ("Closed-loop transfer is single-network (Tier-2 not run)… held-out
nets are not yet built"). It was **swapped** for the R1-b-synced attempted+diagnosed version from
`../aaai27_r1b_fallout_audit/limitations_synced.tex`:

- New paragraph = "Closed-loop evaluation is single-network, by diagnosis not omission" (built Grid-M/
  Grid-L, 100% mapping, smoke-confirmed; transfer ill-posed because held-out GT edge frame collapses
  per-road identity >90% to a single row/col; perfect-predictor control proves harness fair; held-out
  closed-loop transfer scoped open + methodological observation on discriminative benchmark scarcity).
- Old open wording fully removed (grep for "not yet built / Tier-2 not run / nets are not" → 0 hits).
- R1-b closed-loop numbers kept **internal-only**, not written into the main text.
- The frozen P3-boundary "behavioral transfer … remain open" lines in §7.6 / intro are intentional
  OC-6 verbatim scope statements and are consistent with the synced limitation; left untouched.

## B. DQN row + footnote (f) + three-method narrative — DONE

- DQN row 9 (`DQN$^{f}$ … 0.660 / 373.7 / 86.81 / 2.580 / 1.000$^{f}$`) and footnote (f) already ship
  in `competitive_main_table_final.tex`, which §7.4 inputs → so they are in the skeleton.
- Footnote (f) carries: sensing-fidelity, effective CR×CS=0.66, and the DQN B3/B4 bucket sub-table
  (0.27/0.23 completion) — present in the table source (companion block).
- §7.4 narrative previously discussed only Dense + DCRNN on B3/B4. Added one paragraph
  ("Three non-language methods, the same compositional failure") binding Dense (all-or-nothing),
  DCRNN (sensor-invariant 0.70/0.60), and DQN (CR 0.27/0.23, effective 0.66) into the visible
  narrative. All DQN figures reused verbatim from the frozen footnote (f); no number invented.
- Corrected a STALE line in limitations that said "DQN … reported as not-run-with-reason" (DQN was
  run): now DQN is described as run with its caveat; only Graph WaveNet + LLM-A* remain not-run.

## C. Intro centerline — SKIPPED_FOR_SCOPE

The output-interface-as-design-axis + typed-structure-load-bearing centerline is already present and
freeze-vetted in `introduction_narrative.tex` ("Our reframe: grounding as a choice of output
interface" + the $\Theta(k)$ vs $\Theta(|E|)$ checkable framing) and in contribution 1. Injecting an
additional sentence would duplicate vetted content and edit a frozen file without an authorized
wording delta. Per the task's conditional ("only if consistent … else SKIPPED_FOR_SCOPE"), recorded
as SKIPPED_FOR_SCOPE — the centerline exists and is consistent with the frozen hierarchy (H1).

## D. Final integrity grep — PASS

broken refs 0 / footnote collisions 0 / archived paths 0 / stale comments 0. Abstract numbers all
sourced. Details in `final_integrity_grep.md`.

## E. Bib

7 cited keys, all present in `references.bib`. NEEDS_VERIFICATION count = 3 (Mnih/DQN not-yet-cited;
DRGPT author-attribution mismatch; LLM-A* arXiv id). See `bib_needs_verification_list.md`.

## F. Compile — SKIPPED

No TeX engine on PATH (`pdflatex`, `xelatex`, `lualatex`, `latex`, `latexmk`, `tectonic`, `bibtex`
all absent). Static checks substituted: input-chain resolution, ref/label closure, cite/bib closure,
balanced-brace spot check on edited files. No unresolved references found statically.

## Findings flagged for a human scope-owner (NOT changed — out of mandate)

1. **"eight methods" vs 9-row table. — RESOLVED 2026-06-22.** Submission prose now reads "nine
   methods" everywhere (abstract L12, contributions/intro L15, §7.4 L15/L22); `grep -i eight` over
   the submission surface returns zero. Remaining "eight/8-method" mentions are in admin notes only
   (this report, readiness assessment), not in the manuscript.
2. **DRGPT attribution mismatch.** Table cell says "Sun et al. 2024"; bib `drgpt2024` lists
   "Zhou, Zhou, Liu". Reconcile attribution at camera-ready.
3. **Footnote text delivery.** Footnote a–f text + bucket sub-tables live in commented companion
   blocks of the table source (frozen author's design); the rendered output relies on prose to carry
   them. §7.4 now renders the DQN portion in prose; the DCRNN/DRGPT footnote text is still
   comment-only by the original design.

## Files modified

- `aaai27_section7_abstract_intro_frozen/limitations.tex` (closed-loop swap + DQN not-run correction)
- `aaai27_section7_abstract_intro_frozen/section_7_4.tex` (three-method paragraph; footnote (f) note)
- `aaai27_manuscript_assembly/manuscript_skeleton.tex` (header provenance note)
